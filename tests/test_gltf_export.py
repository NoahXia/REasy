from __future__ import annotations

import json
import struct
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from file_handlers.gltf_export import (
    GltfMaterialAsset,
    GltfTextureImage,
    _combine_base_alpha,
    _normal_from_packed,
    _orm_from_wots,
    export_gltf,
)
from file_handlers.motion.evaluation.model import Rig, RigJoint, Transform
from file_handlers.motion.evaluation.profiles import WOTS_EVALUATION_PROFILE
from file_handlers.motion.evaluation.source_adapter import rig_from_motion_skeleton
from file_handlers.motion.mot.model import (
    AnimationNode,
    Joint,
    KeyTrack,
    Motion,
    Skeleton,
    TrackFamily,
)


def _read_glb(path: Path):
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data)
    if (magic, version, total) != (b"glTF", 2, len(data)):
        raise AssertionError("invalid GLB header")
    json_size, json_kind = struct.unpack_from("<I4s", data, 12)
    if json_kind != b"JSON":
        raise AssertionError("missing JSON chunk")
    document = json.loads(data[20 : 20 + json_size])
    binary_header = 20 + json_size
    binary_size, binary_kind = struct.unpack_from("<I4s", data, binary_header)
    if binary_kind != b"BIN\0":
        raise AssertionError("missing BIN chunk")
    binary = data[binary_header + 8 : binary_header + 8 + binary_size]
    return document, binary


class TestGltfExport(unittest.TestCase):
    @staticmethod
    def _triangle_payload(*, skin_weights=None):
        return SimpleNamespace(
            positions=array("f", [0, 0, 1, 1, 0, 1, 0, 1, 1]),
            normals=array("f", [0, 0, 1] * 3),
            uv0=array("d", [0, 0, 1, 0, 0, 1]),
            colors=array("B", [255, 255, 255, 255] * 3),
            faces=array("H", [0, 1, 2]),
            integer_faces=None,
            skin_weights=skin_weights,
            extra_skin_weights=None,
        )

    @staticmethod
    def _triangle_mesh(payload, *, joint_count=0, remap=()):
        submesh = SimpleNamespace(
            buffer_index=0,
            verts_index_offset=0,
            vert_count=3,
            faces_index_offset=0,
            indices_count=3,
            material_index=0,
        )
        return SimpleNamespace(
            mesh_buffer=SimpleNamespace(buffer_payloads={0: payload}),
            meshes=[
                SimpleNamespace(
                    lods=[
                        SimpleNamespace(
                            mesh_groups=[
                                SimpleNamespace(
                                    group_id=0,
                                    submeshes=[submesh],
                                )
                            ]
                        )
                    ]
                )
            ],
            material_names=["body"],
            joint_count=joint_count,
            bone_remap_indices=list(remap),
        )

    def test_lod0_geometry_and_winding(self):
        mesh = self._triangle_mesh(self._triangle_payload())
        with tempfile.TemporaryDirectory() as directory:
            path = export_gltf(Path(directory) / "model.glb", mesh=mesh)
            document, binary = _read_glb(path)
        primitive = document["meshes"][0]["primitives"][0]
        self.assertEqual(document["materials"][0]["name"], "body")
        index_accessor = document["accessors"][primitive["indices"]]
        view = document["bufferViews"][index_accessor["bufferView"]]
        indices = struct.unpack_from("<3I", binary, view["byteOffset"])
        self.assertEqual(indices, (0, 2, 1))
        position_accessor = document["accessors"][primitive["attributes"]["POSITION"]]
        position_view = document["bufferViews"][position_accessor["bufferView"]]
        self.assertEqual(
            struct.unpack_from("<3f", binary, position_view["byteOffset"]),
            (0.0, 0.0, -1.0),
        )

    def test_embeds_pbr_material_textures_in_glb(self):
        mesh = self._triangle_mesh(self._triangle_payload())
        pixel = GltfTextureImage(1, 1, bytes((10, 20, 30, 40)), "source.tex")
        material = GltfMaterialAsset(
            name="body",
            base_color_factor=(0.5, 0.6, 0.7, 0.8),
            metallic_factor=0.0,
            roughness_factor=1.0,
            double_sided=True,
            alpha_mode="MASK",
            alpha_cutoff=0.25,
            base_color=pixel,
            normal=pixel,
            orm=pixel,
            extras={"REasy": {"mmtrPath": "material/test.mmtr"}},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = export_gltf(
                Path(directory) / "material.glb",
                mesh=mesh,
                material_assets={"body": material},
            )
            document, binary = _read_glb(path)

        exported = document["materials"][0]
        pbr = exported["pbrMetallicRoughness"]
        self.assertEqual(pbr["baseColorFactor"], [0.5, 0.6, 0.7, 0.8])
        self.assertIn("baseColorTexture", pbr)
        self.assertIn("metallicRoughnessTexture", pbr)
        self.assertIn("normalTexture", exported)
        self.assertIn("occlusionTexture", exported)
        self.assertTrue(exported["doubleSided"])
        self.assertEqual(exported["alphaMode"], "MASK")
        self.assertEqual(exported["alphaCutoff"], 0.25)
        self.assertEqual(exported["extras"]["REasy"]["mmtrPath"], "material/test.mmtr")
        self.assertEqual(len(document["images"]), 3)
        for image in document["images"]:
            view = document["bufferViews"][image["bufferView"]]
            start = view["byteOffset"]
            self.assertEqual(binary[start : start + 8], b"\x89PNG\r\n\x1a\n")

    def test_wots_packed_texture_conversion(self):
        base = GltfTextureImage(1, 1, bytes((20, 40, 60, 255)))
        alpha = GltfTextureImage(1, 1, bytes((128, 0, 0, 255)))
        combined = _combine_base_alpha(base, alpha, 1.0)
        self.assertEqual(combined.rgba[:3], bytes((20, 40, 60)))
        self.assertEqual(combined.rgba[3], 128)

        # WOTS NRRO: R=roughness, G=normal Y, B=occlusion, A=normal X.
        nrro = GltfTextureImage(1, 1, bytes((80, 128, 200, 128)))
        normal = _normal_from_packed(nrro)
        self.assertEqual(tuple(normal.rgba), (128, 128, 255, 255))
        orm = _orm_from_wots(nrro, None, 0.5, 1.0)
        self.assertEqual(tuple(orm.rgba), (200, 40, 0, 255))

        rcto = GltfTextureImage(1, 1, bytes((100, 20, 30, 220)))
        rcto_orm = _orm_from_wots(None, rcto, 0.5, 1.0)
        self.assertEqual(tuple(rcto_orm.rgba), (220, 50, 0, 255))

    def test_alpha_mask_is_resampled_to_base_color_size(self):
        base = GltfTextureImage(2, 2, bytes((10, 20, 30, 255)) * 4)
        alpha = GltfTextureImage(1, 1, bytes((64, 0, 0, 255)))
        combined = _combine_base_alpha(base, alpha, 1.0)
        pixels = np.frombuffer(combined.rgba, dtype=np.uint8).reshape(-1, 4)
        self.assertEqual(pixels[:, 3].tolist(), [64, 64, 64, 64])

    def test_skin_and_json_sidecar_export(self):
        influences = SimpleNamespace(
            influence_count=6,
            deform_indices=array(
                "I",
                [0, 1, 0, 0, 0, 0] * 3,
            ),
            weights=array(
                "f",
                [0.75, 0.25, 0.0, 0.0, 0.0, 0.0] * 3,
            ),
        )
        mesh = self._triangle_mesh(
            self._triangle_payload(skin_weights=influences),
            joint_count=2,
            remap=(0, 1),
        )
        rig = Rig(
            [
                RigJoint("root", rest=Transform()),
                RigJoint(
                    "child",
                    parent_index=0,
                    rest=Transform(translation=(0.0, 0.0, 1.0)),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = export_gltf(Path(directory) / "model.gltf", mesh=mesh, rig=rig)
            document = json.loads(path.read_text(encoding="utf-8"))
            binary = path.with_suffix(".bin").read_bytes()

        primitive = document["meshes"][0]["primitives"][0]
        self.assertIn("JOINTS_0", primitive["attributes"])
        self.assertIn("WEIGHTS_0", primitive["attributes"])
        self.assertIn("JOINTS_1", primitive["attributes"])
        self.assertIn("WEIGHTS_1", primitive["attributes"])
        self.assertEqual(len(document["skins"]), 1)
        inverse = document["accessors"][document["skins"][0]["inverseBindMatrices"]]
        self.assertEqual((inverse["type"], inverse["count"]), ("MAT4", 2))
        self.assertEqual(document["buffers"][0]["uri"], "model.bin")
        self.assertEqual(document["buffers"][0]["byteLength"], len(binary))

        weight_accessor = document["accessors"][primitive["attributes"]["WEIGHTS_0"]]
        weight_view = document["bufferViews"][weight_accessor["bufferView"]]
        weights = np.frombuffer(
            binary,
            dtype="<f4",
            count=weight_accessor["count"] * 4,
            offset=weight_view["byteOffset"],
        ).reshape(-1, 4)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    def test_animation_is_baked_at_sixty_fps(self):
        joint = Joint("root")
        motion = Motion(
            "move",
            end_frame=2.0,
            skeleton=Skeleton([joint]),
            animation_nodes=[
                AnimationNode(
                    joint,
                    translation=KeyTrack(
                        TrackFamily.VECTOR3, [0, 2], [(0, 0, 0), (0, 0, 2)]
                    ),
                )
            ],
        )
        rig = rig_from_motion_skeleton(motion, scale=(1, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = export_gltf(
                Path(directory) / "animation.glb",
                rig=rig,
                motion=motion,
                evaluation_profile=WOTS_EVALUATION_PROFILE,
            )
            document, binary = _read_glb(path)
        animation = document["animations"][0]
        time_accessor = document["accessors"][animation["samplers"][0]["input"]]
        self.assertEqual(time_accessor["count"], 3)
        self.assertAlmostEqual(time_accessor["max"][0], 2.0 / 60.0, places=7)
        for accessor in document["accessors"]:
            view = document["bufferViews"][accessor["bufferView"]]
            self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(binary))


if __name__ == "__main__":
    unittest.main(verbosity=2)
