from __future__ import annotations

import os
import inspect
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from file_handlers.mdf.shader_types import material_shader_names
from file_handlers.mesh.mesh_file import (
    MeshDependencyMissingError,
    MeshFile,
    MeshMainVersion,
    _decode_skin_weights,
    get_mesh_version,
)
from file_handlers.mesh.material_resolver import MdfSurfaceProfile, MeshMaterialResolver
from ui.scene.gpu_skinning import GpuSkinningDeformer
from ui.scene.mesh_scene import _merge_tangent_frames
from ui.scene.scene_buffers import build_scene_buffer_set
from ui.scene.scene_model import SceneDrawMesh
from ui.scene.scene_preview import ScenePreviewWidget
from ui.scene.studio_material import StudioMaterialRenderer


class TestWotsMesh(unittest.TestCase):
    def test_item_material_candidate_supports_wots_a_suffix(self):
        candidates = list(
            MeshMaterialResolver.iter_mdf_candidates(
                "natives/stm/art/model/item/it0/it000_0000/"
                "it000_0000_00.mesh.260209350"
            )
        )
        self.assertEqual(
            candidates[:2],
            [
                "natives/stm/art/model/item/it0/it000_0000/"
                "it000_0000_00.mdf2",
                "natives/stm/art/model/item/it0/it000_0000/"
                "it000_0000_00_a.mdf2",
            ],
        )

    def test_preview_scene_preserves_secondary_uvs(self):
        mesh = SceneDrawMesh(
            key="face",
            vertices=np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
            ),
            indices=np.asarray([0, 1, 2], dtype=np.uint32),
            color=(1.0, 1.0, 1.0, 1.0),
            uvs=np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
            uvs1=np.asarray(
                [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]],
                dtype=np.float32,
            ),
        )
        buffers = build_scene_buffer_set(
            [mesh], set(), show_only_highlighted=False, force_solid=False
        )
        self.assertIsNotNone(buffers)
        np.testing.assert_allclose(buffers.uvs1, mesh.uvs1)

    def test_preview_tangent_frame_preserves_signed_handedness(self):
        payload = SimpleNamespace(
            tangents=[1.0, 0.0, 0.0],
            tangent_ws=[128],
        )
        tangents = _merge_tangent_frames(
            [(0, payload, 1)],
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(tangents, [[1.0, 0.0, 0.0, -1.0]])

    def test_wots_material_shader_names_do_not_use_legacy_indices(self):
        shader_names = material_shader_names("onimusha_wots")
        self.assertEqual(shader_names[16], "ExpensiveTransparent")
        self.assertEqual(shader_names[19], "PostProcess")
        self.assertEqual(shader_names[34], "EyeballPostProcess")

    def test_preview_material_arguments_match_both_renderers(self):
        preview = SimpleNamespace(
            _material_profiles={
                "face": MdfSurfaceProfile(
                    material_name="face",
                    game_version="ONIWOTS",
                    mmtr_path=(
                        "MaterialShader/Master/Character/Variation/PL/"
                        "V_Chara_PL_Face.mmtr"
                    ),
                    parameters={
                        "UseDetail": (1.0,),
                        "SSSScale": (0.5,),
                        "Dummy_UVScale": (0.5,),
                    },
                )
            },
            _texture_ids={},
        )
        arguments = ScenePreviewWidget._wots_material_inputs(preview, "face")
        self.assertTrue(arguments["use_detail"])
        self.assertEqual(arguments["sss_scale"], 0.5)
        self.assertIn("detail_nrrc_texture_id", arguments)
        self.assertIn("stcm_texture_id", arguments)
        self.assertFalse(arguments["use_secondary_uv"])
        self.assertEqual(arguments["face_uv_scale"], 0.5)
        for renderer in (StudioMaterialRenderer.bind, GpuSkinningDeformer.bind):
            self.assertLessEqual(
                set(arguments),
                set(inspect.signature(renderer).parameters),
            )

    def test_version_pair(self):
        self.assertEqual(
            get_mesh_version(250203152, 260209350),
            MeshMainVersion.ONIMUSHA_WOTS_1010,
        )

    def test_six_packed_indices_and_weights(self):
        indices = (3, 17, 511, 9, 1023, 42)
        first = indices[0] | (indices[1] << 10) | (indices[2] << 20)
        second = indices[3] | (indices[4] << 10) | (indices[5] << 20)
        raw = struct.pack("<II8B", first, second, 90, 60, 45, 30, 20, 10, 0, 0)
        decoded = _decode_skin_weights(
            memoryview(raw), 1, MeshMainVersion.ONIMUSHA_WOTS_1010
        )
        self.assertEqual(list(decoded.deform_indices), list(indices))
        np.testing.assert_allclose(
            decoded.weights,
            np.asarray([90, 60, 45, 30, 20, 10], dtype=np.float32) / 255.0,
        )

    def test_wots_face_stream_uses_marked_eight_influence_layout(self):
        indices = (238, 52, 34, 207, 142, 12, 64, 203)
        raw_weights = (48, 45, 33, 29, 29, 17, 12, 11)
        raw = struct.pack("<8B8B", *indices, *raw_weights)
        decoded = _decode_skin_weights(
            memoryview(raw), 1, MeshMainVersion.ONIMUSHA_WOTS_1010
        )
        self.assertEqual(decoded.influence_count, 8)
        self.assertEqual(list(decoded.deform_indices), list(indices))
        np.testing.assert_allclose(
            decoded.weights,
            np.asarray(raw_weights, dtype=np.float32) / 255.0,
        )

    def test_wots_packed_hair_stream_ignores_high_marker_bits(self):
        indices = (102, 103, 104, 105, 106, 107)
        first = (
            indices[0]
            | (indices[1] << 10)
            | (indices[2] << 20)
            | (3 << 30)
        )
        second = (
            indices[3]
            | (indices[4] << 10)
            | (indices[5] << 20)
            | (3 << 30)
        )
        raw = struct.pack(
            "<II8B", first, second, 90, 60, 45, 30, 20, 10, 0, 0
        )
        decoded = _decode_skin_weights(
            memoryview(raw),
            1,
            MeshMainVersion.ONIMUSHA_WOTS_1010,
            deform_limit=124,
        )
        self.assertEqual(decoded.influence_count, 6)
        self.assertEqual(list(decoded.deform_indices), list(indices))

    def test_six_weight_tail_is_diagnosed(self):
        raw = struct.pack("<II8B", 0, 0, 255, 0, 0, 0, 0, 0, 1, 0)
        with self.assertRaisesRegex(ValueError, "tail bytes"):
            _decode_skin_weights(memoryview(raw), 1, MeshMainVersion.ONIMUSHA_WOTS_1010)


@unittest.skipUnless(
    os.environ.get("REASY_WOTS_ASSET_ROOT"), "set REASY_WOTS_ASSET_ROOT"
)
class TestWotsMeshAssets(unittest.TestCase):
    def test_all_mesh_assets_parse_or_report_missing_streaming(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        meshes = list(root.rglob("*.mesh.260209350"))
        self.assertTrue(meshes)
        for path in meshes:
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0].lower() == "streaming":
                continue
            stream = root / "streaming" / relative
            streaming_data = stream.read_bytes() if stream.is_file() else None
            mesh = MeshFile()
            try:
                mesh.read(
                    path.read_bytes(),
                    file_version=260209350,
                    streaming_data=streaming_data,
                )
            except MeshDependencyMissingError:
                self.assertIsNone(streaming_data, str(path))
                continue
            self.assertIsNotNone(mesh.mesh_buffer, str(path))
            remap_count = len(mesh.bone_remap_indices)
            for payload in mesh.mesh_buffer.buffer_payloads.values():
                for weights in (payload.skin_weights, payload.extra_skin_weights):
                    if weights is None:
                        continue
                    active = np.asarray(weights.weights) > 0
                    indices = np.asarray(weights.deform_indices)
                    self.assertTrue(np.all(indices[active] < remap_count), str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
