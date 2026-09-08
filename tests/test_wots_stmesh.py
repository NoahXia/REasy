from __future__ import annotations

import os
import struct
import unittest
from pathlib import Path

import numpy as np

from file_handlers.mesh.material_resolver import MeshMaterialResolver
from file_handlers.mesh.stmesh_file import StMeshFile, StMeshParseError
from ui.scene.mesh_scene import build_mesh_scene


def _minimal_stmesh() -> bytes:
    metadata_size = 0x20
    locator = metadata_size + 0x1C0
    geometry = 0x200
    lod_table = 0x30
    block = 0x40
    block_size = 32 + 4 + 3 * 8 + 3 * 4 + 3 * 4
    data = bytearray(geometry + block + block_size)
    data[:4] = b"STM8"
    struct.pack_into("<II", data, 8, 16, metadata_size)
    struct.pack_into("<II", data, locator, geometry, len(data) - geometry)
    struct.pack_into("<3fI3fI", data, geometry, 0, 0, 0, 1, 1, 1, 1, 1)
    struct.pack_into("<I", data, geometry + 0x20, lod_table)
    struct.pack_into("<II", data, geometry + lod_table, 1, block)
    start = geometry + block
    struct.pack_into("<3f4B12xI", data, start, 0, 0, 0, 3, 1, 0, 0, 0x808001D0)
    data[start + 0x20 : start + 0x23] = bytes((0, 1, 2))
    cursor = start + 0x24
    for values in ((0, 0, 0, 0), (65535, 0, 0, 0), (0, 65535, 0, 0)):
        struct.pack_into("<4H", data, cursor, *values)
        cursor += 8
    data[cursor : cursor + 12] = bytes((128, 128, 0, 0)) * 3
    cursor += 12
    struct.pack_into("<6e", data, cursor, 0, 0, 1, 0, 0, 1)
    return bytes(data)


class TestWotsStMesh(unittest.TestCase):
    def test_minimal_lod0_scene(self):
        mesh = StMeshFile()
        mesh.read(_minimal_stmesh(), file_version=260209350, material_names=["leaf"])
        self.assertEqual(len(mesh.lods), 1)
        self.assertEqual(mesh.material_names, ["leaf"])
        scene = build_mesh_scene(mesh)[0]
        np.testing.assert_allclose(scene.vertices, ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
        self.assertEqual(scene.indices.tolist(), [0, 1, 2])
        self.assertEqual(scene.batches[0].material_name, "leaf")

    def test_bad_index_reports_block_offset(self):
        data = bytearray(_minimal_stmesh())
        data[0x260] = 3
        with self.assertRaisesRegex(StMeshParseError, r"0x240.*index 3"):
            StMeshFile().read(data, file_version=260209350)

    def test_truncated_geometry_reports_boundary(self):
        data = _minimal_stmesh()[:-1]
        with self.assertRaisesRegex(StMeshParseError, "geometry range ends"):
            StMeshFile().read(data, file_version=260209350)

    def test_material_candidate_uses_wots_a_suffix(self):
        candidates = list(
            MeshMaterialResolver.iter_mdf_candidates(
                "natives/stm/Art/Model/StageModel/tree.stmesh.260209350"
            )
        )
        self.assertEqual(
            candidates[0], "natives/stm/Art/Model/StageModel/tree_A.mdf2.51"
        )


@unittest.skipUnless(
    os.environ.get("REASY_WOTS_ASSET_ROOT"), "set REASY_WOTS_ASSET_ROOT"
)
class TestWotsStMeshAssets(unittest.TestCase):
    def test_all_stmesh_assets_parse(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        paths = list(root.rglob("*.stmesh.260209350"))
        self.assertTrue(paths)
        for path in paths:
            mesh = StMeshFile()
            mesh.read(path.read_bytes(), file_version=260209350)
            scene = build_mesh_scene(mesh)
            self.assertTrue(scene, str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
