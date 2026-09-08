from __future__ import annotations

import os
import struct
import unittest
from pathlib import Path

import numpy as np

from file_handlers.mesh.mesh_file import (
    MeshDependencyMissingError,
    MeshFile,
    MeshMainVersion,
    _decode_skin_weights,
    get_mesh_version,
)


class TestWotsMesh(unittest.TestCase):
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
