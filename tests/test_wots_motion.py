from __future__ import annotations

import os
import struct
import unittest
from pathlib import Path

from file_handlers.motion.binary import ReadContext
from file_handlers.motion.evaluation.model import EvaluatedPose, Transform
from file_handlers.motion.errors import MotionParseError, MotionWriteError
from file_handlers.motion.format_registry import find_motion_format
from file_handlers.motion.mot.model import TrackFamily
from file_handlers.motion.preview.model import MotionPreviewSnapshot
from file_handlers.motion.preview.renderer import MotionPreviewRenderer
from file_handlers.motion.wots_codec import WOTS_MOTION_FORMAT_CODEC, WotsMotParser


def _minimal_mot(name: str = "idle") -> bytes:
    encoded = name.encode("utf-16le") + b"\0\0"
    data = bytearray(0x7C + len(encoded))
    struct.pack_into("<I4s", data, 0, 973, b"mot ")
    struct.pack_into("<Q", data, 0x58, 0x7C)
    struct.pack_into("<ffff", data, 0x60, 1.0, -1.0, 0.0, 1.0)
    struct.pack_into("<H", data, 0x78, 60)
    data[0x7C:] = encoded
    return bytes(data)


def _minimal_motlist(payload: bytes) -> bytes:
    pointer_table = 0x60
    payload_offset = 0x70
    ids_offset = (payload_offset + len(payload) + 0xF) & ~0xF
    data = bytearray(ids_offset + 72)
    struct.pack_into("<I4s", data, 0, 1036, b"mlst")
    struct.pack_into("<QQQ", data, 0x10, pointer_table, ids_offset, 0x40)
    struct.pack_into("<I", data, 0x38, 1)
    data[0x40:0x4A] = "list".encode("utf-16le") + b"\0\0"
    struct.pack_into("<Q", data, pointer_table, payload_offset)
    data[payload_offset : payload_offset + len(payload)] = payload
    struct.pack_into("<H", data, ids_offset + 8, 42)
    return bytes(data)


def _preview_snapshot(
    positions: tuple[tuple[float, float, float], ...],
) -> MotionPreviewSnapshot:
    identity = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    pose = EvaluatedPose(
        frame=0.0,
        local_transforms=tuple(Transform() for _ in positions),
        world_matrices=tuple(identity for _ in positions),
        skin_matrices=tuple(None for _ in positions),
        root_transforms=(),
        node_weights=tuple(1.0 for _ in positions),
    )
    return MotionPreviewSnapshot(
        frame=0.0,
        end_frame=1.0,
        pose=pose,
        joint_names=("root", "spine"),
        joint_positions=positions,
        bone_pairs=((0, 1),),
        node_weights=(1.0, 1.0),
        root_deltas=(),
        deformation_weights=(),
        diagnostics=(),
    )


class _PreviewViewport:
    def __init__(self):
        self.labels = ()
        self.positions = ()

    def set_scene(self, _meshes, *, reset_camera=True):
        self.reset_camera = reset_camera

    def set_bone_name_labels(self, names, positions):
        self.labels = tuple(names)
        self.positions = tuple(positions)

    def clear_bone_name_labels(self):
        self.labels = ()
        self.positions = ()

    def update_mesh_transforms(self, _matrices, *, recompute_bounds=True):
        self.recompute_bounds = recompute_bounds


class TestWotsMotion(unittest.TestCase):
    def test_v1036_dispatch_and_read_only(self):
        codec = find_motion_format(_minimal_motlist(_minimal_mot()))
        self.assertIs(codec, WOTS_MOTION_FORMAT_CODEC)
        model = codec.parse(_minimal_motlist(_minimal_mot()), label="fixture")
        self.assertEqual(model.name, "list")
        self.assertEqual(model.slots[0].motion_id, 42)
        self.assertEqual(model.slots[0].payload.value.name, "idle")
        with self.assertRaisesRegex(MotionWriteError, "read-only"):
            codec.write(model)

    def test_direct_v973(self):
        motion = WOTS_MOTION_FORMAT_CODEC.parse_mot(
            _minimal_mot("walk"), label="fixture"
        )
        self.assertEqual(motion.name, "walk")
        self.assertEqual(motion.end_frame, 1.0)
        self.assertFalse(motion.looping)

    def test_bad_payload_reports_offset_and_type(self):
        data = bytearray(_minimal_motlist(_minimal_mot()))
        data[0x74:0x78] = b"bad!"
        with self.assertRaisesRegex(MotionParseError, r"bad!.*0x70"):
            WOTS_MOTION_FORMAT_CODEC.parse(data, label="fixture")

    def test_unknown_track_compression_reports_track_offset(self):
        context = ReadContext.from_bytes(bytes(16), label="fixture")
        parser = WotsMotParser(context)
        with self.assertRaisesRegex(
            MotionParseError,
            r"unsupported translation compression 0x90000 at track 0x44",
        ):
            parser._decode_values(
                context,
                0,
                1,
                TrackFamily.VECTOR3,
                0x90000,
                (0.0,) * 8,
                "translation",
                0x44,
            )

    def test_preview_renderer_updates_and_clears_bone_name_labels(self):
        viewport = _PreviewViewport()
        renderer = MotionPreviewRenderer(viewport)
        first = _preview_snapshot(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        renderer.present(first, None, reset_camera=True)
        self.assertEqual(viewport.labels, ("root", "spine"))
        self.assertEqual(viewport.positions, first.joint_positions)

        second = _preview_snapshot(((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)))
        renderer.present(second, None, reset_camera=False)
        self.assertEqual(viewport.positions, second.joint_positions)

        renderer.clear()
        self.assertEqual(viewport.labels, ())
        self.assertEqual(viewport.positions, ())


@unittest.skipUnless(
    os.environ.get("REASY_WOTS_ASSET_ROOT"), "set REASY_WOTS_ASSET_ROOT"
)
class TestWotsAssets(unittest.TestCase):
    def test_all_motion_assets_parse(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        motlists = list(root.rglob("*.motlist.1036"))
        motions = list(root.rglob("*.mot.973"))
        self.assertTrue(motlists)
        self.assertTrue(motions)
        for path in motlists:
            WOTS_MOTION_FORMAT_CODEC.parse(path.read_bytes(), label=str(path))
        for path in motions:
            WOTS_MOTION_FORMAT_CODEC.parse_mot(path.read_bytes(), label=str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
