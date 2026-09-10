from __future__ import annotations

import os
import struct
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from file_handlers.motion.binary import ReadContext
from file_handlers.motion.evaluation.model import EvaluatedPose, Transform
from file_handlers.motion.errors import MotionParseError, MotionWriteError
from file_handlers.motion.format_registry import find_motion_format
from file_handlers.motion.mot.model import (
    AnimationNode,
    Joint,
    Motion,
    Skeleton,
    TrackFamily,
)
from file_handlers.motion.mot_clip.model import (
    ClipInterpolation,
    ClipKey,
    ClipNode,
    ClipProperty,
    ClipPropertyType,
    CompactMotClip,
)
from file_handlers.motion.sequence.model import SequenceCategory, SequenceData
from file_handlers.motion.preview.catalog import (
    MotionPreviewCatalog,
    _rebind_motion_skeleton,
    _select_shared_skeleton,
)
from file_handlers.motion.preview.model import MotionPreviewSnapshot
from file_handlers.motion.preview.renderer import MotionPreviewRenderer
from file_handlers.motion.preview.resolution import (
    MotionListDocument,
    MotionPreviewResolver,
    WOTS_TREE_MOTION_REFERENCES,
)
from file_handlers.motion.wots_codec import WOTS_MOTION_FORMAT_CODEC, WotsMotParser
from file_handlers.motbank.motbank_file import MotbankFile, MotlistItem
from utils.hash_util import murmur3_hash
from utils.resource_file_utils import ResourceResolutionContext


def _minimal_mot(name: str = "idle") -> bytes:
    encoded = name.encode("utf-16le") + b"\0\0"
    data = bytearray(0x7C + len(encoded))
    struct.pack_into("<I4s", data, 0, 973, b"mot ")
    struct.pack_into("<Q", data, 0x58, 0x7C)
    struct.pack_into("<ffff", data, 0x60, 1.0, -1.0, 0.0, 1.0)
    struct.pack_into("<H", data, 0x78, 60)
    data[0x7C:] = encoded
    return bytes(data)


def _minimal_mot_with_timeline() -> bytes:
    data = bytearray(0x300)
    struct.pack_into("<I4s", data, 0, 973, b"mot ")
    data[0x7C:0x8A] = "attack".encode("utf-16le") + b"\0\0"
    struct.pack_into("<Q", data, 0x30, 0xA0)
    struct.pack_into("<Q", data, 0x58, 0x7C)
    struct.pack_into("<ffff", data, 0x60, 20.0, -1.0, 0.0, 20.0)
    struct.pack_into("<B", data, 0x74, 1)
    struct.pack_into("<H", data, 0x78, 60)

    sequence = 0xB0
    clip = sequence + 0x40
    node_table = clip + 0x98
    property_table = node_table + 2 * 0x28
    bool_table = property_table + 0x38
    strings8 = bool_table + 0x10
    strings16 = strings8 + 4
    strings16_data = (
        "Root\0app.motion_track.AttackCollision_Wp\0".encode("utf-16le")
    )
    tracks = (strings16 + len(strings16_data) + 0xF) & ~0xF
    end = tracks + 0x1C

    struct.pack_into("<Q", data, 0xA0, sequence)
    struct.pack_into("<QQ", data, sequence + 8, clip, tracks)
    struct.pack_into("<II", data, sequence + 0x1C, 1, 1)
    struct.pack_into("<I", data, sequence + 0x24, 0)

    struct.pack_into("<IIfIIII", data, clip, 0x50494C43, 89, 20.0, 2, 1, 0, 2)
    struct.pack_into("<II", data, clip + 0x1C, 0, 0)
    pointers = (
        node_table,
        property_table,
        bool_table,
        bool_table,
        strings8,
        strings8,
        strings8,
        strings8,
        strings8,
        strings8,
        strings8,
        strings16,
        tracks,
        tracks,
    )
    struct.pack_into("<14Q", data, clip + 0x28, *pointers)

    struct.pack_into("<HHIIIQQQ", data, node_table, 1, 0, 0, 0, 0, 0, 1, 0)
    struct.pack_into(
        "<HHIIIQQQ",
        data,
        node_table + 0x28,
        0,
        1,
        0,
        0,
        0,
        5,
        0,
        0,
    )
    struct.pack_into("<ffIIQ", data, property_table, 10.0, 20.0, 0, 0, 0)
    struct.pack_into(
        "<QHhBBBBQ",
        data,
        property_table + 0x20,
        0,
        2,
        -1,
        0,
        1,
        0,
        0x10,
        0,
    )
    struct.pack_into("<fI", data, bool_table, 10.0, 1 | (13 << 1) | (10 << 10))
    struct.pack_into("<fI", data, bool_table + 8, 20.0, 1 | (1 << 1))
    data[strings8:strings8 + 4] = b"_On\0"
    data[strings16:strings16 + len(strings16_data)] = strings16_data
    struct.pack_into("<I", data, tracks, 0xFFFFFFFF)
    return bytes(data[:end])


def _minimal_motlist(payload: bytes, motion_id: int = 42) -> bytes:
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
    struct.pack_into("<H", data, ids_offset + 8, motion_id)
    return bytes(data)


def _minimal_mottree() -> bytes:
    data = bytearray(0xE0)
    struct.pack_into("<I4s", data, 0, 21, b"mtre")
    struct.pack_into("<QQQ", data, 0x20, 0x70, 0, 0x60)
    struct.pack_into("<Q", data, 0x40, 0x58)
    struct.pack_into("<HHHH", data, 0x48, 1, 0, 0, 0)
    struct.pack_into("<HHHH", data, 0x50, 0, 1, 0, 0)
    struct.pack_into("<II", data, 0x58, 10040, 9)
    data[0x60:0x6A] = "tree".encode("utf-16le") + b"\0\0"
    struct.pack_into("<Q", data, 0x70, 0xAC)
    struct.pack_into("<Q", data, 0x88, 0xB8)
    struct.pack_into("<III", data, 0x90, 0, 0, murmur3_hash("Node1".encode("utf-16le")))
    data[0x9C:0xA0] = bytes((0, 2, 1, 0))
    data[0xA0:0xAB] = b"MotionNode\0"
    data[0xAC:0xB8] = "Node1".encode("utf-16le") + b"\0\0"
    struct.pack_into("<IIQ", data, 0xB8, 6, 0x5CD4677B, 10040)
    struct.pack_into("<IIQ", data, 0xC8, 6, 0x9F595148, 9)
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


class _PakReader:
    def __init__(self, files):
        self.files = files

    def cached_known_paths(self):
        return tuple(self.files)

    def get_file(self, path):
        data = self.files.get(path)
        return BytesIO(data) if data is not None else None


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

    def test_v973_embedded_clip_v89_timeline(self):
        motion = WOTS_MOTION_FORMAT_CODEC.parse_mot(
            _minimal_mot_with_timeline(), label="fixture"
        )
        self.assertEqual(len(motion.sequences), 1)
        sequence = motion.sequences[0]
        self.assertEqual(sequence.category.name, "GAME")
        self.assertEqual(sequence.clip.total_frame, 20.0)
        track = sequence.clip.root.children[0]
        self.assertEqual(track.name, "app.motion_track.AttackCollision_Wp")
        event = track.properties[0]
        self.assertEqual((event.start_frame, event.end_frame), (10.0, 20.0))
        self.assertEqual([key.value for key in event.keys], [True, True])
        self.assertEqual(event.keys[0].interpolation, ClipInterpolation.RANGE)
        from file_handlers.motion.preview.event_timeline import motion_timeline_lanes

        lanes = motion_timeline_lanes(motion)
        self.assertEqual(len(lanes), 1)
        self.assertEqual(lanes[0].category, "GAME")
        self.assertEqual(lanes[0].intervals, ((10.0, 20.0),))
        self.assertIn("_On  [10–20]", lanes[0].details)
        self.assertEqual(lanes[0].name, "Attack Collision · Weapon")
        self.assertIn("Attack Collision (Weapon)", lanes[0].details)

    def test_player_cancel_timeline_exposes_semantic_phase_ranges(self):
        phase = ClipProperty(
            "_Phase",
            ClipPropertyType.ENUM,
            start_frame=20.0,
            end_frame=216.0,
            keys=[
                ClipKey(
                    frame=20.0,
                    interpolation=ClipInterpolation.DISCRETE_TO_END,
                    value="PRE",
                ),
                ClipKey(
                    frame=40.0,
                    interpolation=ClipInterpolation.DISCRETE_TO_END,
                    value="ACTUAL",
                ),
                ClipKey(
                    frame=216.0,
                    interpolation=ClipInterpolation.DISCRETE,
                    value="ACTUAL",
                ),
            ],
        )
        group = ClipProperty(
            "_DodgeGroupCancel",
            ClipPropertyType.NATIVE_CLASS,
            start_frame=0.0,
            end_frame=216.0,
            children=[phase],
        )
        node = ClipNode(
            "app.motion_track.PlayerCommandCancel",
            properties=[group],
        )
        clip = CompactMotClip(
            total_frame=216.0,
            root=ClipNode("Root", children=[node]),
        )
        motion = Motion(
            "cancel",
            sequences=[SequenceData(SequenceCategory.GAME, clip)],
        )

        from file_handlers.motion.preview.event_timeline import (
            _timeline_interval_is_active,
            motion_timeline_lanes,
            timeline_lane_details,
        )

        lanes = motion_timeline_lanes(motion)
        self.assertEqual(len(lanes), 1)
        lane = lanes[0]
        self.assertEqual(lane.name, "Cancel · Dodge")
        self.assertEqual(
            [(item.start, item.end, item.state) for item in lane.segments],
            [(20.0, 40.0, "PRE"), (40.0, 216.0, "ACTUAL")],
        )
        details = timeline_lane_details(lane, 30.0, None)
        self.assertIn("CURRENT STATE: PRE", details)
        self.assertIn("Frames 20–40  PRE", details)
        self.assertNotIn("seconds", details)
        self.assertNotIn("216: ACTUAL", details)
        self.assertTrue(_timeline_interval_is_active(30.0, 20.0, 40.0, 216.0))
        self.assertFalse(_timeline_interval_is_active(40.0, 20.0, 40.0, 216.0))

    def test_v21_mottree_resolves_bank_and_motion_reference(self):
        model = WOTS_MOTION_FORMAT_CODEC.parse(
            _minimal_motlist(_minimal_mottree()), label="fixture"
        )
        self.assertEqual(model.slots[0].payload.value.name, "tree")
        resolution = MotionPreviewResolver(
            lambda _path: None,
            WOTS_TREE_MOTION_REFERENCES,
        ).resolve(MotionListDocument("fixture", model))
        self.assertEqual(
            [(item.bank_id, item.motion_id) for item in resolution.tree_references],
            [(10040, 9)],
        )
        self.assertEqual(resolution.unresolved_bank_ids, (10040,))

    def test_tree_only_list_loads_nearby_motbank_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            common = Path(temporary) / "natives/stm/Motion/Player/Common"
            tree_path = common / "plc_tree/plc_tree.motlist.1036"
            tree_path.parent.mkdir(parents=True)
            tree_path.write_bytes(_minimal_motlist(_minimal_mottree(), 1))
            blend_path = common / "plc_BaseBlend/plc_BaseBlend.motlist.1036"
            blend_path.parent.mkdir(parents=True)
            blend_path.write_bytes(_minimal_motlist(_minimal_mot("blend"), 9))

            bank = MotbankFile()
            bank.version = 4
            bank.items = [
                MotlistItem(
                    bank_id=10040,
                    path="Motion/Player/Common/plc_BaseBlend/plc_BaseBlend.motlist",
                )
            ]
            (common / "Common.motbank.4").write_bytes(bank.write())

            model = WOTS_MOTION_FORMAT_CODEC.parse(
                tree_path.read_bytes(), label=str(tree_path)
            )
            catalog = MotionPreviewCatalog(
                MotionListDocument(str(tree_path), model),
                WOTS_TREE_MOTION_REFERENCES,
                WOTS_MOTION_FORMAT_CODEC,
            )
            resolution = catalog.refresh()
            self.assertEqual(resolution.unresolved_bank_ids, ())
            self.assertEqual(
                [
                    (entry.bank_id, entry.motion_id, entry.name)
                    for entry in resolution.entries
                ],
                [(10040, 9, "blend")],
            )

    def test_tree_only_pak_resource_loads_owning_motbank(self):
        tree_path = "natives/stm/Motion/Player/Common/plc_tree/plc_tree.motlist.1036"
        blend_path = "natives/stm/Motion/Player/Common/plc_BaseBlend/plc_BaseBlend.motlist.1036"
        bank_path = "natives/stm/Motion/Player/Common/Common.motbank.4"
        bank = MotbankFile()
        bank.version = 4
        bank.items = [
            MotlistItem(
                bank_id=10040,
                path="Motion/Player/Common/plc_BaseBlend/plc_BaseBlend.motlist",
            )
        ]
        files = {
            tree_path: _minimal_motlist(_minimal_mottree(), 1),
            blend_path: _minimal_motlist(_minimal_mot("blend"), 9),
            bank_path: bank.write(),
        }
        reader = _PakReader(files)
        model = WOTS_MOTION_FORMAT_CODEC.parse(files[tree_path], label=tree_path)
        catalog = MotionPreviewCatalog(
            MotionListDocument(tree_path, model),
            WOTS_TREE_MOTION_REFERENCES,
            WOTS_MOTION_FORMAT_CODEC,
            resource_context=ResourceResolutionContext(pak_cached_reader=reader),
        )
        resolution = catalog.refresh()
        self.assertEqual(resolution.unresolved_bank_ids, ())
        self.assertEqual(
            [(entry.bank_id, entry.motion_id, entry.name) for entry in resolution.entries],
            [(10040, 9, "blend")],
        )

    def test_shared_skeleton_rebind_restores_names_and_topology(self):
        root = Joint("root", binding_hash=1)
        child = Joint("child", parent=root, binding_hash=2)
        root.children.append(child)
        placeholder = Joint("bone_00000002", binding_hash=2)
        motion = Motion(
            "shared",
            skeleton=Skeleton([placeholder]),
            animation_nodes=[AnimationNode(placeholder)],
            source_joint_count=2,
        )
        _rebind_motion_skeleton(motion, Skeleton([root, child]))
        self.assertEqual([joint.name for joint in motion.skeleton.joints], ["root", "child"])
        self.assertIs(motion.animation_nodes[0].joint, motion.skeleton.joints[1])
        self.assertIs(motion.skeleton.joints[1].parent, motion.skeleton.joints[0])

    def test_shared_skeleton_can_be_smaller_when_it_covers_every_track_hash(self):
        root = Joint("root", binding_hash=1)
        animated = Joint("animated", parent=root, binding_hash=2)
        root.children.append(animated)
        nearby_extra = Joint("nearby_extra", binding_hash=3)
        placeholder = Joint("bone_00000002", binding_hash=2)
        motion = Motion(
            "variant",
            skeleton=Skeleton([placeholder]),
            animation_nodes=[AnimationNode(placeholder)],
            source_joint_count=4,
        )
        embedded = Skeleton([root, animated])
        nearby = Skeleton([root, animated, nearby_extra])
        selected = _select_shared_skeleton(
            motion,
            [embedded, nearby],
            {id(embedded)},
        )
        self.assertIs(selected, embedded)

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

    def test_base_damage_010_and_019_use_shared_bank_skeleton(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = (
            root
            / "Motion/Player/Common/plc_BaseDamage/plc_BaseDamage.motlist.1036"
        )
        model = WOTS_MOTION_FORMAT_CODEC.parse(path.read_bytes(), label=str(path))
        catalog = MotionPreviewCatalog(
            MotionListDocument(str(path), model),
            WOTS_TREE_MOTION_REFERENCES,
            WOTS_MOTION_FORMAT_CODEC,
        )
        resolution = catalog.refresh()
        by_id = {entry.motion_id: entry.resolve_motion() for entry in resolution.entries}
        for motion_id in (10, 19):
            motion = by_id[motion_id]
            self.assertEqual(len(motion.skeleton.joints), 319)
            self.assertTrue(
                any(joint.parent is not None for joint in motion.skeleton.joints)
            )
            self.assertFalse(
                any(joint.name.startswith("bone_") for joint in motion.skeleton.joints)
            )

    def test_katate_attack_202_uses_its_embedded_skeleton(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = (
            root
            / "Motion/Player/Weapon/plw_KatateAttack/plw_KatateAttack.motlist.1036"
        )
        model = WOTS_MOTION_FORMAT_CODEC.parse(path.read_bytes(), label=str(path))
        catalog = MotionPreviewCatalog(
            MotionListDocument(str(path), model),
            WOTS_TREE_MOTION_REFERENCES,
            WOTS_MOTION_FORMAT_CODEC,
        )
        resolution = catalog.refresh()
        by_id = {entry.motion_id: entry.resolve_motion() for entry in resolution.entries}
        motion = by_id[202]
        self.assertEqual(motion.source_joint_count, 323)
        self.assertEqual(len(motion.skeleton.joints), 318)
        self.assertTrue(any(joint.parent is not None for joint in motion.skeleton.joints))
        self.assertFalse(
            any(node.joint.name.startswith("bone_") for node in motion.animation_nodes)
        )
        self.assertNotIn("no compatible 323-joint skeleton", "\n".join(catalog.messages))
        self.assertNotIn("no compatible 409-joint skeleton", "\n".join(catalog.messages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
