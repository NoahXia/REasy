from __future__ import annotations

import os
import struct
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from file_handlers.motion.binary import ReadContext
from file_handlers.motion.evaluation.model import (
    EvaluatedPose,
    Rig,
    RigJoint,
    Transform,
)
from file_handlers.motion.evaluation.shared_rig import SharedRigPoseMapper
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
from file_handlers.motion.preview.controller import MotionPreviewController
from file_handlers.motion.preview.model import MotionPreviewSnapshot
from file_handlers.motion.preview.renderer import MotionPreviewRenderer
from file_handlers.motion.preview.support_registry import entity_motion_support_for_format
from file_handlers.motion.preview.target import (
    WOTS_MESH_PREVIEW_PRESETS,
    load_re_engine_mesh_preset_target,
)
from file_handlers.motion.preview.attack_collision import (
    AttackCollisionOverlay,
    AttackCollisionSource,
    active_attack_collisions,
    attack_collision_geometry_diagnostic,
    attack_rcol_resource_candidates,
    collision_type_label,
    load_attack_collision_resource,
)
from file_handlers.motion.preview.resolution import (
    MotionListDocument,
    MotionPreviewResolver,
    WOTS_TREE_MOTION_REFERENCES,
)
from file_handlers.motion.wots_codec import WOTS_MOTION_FORMAT_CODEC, WotsMotParser
from file_handlers.motbank.motbank_file import MotbankFile, MotlistItem
from file_handlers.rcol.shape_types import Sphere
from utils.hash_util import murmur3_hash
from utils.resource_file_utils import ResourceResolutionContext
from utils.type_registry import TypeRegistry


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
    def __init__(self, gpu_palette_limit=None):
        self.labels = ()
        self.positions = ()
        self.scene = []
        self.wireframe_overlays = []
        self.skinning = {}
        self.geometry_updates = set()
        self.scene_update_count = 0
        self.gpu_palette_limit = gpu_palette_limit

    def set_scene(self, meshes, *, reset_camera=True):
        self.scene = list(meshes)
        self.reset_camera = reset_camera
        self.scene_update_count += 1

    def set_wireframe_overlays(self, meshes):
        self.wireframe_overlays = list(meshes)

    def update_wireframe_overlay_geometries(self, geometries):
        by_key = {mesh.key: mesh for mesh in self.wireframe_overlays}
        for key, vertices in geometries.items():
            mesh = by_key.get(key)
            if mesh is not None:
                mesh.vertices = np.asarray(vertices, dtype=np.float32)
                self.geometry_updates.add(key)

    def set_mesh_skinning(self, key, binding):
        self.skinning[key] = binding

    def can_use_gpu_skinning(self, binding):
        if self.gpu_palette_limit is None:
            return True
        active = binding.weights > 0.0
        return (
            len(np.unique(binding.joint_indices[active]))
            <= self.gpu_palette_limit
        )

    def update_mesh_skinning(self, _key, _matrices):
        pass

    def update_mesh_skinning_source(self, _key, _positions, _normals):
        pass

    def clear_mesh_skinning(self, keys=None):
        if keys is None:
            self.skinning.clear()
        else:
            for key in keys:
                self.skinning.pop(key, None)

    def set_bone_name_labels(self, names, positions):
        self.labels = tuple(names)
        self.positions = tuple(positions)

    def clear_bone_name_labels(self):
        self.labels = ()
        self.positions = ()

    def update_mesh_transforms(self, _matrices, *, recompute_bounds=True):
        self.recompute_bounds = recompute_bounds

    def update_mesh_geometry(
        self,
        key,
        _vertices,
        _normals=None,
        *,
        recompute_bounds=True,
    ):
        self.geometry_updates.add(key)
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
    def test_shared_rig_root_can_follow_an_owner_attachment_joint(self):
        owner = Rig([
            RigJoint("root"),
            RigJoint("R_Wep", 0),
        ])
        weapon = Rig([
            RigJoint("root"),
            RigJoint("Base", 0),
        ])
        mapper = SharedRigPoseMapper(
            owner,
            weapon,
            np.identity(4, dtype=np.float32),
            root_attachment_joint="R_Wep",
        )
        root = np.identity(4, dtype=np.float32)
        hand = np.identity(4, dtype=np.float32)
        hand[3, :3] = (1.0, 2.0, 3.0)
        world = mapper.world_matrices((root, hand))
        np.testing.assert_allclose(world[0], hand)
        np.testing.assert_allclose(world[1], hand)

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
        self.assertEqual(lanes[0].properties[0].name, "_On")
        self.assertEqual(lanes[0].name, "Attack Collision · Weapon")

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
            timeline_event_selection,
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
        self.assertIn("EventName: Cancel · Dodge", details)
        self.assertIn("StartFrame: 20", details)
        self.assertIn("EndFrame: 40", details)
        self.assertIn("FrameCount: 20", details)
        self.assertNotIn("RAW TRACK DATA", details)
        selection = timeline_event_selection(lane, 30.0, None)
        self.assertEqual(selection.properties[0].name, "_Phase")
        self.assertEqual((selection.start_frame, selection.end_frame), (20.0, 40.0))
        self.assertTrue(_timeline_interval_is_active(30.0, 20.0, 40.0, 216.0))
        self.assertFalse(_timeline_interval_is_active(40.0, 20.0, 40.0, 216.0))

    def test_timeline_uses_playable_motion_duration(self):
        from file_handlers.motion.preview.event_timeline import (
            TimelineLane,
            _display_property_name,
            _format_frame,
            _humanize_identifier,
            _timeline_end_frame,
        )

        lanes = (
            TimelineLane("VFX", "VFXRange", 347.0, ((15.0, 25.0),), (), ""),
        )
        motion = Motion("attack", end_frame=184.0)
        self.assertEqual(_timeline_end_frame(motion, lanes), 184.0)

        motion.end_frame = 0.0
        self.assertEqual(_timeline_end_frame(motion, lanes), 347.0)
        self.assertEqual(_format_frame(24.0), "24")
        self.assertEqual(_format_frame(24.25), "24.25")
        self.assertEqual(_display_property_name("_AttackParamID"), "AttackParamID")
        self.assertEqual(_display_property_name("__Internal"), "Internal")
        self.assertEqual(
            _display_property_name("_<BlendRate>k__BackingField"),
            "BlendRate",
        )
        self.assertEqual(
            _humanize_identifier("VFXTrigger_Wp"),
            "VFX Trigger Wp",
        )

    def test_attack_collision_semantics_and_rcol_resource_selection(self):
        on = ClipProperty(
            "_On",
            ClipPropertyType.BOOL,
            start_frame=10.0,
            end_frame=20.0,
            keys=[ClipKey(frame=10.0, value=True)],
        )
        request = ClipProperty(
            "_RequestSetID",
            ClipPropertyType.U32,
            start_frame=0.0,
            end_frame=30.0,
            keys=[ClipKey(frame=0.0, value=120)],
        )
        attack = ClipProperty(
            "_AttackParamID",
            ClipPropertyType.U32,
            start_frame=0.0,
            end_frame=30.0,
            keys=[ClipKey(frame=0.0, value=42)],
        )
        collision = ClipProperty(
            "_CollisionType",
            ClipPropertyType.U32,
            start_frame=0.0,
            end_frame=30.0,
            keys=[ClipKey(frame=0.0, value=2)],
        )
        motion = Motion(
            "attack",
            end_frame=30.0,
            sequences=[SequenceData(
                SequenceCategory.GAME,
                CompactMotClip(
                    total_frame=30.0,
                    root=ClipNode(
                        "Root",
                        children=[ClipNode(
                            "app.motion_track.AttackCollision_Wp",
                            properties=[request, on, attack, collision],
                        )],
                    ),
                ),
            )],
        )
        self.assertEqual(active_attack_collisions(motion, 9.0), ())
        active = active_attack_collisions(motion, 12.0)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].request_set_id, 120)
        self.assertEqual(active[0].attack_param_id, 42)
        self.assertEqual(active[0].collision_type_name, "MAIN_WEAPON")
        self.assertEqual(collision_type_label(8), "SUB7_WEAPON")
        self.assertEqual(
            attack_rcol_resource_candidates(
                "natives/stm/Motion/Player/Weapon/test.motlist.1036"
            ),
            (
                "natives/stm/GameDesign/Action/Player/Collision/Collider/PlayerAttack.rcol.37",
            ),
        )

    def test_attack_overlay_prefers_authored_request_set_id(self):
        payload = Sphere()
        payload.center = [0.0, 1.0, 0.0]
        payload.radius = 0.25
        shape = SimpleNamespace(
            shape=payload,
            info=SimpleNamespace(
                primary_joint_name_str="root",
                secondary_joint_name_str="",
                primary_joint_name_hash=0,
                secondary_joint_name_hash=0,
            ),
        )
        authored_group = SimpleNamespace(shapes=[shape], extra_shapes=[])
        fallback_group = SimpleNamespace(shapes=[], extra_shapes=[])
        source = AttackCollisionSource(
            "fixture.rcol.37",
            SimpleNamespace(request_sets=[
                SimpleNamespace(
                    info=SimpleNamespace(field0=120, id=37),
                    group=authored_group,
                ),
                SimpleNamespace(
                    info=SimpleNamespace(field0=999, id=120),
                    group=fallback_group,
                ),
            ]),
        )
        self.assertIs(source.request_set(120).group, authored_group)

        event_motion = Motion(
            "attack",
            end_frame=30.0,
            sequences=[SequenceData(
                SequenceCategory.GAME,
                CompactMotClip(
                    total_frame=30.0,
                    root=ClipNode(
                        "Root",
                        children=[ClipNode(
                            "app.motion_track.AttackCollision_Body",
                            properties=[
                                ClipProperty(
                                    "_RequestSetID",
                                    ClipPropertyType.U32,
                                    0.0,
                                    30.0,
                                    keys=[ClipKey(frame=0.0, value=120)],
                                ),
                                ClipProperty(
                                    "_On",
                                    ClipPropertyType.BOOL,
                                    5.0,
                                    15.0,
                                    keys=[ClipKey(frame=5.0, value=True)],
                                ),
                                ClipProperty(
                                    "_AttackParamID",
                                    ClipPropertyType.U32,
                                    0.0,
                                    30.0,
                                    keys=[ClipKey(frame=0.0, value=7)],
                                ),
                            ],
                        )],
                    ),
                ),
            )],
        )
        snapshot = _preview_snapshot(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        snapshot = MotionPreviewSnapshot(
            frame=10.0,
            end_frame=snapshot.end_frame,
            pose=snapshot.pose,
            joint_names=snapshot.joint_names,
            joint_positions=snapshot.joint_positions,
            bone_pairs=snapshot.bone_pairs,
            node_weights=snapshot.node_weights,
            root_deltas=snapshot.root_deltas,
            deformation_weights=snapshot.deformation_weights,
            diagnostics=snapshot.diagnostics,
        )
        overlay = AttackCollisionOverlay(source)
        meshes = overlay.meshes(event_motion, snapshot)
        self.assertEqual(overlay.signature(event_motion, 10.0), (120,))
        self.assertEqual(len(meshes), 1)
        self.assertTrue(meshes[0].key.startswith(overlay.KEY_PREFIX))

        # Weapon-track RCOL geometry belongs to a separate weapon
        # ColliderSwitcher. It must never be interpreted in actor-root space.
        track = event_motion.sequences[0].clip.root.children[0]
        track.name = "app.motion_track.AttackCollision_Wp"
        track.properties.append(ClipProperty(
            "_CollisionType",
            ClipPropertyType.U32,
            0.0,
            30.0,
            keys=[ClipKey(frame=0.0, value=2)],
        ))
        active = active_attack_collisions(event_motion, 10.0)
        self.assertEqual(len(active), 1)
        self.assertIn(
            "weapon-local collider space",
            attack_collision_geometry_diagnostic(active[0]),
        )
        self.assertEqual(overlay.signature(event_motion, 10.0), ())
        self.assertEqual(overlay.meshes(event_motion, snapshot), [])
        renderer = MotionPreviewRenderer(_PreviewViewport())
        renderer.set_attack_collision_source(source)
        status = renderer.attack_collision_status(event_motion, 10.0)
        self.assertIn("Attack hitbox not drawn: RequestSet 120", status)
        self.assertIn("MAIN_WEAPON uses weapon-local collider space", status)

    def test_sound_event_profile_uses_enabled_trigger_keys(self):
        trigger = ClipProperty(
            "Trigger",
            ClipPropertyType.BOOL,
            start_frame=10.0,
            end_frame=20.0,
            keys=[
                ClipKey(frame=10.0, value=True),
                ClipKey(frame=20.0, value=False),
            ],
        )
        motion = Motion(
            "sound",
            end_frame=30.0,
            sequences=[
                SequenceData(
                    SequenceCategory.SOUND,
                    CompactMotClip(
                        total_frame=30.0,
                        root=ClipNode(
                            "Root",
                            children=[
                                ClipNode(
                                    "app.snd_mot_track.SoundTriggerTracksApp",
                                    properties=[trigger],
                                )
                            ],
                        ),
                    ),
                )
            ],
        )

        from file_handlers.motion.preview.event_timeline import motion_timeline_lanes

        lane = motion_timeline_lanes(motion)[0]
        self.assertEqual(lane.name, "Sound Trigger")
        self.assertEqual(lane.intervals, ((10.0, 10.0),))
        self.assertEqual(lane.markers, (10.0,))
        self.assertIn("Wwise", lane.details)

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
    def test_ch001_model_preset_composes_character_and_weapon_parts(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        preset = WOTS_MESH_PREVIEW_PRESETS[0]
        resources = []
        for resource_path in preset.resource_paths:
            relative = resource_path.replace("\\", "/")
            relative = relative.split("natives/stm/", 1)[-1]
            path = root / Path(relative)
            resources.append((str(path), path.read_bytes()))
        target = load_re_engine_mesh_preset_target(preset, tuple(resources))
        self.assertEqual(len(target.render_parts), 5)
        self.assertEqual(
            tuple(part.attachment_joint for part in target.render_parts),
            ("", "", "", "R_Wep", "Katana_root"),
        )
        self.assertEqual(
            tuple(part.weapon_collision_type for part in target.render_parts),
            (None, None, None, 2, None),
        )

        motlist = (
            root
            / "Motion/Player/Weapon/plw_KatateAttack/"
            "plw_KatateAttack.motlist.1036"
        )
        model = WOTS_MOTION_FORMAT_CODEC.parse(
            motlist.read_bytes(),
            label=str(motlist),
        )
        catalog = MotionPreviewCatalog(
            MotionListDocument(str(motlist), model),
            WOTS_TREE_MOTION_REFERENCES,
            WOTS_MOTION_FORMAT_CODEC,
        )
        motion = next(
            entry.resolve_motion()
            for entry in catalog.refresh().entries
            if entry.name.casefold() == "plw_katateattack_000"
        )
        support = entity_motion_support_for_format(WOTS_MOTION_FORMAT_CODEC)
        self.assertIsNotNone(support)
        controller = MotionPreviewController(support.evaluation)
        self.assertTrue(
            controller.load(motion, target.rig),
            controller.error_message,
        )
        controller.set_frame(24.0)
        viewport = _PreviewViewport()
        renderer = MotionPreviewRenderer(viewport)
        renderer.present(
            controller.sample(),
            target,
            reset_camera=True,
            motion=motion,
        )
        expected = {
            "motion-preview:target:0",
            "motion-preview:target:1",
            "motion-preview:target:2",
            "motion-preview:target:3",
            "motion-preview:target:4",
        }
        self.assertEqual({mesh.key for mesh in viewport.scene}, expected)
        self.assertEqual(set(viewport.skinning), expected)

        limited_viewport = _PreviewViewport(gpu_palette_limit=1)
        limited_renderer = MotionPreviewRenderer(limited_viewport)
        limited_renderer.present(
            controller.sample(),
            target,
            reset_camera=True,
            motion=motion,
        )
        oversized = {
            key
            for key, deformer in limited_renderer._deformers.items()
            if len(np.unique(
                deformer.binding.joint_indices[deformer.binding.weights > 0.0]
            )) > 1
        }
        self.assertTrue(oversized)
        self.assertTrue(oversized.isdisjoint(limited_viewport.skinning))

        def resource_loader(resource_path):
            relative = resource_path.replace("\\", "/")
            relative = relative.split("natives/stm/", 1)[-1]
            path = root / Path(relative)
            return (str(path), path.read_bytes()) if path.is_file() else None

        registry = TypeRegistry(str(
            Path(__file__).resolve().parents[1]
            / "resources/data/dumps/rszoniwots.json"
        ))
        actor_source, actor_errors = load_attack_collision_resource(
            "natives/stm/GameDesign/Action/Player/Collision/Collider/"
            "PlayerAttack.rcol.37",
            resource_loader,
            type_registry=registry,
        )
        weapon_source, weapon_errors = load_attack_collision_resource(
            preset.weapon_attack_resources[0][1],
            resource_loader,
            type_registry=registry,
        )
        self.assertFalse(actor_errors)
        self.assertFalse(weapon_errors)
        self.assertIsNotNone(actor_source)
        self.assertIsNotNone(weapon_source)

        collision_viewport = _PreviewViewport()
        collision_renderer = MotionPreviewRenderer(collision_viewport)
        collision_renderer.set_attack_collision_source(actor_source)
        collision_renderer.set_weapon_attack_collision_sources({2: weapon_source})
        collision_renderer.present(
            controller.sample(),
            target,
            reset_camera=True,
            motion=motion,
        )
        scene_update_count = collision_viewport.scene_update_count
        controller.set_frame(27.0)
        snapshot = controller.sample()
        collision_renderer.present(
            snapshot,
            target,
            reset_camera=False,
            motion=motion,
        )
        hitboxes = [
            mesh
            for mesh in collision_viewport.wireframe_overlays
            if mesh.key.startswith(AttackCollisionOverlay.KEY_PREFIX)
        ]
        self.assertEqual(len(hitboxes), 8)
        self.assertTrue(all(np.isfinite(mesh.vertices).all() for mesh in hitboxes))
        self.assertTrue(all(mesh.exclude_from_bounds for mesh in hitboxes))
        self.assertEqual(collision_viewport.scene_update_count, scene_update_count)
        controller.set_frame(28.0)
        collision_renderer.present(
            controller.sample(),
            target,
            reset_camera=False,
            motion=motion,
        )
        self.assertEqual(collision_viewport.scene_update_count, scene_update_count)
        status = collision_renderer.attack_collision_status(
            motion,
            snapshot.frame,
        )
        self.assertIn("RequestSet 0, 36", status)
        self.assertIn("MAIN_WEAPON, BODY", status)
        self.assertNotIn("not drawn", status)

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
