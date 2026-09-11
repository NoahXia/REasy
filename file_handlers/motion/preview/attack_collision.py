from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from file_handlers.rcol.rcol_file import RcolFile
from file_handlers.rcol.rcol_scene import _mesh_for_shape
from file_handlers.rcol.shape_types import (
    AABB,
    Capsule,
    Cylinder,
    OBB,
    ShapeType,
    Sphere,
)
from ui.scene.scene_model import SceneDrawMesh
from utils.hash_util import murmur3_hash_utf16le

from ..mot.model import Motion
from ..mot_clip.model import ClipNode, ClipProperty
from .model import MotionPreviewSnapshot


WOTS_WEAPON_COLLISION_TYPES = {
    0: "NONE",
    1: "SUB1_WEAPON",
    2: "MAIN_WEAPON",
    3: "SUB2_WEAPON",
    4: "SUB3_WEAPON",
    5: "SUB4_WEAPON",
    6: "SUB5_WEAPON",
    7: "SUB6_WEAPON",
    8: "SUB7_WEAPON",
    9: "SUB8_WEAPON",
    10: "SUB9_WEAPON",
    11: "MAX",
}

_ATTACK_TRACKS = frozenset((
    "AttackCollision",
    "AttackCollision_Parryable",
    "AttackCollision_Wp",
    "AttackCollision_Body",
    "AttackCollision_Gimmick",
))


@dataclass(frozen=True, slots=True)
class ActiveAttackCollision:
    track_type: str
    request_set_id: int
    attack_param_id: int
    collision_type: int | None
    start_frame: float
    end_frame: float

    @property
    def collision_type_name(self) -> str:
        if self.collision_type is None:
            return "BODY"
        return WOTS_WEAPON_COLLISION_TYPES.get(
            self.collision_type,
            f"UNKNOWN_{self.collision_type}",
        )


@dataclass(frozen=True, slots=True)
class AttackCollisionSource:
    resource_path: str
    rcol: RcolFile

    def request_set(self, request_set_id: int):
        # WOTS Motion tracks normally address RequestSetInfo.field0.  A few
        # legacy/default tracks use the table index instead, so retain the
        # index fallback only when no authored ID exists.
        authored = [
            item
            for item in self.rcol.request_sets
            if int(getattr(item.info, "field0", -1)) == request_set_id
        ]
        if authored:
            return authored[0]
        return next(
            (
                item
                for item in self.rcol.request_sets
                if int(getattr(item.info, "id", -1)) == request_set_id
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class AttackCollisionOwnerPose:
    """Joint transforms for one collision owner in preview display space."""

    matrices: dict[str, np.ndarray]
    root_matrix: np.ndarray

    @classmethod
    def from_rig(cls, rig, world_matrices) -> "AttackCollisionOwnerPose":
        world = np.asarray(world_matrices, dtype=np.float32).reshape(-1, 4, 4)
        if len(world) != len(rig.joints):
            raise ValueError("collision owner pose does not match its rig")
        matrices: dict[str, np.ndarray] = {}
        roots = []
        for index, (joint, matrix) in enumerate(zip(rig.joints, world, strict=True)):
            matrices.setdefault(joint.name.casefold(), matrix)
            matrices.setdefault(f"#{murmur3_hash_utf16le(joint.name):08x}", matrix)
            if joint.parent_index is None:
                roots.append(index)
        root_index = roots[0] if roots else 0
        return cls(matrices, world[root_index])


@dataclass(frozen=True, slots=True)
class _ShapeBinding:
    key: str
    request_set_id: int
    shape_index: int
    shape: object
    primary_joint: str
    secondary_joint: str
    mirror: bool
    weapon_collision_type: int | None = None


def collision_type_label(value: int) -> str:
    return WOTS_WEAPON_COLLISION_TYPES.get(value, f"UNKNOWN_{value}")


def selected_attack_collision(
    track_type: str,
    properties: Iterable[ClipProperty],
    frame: float,
    start_frame: float,
    end_frame: float,
) -> ActiveAttackCollision | None:
    """Resolve the collision identifiers represented by a selected lane."""
    if track_type not in _ATTACK_TRACKS:
        return None
    request_set_id = _integer_property(properties, "_RequestSetID", frame)
    attack_param_id = _integer_property(properties, "_AttackParamID", frame)
    if request_set_id is None or attack_param_id is None:
        return None
    return ActiveAttackCollision(
        track_type,
        request_set_id,
        attack_param_id,
        _integer_property(properties, "_CollisionType", frame),
        start_frame,
        end_frame,
    )


def attack_collision_detail_sections(
    source: AttackCollisionSource | None,
    event: ActiveAttackCollision,
    diagnostics: Iterable[str] = (),
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Describe the exact RCOL request set selected by a motion event."""
    owner = event.collision_type_name
    if source is None:
        message = "; ".join(diagnostics) or "matching attack RCOL was not resolved"
        return ((
            "RCOL REQUEST SET",
            (
                ("Owner", owner),
                ("RequestSetID", str(event.request_set_id)),
                ("Status", message),
            ),
        ),)
    request_set = source.request_set(event.request_set_id)
    if request_set is None:
        return ((
            "RCOL REQUEST SET",
            (
                ("Resource", source.resource_path),
                ("Owner", owner),
                ("RequestSetID", str(event.request_set_id)),
                ("Status", "request set was not found in this RCOL"),
            ),
        ),)

    info = request_set.info
    group = request_set.group
    regular = tuple(getattr(group, "shapes", ()) or ()) if group is not None else ()
    mirrored = (
        tuple(getattr(group, "extra_shapes", ()) or ())
        if group is not None
        else ()
    )
    all_shapes = tuple((shape, False) for shape in regular) + tuple(
        (shape, True) for shape in mirrored
    )
    type_counts: dict[str, int] = {}
    for shape, _mirror in all_shapes:
        name = _shape_type_name(shape)
        type_counts[name] = type_counts.get(name, 0) + 1
    type_summary = ", ".join(
        f"{name} ×{count}" if count > 1 else name
        for name, count in sorted(type_counts.items())
    ) or "None"
    group_index = int(getattr(info, "group_index", -1))
    group_name = str(getattr(getattr(group, "info", None), "name", "") or "")
    group_text = f"{group_index} · {group_name}" if group_name else str(group_index)
    rows = (
        ("Resource", source.resource_path),
        ("Owner", owner),
        ("RequestSetID", str(event.request_set_id)),
        ("Request Set Name", str(getattr(info, "name", "") or "—")),
        ("Group", group_text),
        ("Shape Count", str(len(regular))),
        ("Mirrored Shape Count", str(len(mirrored))),
        ("Shape Types", type_summary),
    )
    shape_rows = tuple(
        (f"Shape {index}", _shape_summary(shape, mirror))
        for index, (shape, mirror) in enumerate(all_shapes)
    )
    sections = [("RCOL REQUEST SET", rows)]
    if shape_rows:
        sections.append(("RCOL SHAPES", shape_rows))
    return tuple(sections)


def _shape_type_name(shape) -> str:
    value = getattr(getattr(shape, "info", None), "shape_type", None)
    name = getattr(value, "name", None)
    if name:
        return name
    try:
        return ShapeType(int(value)).name
    except (TypeError, ValueError):
        return str(value)


def _shape_summary(shape, mirror: bool) -> str:
    info = shape.info
    parts = [_shape_type_name(shape)]
    if mirror:
        parts.append("mirrored")
    if info.name:
        parts.append(info.name)
    joints = [
        value
        for value in (info.primary_joint_name_str, info.secondary_joint_name_str)
        if value
    ]
    if joints:
        parts.append("joints " + " → ".join(joints))
    payload = shape.shape
    if isinstance(payload, (Capsule, Cylinder)):
        parts.append(
            f"start {_vector_text(payload.start)}, end {_vector_text(payload.end)}, "
            f"radius {_number_text(payload.radius)}"
        )
    elif isinstance(payload, Sphere):
        parts.append(
            f"center {_vector_text(payload.center)}, radius {_number_text(payload.radius)}"
        )
    elif isinstance(payload, AABB):
        parts.append(f"min {_vector_text(payload.min)}, max {_vector_text(payload.max)}")
    elif isinstance(payload, OBB):
        parts.append(f"extent {_vector_text(payload.extent)}")
    return " · ".join(parts)


def _number_text(value) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _vector_text(values) -> str:
    return "(" + ", ".join(_number_text(value) for value in values) + ")"


def attack_rcol_resource_candidates(anchor_path: str) -> tuple[str, ...]:
    normalized = str(anchor_path or "").replace("\\", "/")
    lowered = normalized.casefold()
    if "/motion/player/" in f"/{lowered}":
        return (
            "natives/stm/GameDesign/Action/Player/Collision/Collider/PlayerAttack.rcol.37",
        )
    match = re.search(r"(?:^|/)motion/enemy/(em\d+)/(\d{2})(?:/|$)", lowered)
    if match is None:
        return ()
    actor = match.group(1)
    variant = match.group(2)
    display_actor = actor[:1].upper() + actor[1:]
    return (
        "natives/stm/GameDesign/Action/Enemy/"
        f"{display_actor}/{variant}/Collision/Collider/"
        f"{display_actor}_{variant}_Attack.rcol.37",
    )


def load_attack_collision_source(
    anchor_path: str,
    resource_loader,
    *,
    type_registry=None,
) -> tuple[AttackCollisionSource | None, tuple[str, ...]]:
    candidates = attack_rcol_resource_candidates(anchor_path)
    if not candidates:
        return None, ()
    diagnostics: list[str] = []
    for candidate in candidates:
        source, error = load_attack_collision_resource(
            candidate,
            resource_loader,
            type_registry=type_registry,
        )
        if source is not None:
            return source, tuple(diagnostics)
        diagnostics.extend(error)
    return None, tuple(diagnostics)


def load_attack_collision_resource(
    resource_path: str,
    resource_loader,
    *,
    type_registry=None,
) -> tuple[AttackCollisionSource | None, tuple[str, ...]]:
    hit = resource_loader(resource_path)
    if hit is None:
        return None, (f"attack RCOL was not found: {resource_path}",)
    resolved_path, data = hit
    try:
        rcol = RcolFile()
        rcol.type_registry = type_registry
        if not rcol.read(data, file_version=37, file_path=resolved_path):
            raise ValueError("RCOL parser rejected the file")
        return AttackCollisionSource(resolved_path, rcol), ()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, (f"could not parse attack RCOL {resolved_path!r}: {exc}",)


def active_attack_collisions(
    motion: Motion | None,
    frame: float,
) -> tuple[ActiveAttackCollision, ...]:
    if motion is None:
        return ()
    result: list[ActiveAttackCollision] = []
    for sequence in motion.sequences:
        total = max(0.0, float(sequence.clip.total_frame))
        for node in _walk_nodes(sequence.clip.root):
            track_type = node.name.rsplit(".", 1)[-1].removesuffix("Track")
            if track_type not in _ATTACK_TRACKS:
                continue
            on_clip_box = _find_property(node.properties, "_OnClipBox")
            if on_clip_box is None:
                # CompactMotClip exposes the nested cOnClipBox payload as _On.
                on_clip_box = _find_property(node.properties, "_On")
            if on_clip_box is None:
                continue
            start = max(0.0, float(on_clip_box.start_frame))
            end = min(total, max(start, float(on_clip_box.end_frame)))
            if not _interval_active(frame, start, end, total):
                continue
            request_set_id = _integer_property(node.properties, "_RequestSetID", frame)
            attack_param_id = _integer_property(node.properties, "_AttackParamID", frame)
            if request_set_id is None or attack_param_id is None:
                continue
            collision_type = _integer_property(node.properties, "_CollisionType", frame)
            result.append(
                ActiveAttackCollision(
                    track_type,
                    request_set_id,
                    attack_param_id,
                    collision_type,
                    start,
                    end,
                )
            )
    return tuple(result)


class AttackCollisionOverlay:
    """Build and pose the RCOL shapes selected by active MOT events."""

    KEY_PREFIX = "motion-preview:attack-hitbox:"

    def __init__(
        self,
        source: AttackCollisionSource | None,
        weapon_sources: dict[int, AttackCollisionSource] | None = None,
    ):
        self.source = source
        self.weapon_sources = dict(weapon_sources or {})
        self._bindings: tuple[_ShapeBinding, ...] = ()

    def signature(
        self,
        motion: Motion | None,
        frame: float,
        weapon_poses: dict[int, AttackCollisionOwnerPose] | None = None,
    ) -> tuple[int, ...]:
        weapon_poses = weapon_poses or {}
        return tuple(sorted({
            self._signature_value(event)
            for event in active_attack_collisions(motion, frame)
            if self.supports_event(event, weapon_poses)
            if self.source_for_event(event).request_set(event.request_set_id) is not None
        }))

    def source_for_event(
        self,
        event: ActiveAttackCollision,
    ) -> AttackCollisionSource | None:
        if event.track_type == "AttackCollision_Wp":
            return self.weapon_sources.get(int(event.collision_type or 0))
        return self.source

    def supports_event(
        self,
        event: ActiveAttackCollision,
        weapon_poses: dict[int, AttackCollisionOwnerPose] | None = None,
    ) -> bool:
        source = self.source_for_event(event)
        if source is None:
            return False
        if event.track_type != "AttackCollision_Wp":
            return True
        return int(event.collision_type or 0) in (weapon_poses or {})

    @staticmethod
    def _signature_value(event: ActiveAttackCollision) -> int:
        if event.track_type != "AttackCollision_Wp":
            return event.request_set_id
        owner = int(event.collision_type or 0) + 1
        return (owner << 32) | (event.request_set_id & 0xFFFFFFFF)

    def meshes(
        self,
        motion: Motion | None,
        snapshot: MotionPreviewSnapshot,
        weapon_poses: dict[int, AttackCollisionOwnerPose] | None = None,
    ) -> list[SceneDrawMesh]:
        weapon_poses = weapon_poses or {}
        bindings: list[_ShapeBinding] = []
        meshes: list[SceneDrawMesh] = []
        seen = set()
        for event in active_attack_collisions(motion, snapshot.frame):
            if not self.supports_event(event, weapon_poses):
                continue
            owner_key = (
                int(event.collision_type or 0)
                if event.track_type == "AttackCollision_Wp"
                else -1
            )
            request_key = owner_key, event.request_set_id
            if request_key in seen:
                continue
            seen.add(request_key)
            source = self.source_for_event(event)
            assert source is not None
            request_id = event.request_set_id
            request_set = source.request_set(request_id)
            group = getattr(request_set, "group", None)
            if group is None:
                continue
            for mirror, shapes in (
                (False, getattr(group, "shapes", ()) or ()),
                (True, getattr(group, "extra_shapes", ()) or ()),
            ):
                for shape_index, shape in enumerate(shapes):
                    key = (
                        f"{self.KEY_PREFIX}{owner_key}:{request_id}:"
                        f"{'m' if mirror else 'r'}:{shape_index}"
                    )
                    binding = _ShapeBinding(
                        key,
                        request_id,
                        shape_index,
                        shape,
                        str(getattr(shape.info, "primary_joint_name_str", "") or ""),
                        str(getattr(shape.info, "secondary_joint_name_str", "") or ""),
                        mirror,
                        (
                            int(event.collision_type or 0)
                            if event.track_type == "AttackCollision_Wp"
                            else None
                        ),
                    )
                    geometry = _shape_world_geometry(
                        binding,
                        snapshot,
                        weapon_poses,
                    )
                    if geometry is None:
                        continue
                    vertices, indices = geometry
                    bindings.append(binding)
                    meshes.append(
                        SceneDrawMesh(
                            key=key,
                            vertices=vertices,
                            indices=indices,
                            color=(1.0, 0.18 if not mirror else 0.42, 0.04),
                            force_solid=True,
                            ignore_highlight_filter=True,
                            exclude_from_bounds=True,
                        )
                    )
        self._bindings = tuple(bindings)
        return meshes

    def geometry_updates(
        self,
        snapshot: MotionPreviewSnapshot,
        weapon_poses: dict[int, AttackCollisionOwnerPose] | None = None,
    ) -> dict[str, np.ndarray]:
        weapon_poses = weapon_poses or {}
        result = {}
        for binding in self._bindings:
            geometry = _shape_world_geometry(binding, snapshot, weapon_poses)
            if geometry is not None:
                result[binding.key] = geometry[0]
        return result


def attack_collision_geometry_diagnostic(
    event: ActiveAttackCollision,
    *,
    weapon_source_available: bool = False,
    weapon_owner_available: bool = False,
) -> str | None:
    """Explain why an active event cannot be drawn in the actor preview.

    WOTS ``AttackCollision_Wp`` request sets belong to a separate weapon
    collision owner. Their geometry can only be drawn when both that weapon's
    Attack RCOL and its animated owner transform are available.
    """
    if event.track_type == "AttackCollision_Wp":
        if not weapon_source_available:
            return (
                f"{event.collision_type_name} uses weapon-local collider space; "
                "the matching weapon Attack RCOL is unavailable"
            )
        if weapon_owner_available:
            return None
        return (
            f"{event.collision_type_name} uses weapon-local collider space; "
            "weapon owner transform is unavailable, so geometry is not drawn"
        )
    return None


def _walk_nodes(root: ClipNode) -> Iterable[ClipNode]:
    for node in root.children:
        yield node
        yield from _walk_nodes(node)


def _find_property(
    properties: Iterable[ClipProperty],
    name: str,
) -> ClipProperty | None:
    for prop in properties:
        if prop.name == name:
            return prop
        nested = _find_property(prop.children, name)
        if nested is not None:
            return nested
    return None


def _integer_property(
    properties: Iterable[ClipProperty],
    name: str,
    frame: float,
) -> int | None:
    prop = _find_property(properties, name)
    if prop is None or not prop.keys:
        return None
    keys = sorted(prop.keys, key=lambda item: item.frame)
    key = max((item for item in keys if item.frame <= frame), key=lambda item: item.frame, default=keys[0])
    value = key.value
    return int(value) if isinstance(value, (bool, int, float)) else None


def _interval_active(frame: float, start: float, end: float, total: float) -> bool:
    if abs(end - start) < 1e-6:
        return abs(frame - start) < 0.5
    return start <= frame < end or (frame == total and end == total)


def _actor_owner_pose(snapshot: MotionPreviewSnapshot) -> AttackCollisionOwnerPose:
    matrices = np.asarray(snapshot.pose.world_matrices, dtype=np.float32).reshape(-1, 4, 4).copy()
    parents = {child: parent for parent, child in snapshot.bone_pairs}
    root_deltas = dict(snapshot.root_deltas)
    for index in range(len(matrices)):
        root = index
        while root in parents:
            root = parents[root]
        delta = root_deltas.get(root)
        if delta is not None:
            matrices[index, :3, 3] -= np.asarray(delta, dtype=np.float32)
    result: dict[str, np.ndarray] = {}
    for name, matrix in zip(snapshot.joint_names, matrices, strict=True):
        result.setdefault(name.casefold(), matrix)
        result.setdefault(f"#{murmur3_hash_utf16le(name):08x}", matrix)
    roots = sorted(set(range(len(matrices))) - set(parents))
    root_index = roots[0] if roots else 0
    return AttackCollisionOwnerPose(result, matrices[root_index])


def _binding_matrix(
    matrices: dict[str, np.ndarray],
    name: str,
    name_hash: int,
) -> np.ndarray | None:
    if name:
        result = matrices.get(name.casefold())
        if result is not None:
            return result
    if name_hash:
        return matrices.get(f"#{int(name_hash) & 0xFFFFFFFF:08x}")
    return None


def _transform_point(point, matrix: np.ndarray | None) -> np.ndarray:
    value = np.asarray((*point, 1.0), dtype=np.float32)
    return (value @ matrix)[:3] if matrix is not None else value[:3]


def _transform_vertices_row(
    vertices: np.ndarray,
    matrix: np.ndarray | None,
) -> np.ndarray:
    if matrix is None or vertices.size == 0:
        return vertices
    hom = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float32)],
        axis=1,
    )
    return (hom @ matrix)[:, :3].astype(np.float32)


def _matrix_scale(matrix: np.ndarray | None) -> float:
    if matrix is None:
        return 1.0
    values = np.linalg.norm(matrix[:3, :3], axis=0)
    return float(np.mean(values)) if np.isfinite(values).all() else 1.0


def _shape_world_geometry(
    binding: _ShapeBinding,
    snapshot: MotionPreviewSnapshot,
    weapon_poses: dict[int, AttackCollisionOwnerPose] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    shape_record = binding.shape
    payload = getattr(shape_record, "shape", None)
    if payload is None:
        return None
    owner = (
        (weapon_poses or {}).get(binding.weapon_collision_type)
        if binding.weapon_collision_type is not None
        else _actor_owner_pose(snapshot)
    )
    if owner is None:
        return None
    matrices = owner.matrices
    info = shape_record.info
    primary = _binding_matrix(
        matrices,
        binding.primary_joint,
        int(getattr(info, "primary_joint_name_hash", 0) or 0),
    )
    secondary = _binding_matrix(
        matrices,
        binding.secondary_joint,
        int(getattr(info, "secondary_joint_name_hash", 0) or 0),
    )
    primary = primary if primary is not None else owner.root_matrix
    secondary = secondary if secondary is not None else owner.root_matrix
    if isinstance(payload, (Capsule, Cylinder)) and secondary is not None:
        from file_handlers.rcol.rcol_scene import _capsule

        start = _transform_point(payload.start, primary)
        end = _transform_point(payload.end, secondary)
        radius = float(payload.radius) * max(
            1e-6,
            (_matrix_scale(primary) + _matrix_scale(secondary)) * 0.5,
        )
        return _capsule(start, end, radius)
    geometry = _mesh_for_shape(payload)
    if geometry is None:
        return None
    vertices, indices = geometry
    return _transform_vertices_row(vertices, primary), indices
