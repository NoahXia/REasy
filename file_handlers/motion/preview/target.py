from __future__ import annotations

from dataclasses import dataclass

from file_handlers.mesh.mesh_handler import MeshHandler
from utils.resource_file_utils import ResourceResolutionContext

from ..evaluation.mesh_adapter import rig_from_re_engine_mesh
from ..evaluation.model import Rig


@dataclass(frozen=True, slots=True)
class RigPreviewPart:
    """One independently skinned mesh in a composite preview target."""

    label: str
    rig: Rig
    mesh: object
    handler: MeshHandler
    material_scope: str = ""
    attachment_joint: str = ""
    weapon_collision_type: int | None = None


@dataclass(frozen=True, slots=True)
class RigPreviewTarget:
    """A target rig and its optional renderable source asset."""

    label: str
    rig: Rig
    mesh: object | None = None
    handler: MeshHandler | None = None
    parts: tuple[RigPreviewPart, ...] = ()

    @property
    def render_parts(self) -> tuple[RigPreviewPart, ...]:
        if self.parts:
            return self.parts
        if self.mesh is None or self.handler is None:
            return ()
        return (
            RigPreviewPart(
                self.label,
                self.rig,
                self.mesh,
                self.handler,
            ),
        )


@dataclass(frozen=True, slots=True)
class MeshPreviewPreset:
    key: str
    label: str
    resource_paths: tuple[str, ...]
    part_labels: tuple[str, ...] = ()
    attachment_joints: tuple[str, ...] = ()
    weapon_collision_types: tuple[int | None, ...] = ()
    weapon_attack_resources: tuple[tuple[int, str], ...] = ()


WOTS_MESH_PREVIEW_PRESETS = (
    MeshPreviewPreset(
        "wots_ch001_00",
        "WOTS · ch001_00 (body + head + hair + sword + sheath)",
        (
            "natives/stm/Art/Model/Character/ch0/ch001_00/00/"
            "ch001_00_00.mesh.260209350",
            "natives/stm/Art/Model/Character/ch0/ch001_00/10/"
            "ch001_00_10.mesh.260209350",
            "natives/stm/Art/Model/Character/ch0/ch001_00/20/"
            "ch001_00_20.mesh.260209350",
            "natives/stm/Art/Model/Item/it0/it000_0000/"
            "it000_0000_00.mesh.260209350",
            "natives/stm/Art/Model/Item/it0/it000_0000/"
            "it000_0000_10.mesh.260209350",
        ),
        ("Body", "Head", "Hair", "Sword", "Sheath"),
        ("", "", "", "R_Wep", "Katana_root"),
        (None, None, None, 2, None),
        ((
            2,
            "natives/stm/GameDesign/Gimmick/Gm800/Gm800_000/Collision/"
            "Gm800_000_Attack.rcol.37",
        ),),
    ),
)


def motion_target_from_mesh_handler(
    label: str,
    handler: MeshHandler,
) -> RigPreviewTarget:
    if handler.mesh is None:
        raise ValueError("target mesh did not produce a parsed model")
    return RigPreviewTarget(
        label=label,
        rig=rig_from_re_engine_mesh(handler.mesh),
        mesh=handler.mesh,
        handler=handler,
    )


def motion_target_from_mesh_handlers(
    label: str,
    handlers: tuple[tuple[str, MeshHandler, str, int | None], ...],
) -> RigPreviewTarget:
    if not handlers:
        raise ValueError("model preset contains no mesh resources")
    parts = []
    for index, (
        part_label,
        handler,
        attachment_joint,
        weapon_collision_type,
    ) in enumerate(handlers):
        if handler.mesh is None:
            raise ValueError(
                f"preset mesh {part_label!r} did not produce a parsed model"
            )
        parts.append(
            RigPreviewPart(
                part_label,
                rig_from_re_engine_mesh(handler.mesh),
                handler.mesh,
                handler,
                material_scope=f"target:{index}",
                attachment_joint=attachment_joint,
                weapon_collision_type=weapon_collision_type,
            )
        )
    primary = parts[0]
    return RigPreviewTarget(
        label=label,
        rig=primary.rig,
        mesh=primary.mesh,
        handler=primary.handler,
        parts=tuple(parts),
    )


def load_re_engine_mesh_preset_target(
    preset: MeshPreviewPreset,
    resources: tuple[tuple[str, bytes], ...],
    *,
    app=None,
    resource_context: ResourceResolutionContext | None = None,
) -> RigPreviewTarget:
    if len(resources) != len(preset.resource_paths):
        raise ValueError(
            f"preset {preset.label!r} needs {len(preset.resource_paths)} meshes, "
            f"got {len(resources)}"
        )
    handlers = []
    for index, (filepath, data) in enumerate(resources):
        target = load_re_engine_mesh_target(
            filepath,
            data,
            app=app,
            resource_context=resource_context,
        )
        assert target.handler is not None
        part_label = (
            preset.part_labels[index]
            if index < len(preset.part_labels)
            else filepath
        )
        attachment_joint = (
            preset.attachment_joints[index]
            if index < len(preset.attachment_joints)
            else ""
        )
        weapon_collision_type = (
            preset.weapon_collision_types[index]
            if index < len(preset.weapon_collision_types)
            else None
        )
        handlers.append((
            part_label,
            target.handler,
            attachment_joint,
            weapon_collision_type,
        ))
    return motion_target_from_mesh_handlers(preset.label, tuple(handlers))


def load_re_engine_mesh_target(
    filepath: str,
    data: bytes,
    *,
    app=None,
    resource_context: ResourceResolutionContext | None = None,
) -> RigPreviewTarget:
    if not filepath:
        raise ValueError("target mesh needs a path with a numeric version suffix")
    if not MeshHandler.can_handle(data):
        raise ValueError("target is not an RE Engine mesh")
    try:
        handler = MeshHandler.from_bytes(
            filepath,
            data,
            app=app,
            resource_context=resource_context,
        )
    except Exception as exc:
        raise ValueError(f"could not parse target mesh: {exc}") from exc
    return motion_target_from_mesh_handler(filepath, handler)
