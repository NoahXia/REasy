from __future__ import annotations

"""Build an animated WOTS character target from a Character PFB assembly."""

from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace

from file_handlers.motion.evaluation.mesh_adapter import rig_from_re_engine_mesh
from file_handlers.motion.preview.target import RigPreviewPart, RigPreviewTarget
from file_handlers.mesh.mesh_handler import MeshHandler
from utils.resource_file_utils import (
    normalize_resource_path,
    resource_context_for_handler,
    resolve_handler_resource_data,
)

from .rsz_file import RszFile
from .scn_scene_loader import ScnSceneLoader, ScnSceneSource


_CHARACTER_PFB_PREFIX = (
    "natives/stm/GameDesign/Action/Enemy/_Prefab/Character/"
)


@dataclass(frozen=True, slots=True)
class WotsCharacterPfbTargetResult:
    target: RigPreviewTarget
    resource_path: str
    part_paths: tuple[str, ...]
    weapon_path: str = ""
    weapon_candidates: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


def inferred_wots_enemy_pfb_path(source_path: str) -> str:
    """Infer the Character PFB belonging to an enemy action or motion resource."""
    parts = normalize_resource_path(source_path).split("/")
    lowered = [part.casefold() for part in parts]
    # Paired interaction animations for the player live under the owning
    # enemy's Motion directory. Their ``plw_`` basename is the authoritative
    # actor marker and must not select that enemy's Character PFB.
    if lowered and lowered[-1].startswith("plw_"):
        return ""
    try:
        enemy_index = next(
            index
            for index, part in enumerate(lowered[:-2])
            if part == "enemy"
            and index > 0
            and lowered[index - 1] in {"action", "motion"}
        )
    except StopIteration:
        return ""
    actor = parts[enemy_index + 1]
    variant = parts[enemy_index + 2]
    if not actor.casefold().startswith("em") or not variant.isdecimal():
        return ""
    return f"{_CHARACTER_PFB_PREFIX}{actor}_{variant}.pfb.18"


def discover_wots_character_pfb_paths(owner_handler, source_path: str = "") -> tuple[str, ...]:
    """List known Character PFBs, with the source enemy's PFB first."""
    inferred = inferred_wots_enemy_pfb_path(source_path)
    paths: dict[str, str] = {}

    def add(value: str) -> None:
        normalized = normalize_resource_path(value)
        key = normalized.casefold()
        if (
            key.startswith(_CHARACTER_PFB_PREFIX.casefold())
            and ".pfb." in key
        ):
            paths.setdefault(key, normalized)

    if inferred:
        add(inferred)
    context = resource_context_for_handler(owner_handler)
    reader = getattr(context, "pak_cached_reader", None) if context else None
    if reader is not None and bool(getattr(reader, "cache_ready", False)):
        try:
            known = (
                reader.cached_known_paths()
                if callable(getattr(reader, "cached_known_paths", None))
                else reader.cached_paths(include_unknown=False)
            )
            for path in known:
                add(str(path))
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    for root_value in (
        getattr(context, "project_dir", "") if context else "",
        getattr(context, "unpacked_dir", "") if context else "",
    ):
        if not root_value:
            continue
        folder = Path(root_value, *_CHARACTER_PFB_PREFIX.split("/"))
        if not folder.is_dir():
            continue
        try:
            for item in folder.iterdir():
                if item.is_file() and ".pfb." in item.name.casefold():
                    add(f"{_CHARACTER_PFB_PREFIX}{item.name}")
        except OSError:
            pass

    ordered = sorted(paths.values(), key=str.casefold)
    if inferred:
        inferred_key = inferred.casefold()
        ordered.sort(key=lambda item: item.casefold() != inferred_key)
    return tuple(ordered)


def discover_wots_character_weapon_paths(
    owner_handler,
    character_mesh_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Find enemy item meshes whose asset number matches the PFB's character."""
    character_ids = {
        match.group(1)
        for path in character_mesh_paths
        for match in (
            re.search(r"/ch\d+/ch(\d{3})(?:_|/)", f"/{normalize_resource_path(path)}", re.I),
        )
        if match is not None
    }
    if not character_ids:
        return ()
    context = resource_context_for_handler(owner_handler)
    reader = getattr(context, "pak_cached_reader", None) if context else None
    if reader is None or not bool(getattr(reader, "cache_ready", False)):
        return ()
    try:
        known = (
            reader.cached_known_paths()
            if callable(getattr(reader, "cached_known_paths", None))
            else reader.cached_paths(include_unknown=False)
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return ()

    matches = []
    for value in known:
        path = normalize_resource_path(str(value))
        lowered = path.casefold()
        if not lowered.startswith("natives/stm/art/model/item/") or ".mesh." not in lowered:
            continue
        if not any(
            re.search(rf"/it\d+_{re.escape(character_id)}\d(?:/|_)", lowered)
            for character_id in character_ids
        ):
            continue
        matches.append(path)
    return tuple(sorted(dict.fromkeys(matches), key=str.casefold))


def _renderable_label(graph, renderable) -> tuple[str, str]:
    document = graph.documents.get(renderable.source_object_id.document_id)
    source_object = (
        document.objects.get(renderable.source_object_id)
        if document is not None
        else None
    )
    label = str(getattr(source_object, "name", "") or "").strip()
    if not label:
        label = Path(normalize_resource_path(renderable.mesh_path)).name
    parent_joint = ""
    transform = getattr(source_object, "transform", None)
    if transform is not None:
        parent_joint = str(getattr(transform, "parent_joint", "") or "").strip()
    return label, parent_joint


def load_wots_character_pfb_target(
    owner_handler,
    resource_path: str,
    parent=None,
    *,
    weapon_path: str | None = None,
    type_registry=None,
) -> WotsCharacterPfbTargetResult:
    """Resolve a PFB and convert its visible model assembly to a motion target."""
    requested_path = str(resource_path or "").strip()
    if not requested_path:
        raise ValueError("Select a WOTS Character PFB first.")

    direct = Path(requested_path)
    hit = (
        (str(direct), direct.read_bytes())
        if direct.is_file()
        else resolve_handler_resource_data(
            owner_handler,
            requested_path,
            parent,
            allow_selection_dialog=False,
        )
    )
    if hit is None:
        raise ValueError(f"Character PFB was not found: {requested_path}")
    filepath, data = hit
    if data[:4] != b"PFB\0":
        raise ValueError(f"Selected resource is not a PFB: {filepath}")

    owner_rsz = getattr(owner_handler, "rsz_file", None)
    registry = type_registry or getattr(owner_rsz, "type_registry", None)
    if registry is None:
        raise ValueError("The owning document has no WOTS RSZ type registry.")
    game_version = str(
        getattr(owner_rsz, "game_version", "")
        or getattr(owner_handler, "game_version", "")
        or "OnimushaWOTS"
    )
    from .wots_registry import apply_wots_registry_overlay

    registry = apply_wots_registry_overlay(registry, game_version)
    parsed = RszFile()
    parsed.filepath = filepath
    parsed.game_version = game_version
    parsed.type_registry = registry
    try:
        parsed.read(data, validate_type_registry=True)
    except Exception as exc:
        raise ValueError(f"Could not parse Character PFB: {exc}") from exc

    context = resource_context_for_handler(owner_handler)
    proxy = SimpleNamespace(
        app=getattr(owner_handler, "app", None),
        filepath=filepath,
        rsz_file=parsed,
        type_registry=registry,
        game_version=game_version,
        resource_context=context,
    )
    source = ScnSceneSource(
        path=requested_path,
        handler=proxy,
        label=Path(requested_path).name,
        game_version=game_version,
    )
    loader = ScnSceneLoader()
    graphs = loader.build_graphs([source], max_depth=8)
    renderables = [
        (graph, renderable)
        for graph in graphs
        for renderable in graph.renderables
        if renderable.visible_by_default and renderable.mesh_path
    ]
    if not renderables:
        raise ValueError("Character PFB contains no visible Mesh or Montage parts.")

    loaded = []
    diagnostics: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for graph, renderable in renderables:
        label, parent_joint = _renderable_label(graph, renderable)
        mesh_path = normalize_resource_path(renderable.mesh_path)
        key = (
            mesh_path.casefold(),
            normalize_resource_path(renderable.mdf_path).casefold(),
            parent_joint.casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        resolved = loader.resolve_resource_for_source(
            loader.source_for_graph(graph),
            mesh_path,
        )
        if resolved is None or resolved.data is None:
            diagnostics.append(f"Missing mesh: {mesh_path}")
            continue
        try:
            mesh_handler = MeshHandler.from_bytes(
                resolved.path,
                resolved.data,
                app=getattr(owner_handler, "app", None),
                resource_context=context,
                game_version=game_version,
            )
            mesh = mesh_handler.mesh
            if mesh is None:
                raise ValueError("mesh parser returned no model")
            rig = rig_from_re_engine_mesh(mesh)
        except Exception as exc:
            diagnostics.append(f"Skipped {mesh_path}: {exc}")
            continue
        is_item = "/art/model/item/" in mesh_path.casefold()
        loaded.append((
            label,
            mesh_path,
            normalize_resource_path(renderable.mdf_path),
            parent_joint if is_item else "",
            2 if is_item else None,
            mesh_handler,
            mesh,
            rig,
        ))

    if not loaded:
        detail = "; ".join(diagnostics) or "no skinnable meshes were resolved"
        raise ValueError(f"Character PFB could not build an animated model: {detail}")

    character_mesh_paths = tuple(item[1] for item in loaded)
    weapon_candidates = discover_wots_character_weapon_paths(
        owner_handler,
        character_mesh_paths,
    )
    selected_weapon = (
        weapon_candidates[0]
        if weapon_path is None and weapon_candidates
        else normalize_resource_path(weapon_path or "")
    )
    if selected_weapon:
        if selected_weapon in character_mesh_paths:
            selected_weapon = ""
        else:
            hit = resolve_handler_resource_data(
                owner_handler,
                selected_weapon,
                parent,
                allow_selection_dialog=False,
            )
            if hit is None:
                diagnostics.append(f"Missing weapon mesh: {selected_weapon}")
                selected_weapon = ""
            else:
                weapon_filepath, weapon_data = hit
                try:
                    weapon_handler = MeshHandler.from_bytes(
                        weapon_filepath,
                        weapon_data,
                        app=getattr(owner_handler, "app", None),
                        resource_context=context,
                        game_version=game_version,
                    )
                    weapon_mesh = weapon_handler.mesh
                    if weapon_mesh is None:
                        raise ValueError("mesh parser returned no model")
                    weapon_rig = rig_from_re_engine_mesh(weapon_mesh)
                    loaded.append((
                        "Weapon",
                        selected_weapon,
                        "",
                        "R_Wep",
                        2,
                        weapon_handler,
                        weapon_mesh,
                        weapon_rig,
                    ))
                except Exception as exc:
                    diagnostics.append(f"Skipped weapon {selected_weapon}: {exc}")
                    selected_weapon = ""

    # A body mesh normally owns the most complete skeleton. Put it first so it
    # becomes the animation target rig; head, hair, armor and weapon meshes are
    # remapped onto that rig by joint name.
    loaded.sort(
        key=lambda item: (
            bool(item[3]),
            -len(item[7].joints),
            "/art/model/item/" in item[1].casefold(),
            item[0].casefold(),
        )
    )
    parts = tuple(
        RigPreviewPart(
            label=item[0],
            rig=item[7],
            mesh=item[6],
            handler=item[5],
            material_scope=f"pfb:{index}",
            explicit_mdf_path=item[2],
            attachment_joint=item[3],
            weapon_collision_type=item[4],
        )
        for index, item in enumerate(loaded)
    )
    primary = parts[0]
    target = RigPreviewTarget(
        label=f"WOTS Character PFB · {Path(requested_path).name}",
        rig=primary.rig,
        mesh=primary.mesh,
        handler=primary.handler,
        parts=parts,
    )
    return WotsCharacterPfbTargetResult(
        target,
        requested_path,
        tuple(item[1] for item in loaded),
        selected_weapon,
        weapon_candidates,
        tuple(diagnostics),
    )
