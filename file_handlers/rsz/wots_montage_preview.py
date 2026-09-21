from __future__ import annotations

"""Read-only WOTS Montage model assembly for SCN/PFB previews."""

from dataclasses import dataclass
from typing import Callable, Mapping

from .rsz_data_types import (
    ArrayData,
    GuidData,
    ObjectData,
    ResourceData,
    StringData,
    StructData,
    UserDataData,
)


WOTS_MONTAGE = "app.montage.Montage"
WOTS_MONTAGE_PARTS = "app.montage.MontageParts"
WOTS_DATA_CONTAINER = "app.montage.data.MontageDataContainer"
WOTS_RANDOM_STANDARD_DATA = "app.montage.data.RandomChoiceStandardMontageData"
WOTS_STANDARD_DATA = "app.montage.data.StandardMontageData"
WOTS_MODEL_DATA = "app.montage.data.MontagePartsModelData"


@dataclass(frozen=True, slots=True)
class WotsMontagePart:
    source_object: object
    source_component: object
    parts_id: str
    label: str
    mesh_path: str
    material_path: str
    holder_id: int
    merge_priority: int


@dataclass(frozen=True, slots=True)
class WotsMontageDiagnostic:
    code: str
    message: str
    source_object: object | None = None
    source_component: object | None = None
    path: str = ""


@dataclass(frozen=True, slots=True)
class WotsMontageAssembly:
    parts: tuple[WotsMontagePart, ...]
    diagnostics: tuple[WotsMontageDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _Node:
    rsz: object
    instance_id: int
    fields: Mapping[str, object]
    type_name: str


UserDataLoader = Callable[[object, UserDataData], object | None]


def _type_name(rsz, instance_id: int) -> str:
    infos = getattr(rsz, "instance_infos", ()) or ()
    registry = getattr(rsz, "type_registry", None)
    if registry is None or not 0 <= instance_id < len(infos):
        return ""
    info = registry.get_type_info(int(getattr(infos[instance_id], "type_id", 0)))
    return str((info or {}).get("name", "") or "")


def _is_a(rsz, type_name: str, base_type: str) -> bool:
    if not base_type or type_name == base_type:
        return True
    registry = getattr(rsz, "type_registry", None)
    try:
        return base_type in (registry.getTypeParents(type_name) or ())
    except (AttributeError, TypeError, ValueError):
        return False


def _node(rsz, instance_id: int) -> _Node | None:
    fields = getattr(rsz, "parsed_elements", {}).get(instance_id, {})
    if not isinstance(fields, Mapping):
        return None
    return _Node(rsz, int(instance_id), fields, _type_name(rsz, int(instance_id)))


def _root_node(rsz, expected_type: str) -> _Node | None:
    preferred = tuple(reversed(tuple(getattr(rsz, "object_table", ()) or ())))
    remaining = tuple(reversed(tuple(getattr(rsz, "parsed_elements", {}).keys())))
    seen: set[int] = set()
    for value in (*preferred, *remaining):
        instance_id = int(value)
        if instance_id in seen:
            continue
        seen.add(instance_id)
        candidate = _node(rsz, instance_id)
        if candidate is not None and _is_a(rsz, candidate.type_name, expected_type):
            return candidate
    return None


def _resolve_node(
    owner_rsz,
    value,
    load_user_data: UserDataLoader,
) -> _Node | None:
    if isinstance(value, UserDataData):
        if str(value.string or "").strip().rstrip("\0"):
            loaded = load_user_data(owner_rsz, value)
            return _root_node(loaded, value.orig_type) if loaded is not None else None
        return _node(owner_rsz, int(value.value or 0))
    if isinstance(value, ObjectData):
        return _node(owner_rsz, int(value.value or 0))
    if isinstance(value, StructData):
        fields = next((item for item in value.values if isinstance(item, Mapping)), None)
        return _Node(owner_rsz, 0, fields, value.orig_type) if fields is not None else None
    if isinstance(value, Mapping):
        return _Node(owner_rsz, 0, value, "")
    return None


def _array_values(value) -> tuple[object, ...]:
    return tuple(value.values) if isinstance(value, (ArrayData, StructData)) else ()


def _scalar(value, default=0):
    return getattr(value, "value", value) if value is not None else default


def _guid(value) -> str:
    if isinstance(value, GuidData):
        return str(value.guid_str or "").strip().casefold()
    return str(getattr(value, "guid_str", "") or "").strip().casefold()


def _resource_path(value) -> str:
    if isinstance(value, (ResourceData, StringData)):
        return str(value.value or "").strip().rstrip("\0").replace("\\", "/")
    return ""


def _parts_label(rsz, fields, load_user_data: UserDataLoader) -> str:
    label = _resolve_node(rsz, fields.get("_PartsLabel"), load_user_data)
    return _guid(label.fields.get("_Id")) if label is not None else ""


def _standard_data_node(
    node: _Node,
    load_user_data: UserDataLoader,
) -> _Node | None:
    if _is_a(node.rsz, node.type_name, WOTS_STANDARD_DATA):
        return node
    if node.type_name != WOTS_RANDOM_STANDARD_DATA:
        return None
    for value in _array_values(node.fields.get("_MontageDatas")):
        candidate = _resolve_node(node.rsz, value, load_user_data)
        if candidate is not None and _is_a(
            candidate.rsz,
            candidate.type_name,
            WOTS_STANDARD_DATA,
        ):
            return candidate
    return None


def resolve_wots_montage_assembly(
    document,
    load_user_data: UserDataLoader,
) -> WotsMontageAssembly:
    """Resolve the authored default WOTS Montage costume for one PFB document."""
    rsz = document.rsz_file
    diagnostics: list[WotsMontageDiagnostic] = []
    labels: dict[str, tuple[object, object]] = {}

    for component in document.components.values():
        if component.type_name != WOTS_MONTAGE_PARTS:
            continue
        source_object = document.objects.get(component.owner)
        parts_id = _parts_label(rsz, component.fields, load_user_data)
        if not parts_id:
            diagnostics.append(
                WotsMontageDiagnostic(
                    "wots_montage_missing_parts_label",
                    f"MontageParts component {component.id.instance_id} has no PartsLabel GUID.",
                    source_object,
                    component,
                )
            )
            continue
        labels[parts_id] = (source_object, component)

    if not labels:
        return WotsMontageAssembly((), tuple(diagnostics))

    resolved_parts: dict[str, WotsMontagePart] = {}
    montage_components = tuple(
        component
        for component in document.components.values()
        if component.type_name == WOTS_MONTAGE
    )
    for montage in montage_components:
        source_object = document.objects.get(montage.owner)
        container_ref = montage.fields.get("_DataContainer")
        container = _resolve_node(rsz, container_ref, load_user_data)
        if container is None:
            path = str(getattr(container_ref, "string", "") or "").rstrip("\0")
            diagnostics.append(
                WotsMontageDiagnostic(
                    "wots_montage_missing_container",
                    "Unable to resolve the Montage DataContainer.",
                    source_object,
                    montage,
                    path,
                )
            )
            continue

        holders: list[tuple[int, _Node]] = []
        for value in _array_values(container.fields.get("_Holders")):
            holder = _resolve_node(container.rsz, value, load_user_data)
            if holder is not None:
                holders.append((int(_scalar(holder.fields.get("_Id"), 0) or 0), holder))
        if not holders:
            diagnostics.append(
                WotsMontageDiagnostic(
                    "wots_montage_empty_container",
                    "Montage DataContainer has no AssetHolder entries.",
                    source_object,
                    montage,
                )
            )
            continue

        holder_id, holder = min(holders, key=lambda item: (item[0] != 0, item[0]))
        data_ref = holder.fields.get("_Data")
        data_node = _resolve_node(holder.rsz, data_ref, load_user_data)
        standard = (
            _standard_data_node(data_node, load_user_data)
            if data_node is not None
            else None
        )
        if standard is None:
            path = str(getattr(data_ref, "string", "") or "").rstrip("\0")
            diagnostics.append(
                WotsMontageDiagnostic(
                    "wots_montage_unsupported_selector",
                    f"Holder {holder_id} has no supported Standard Montage data.",
                    source_object,
                    montage,
                    path,
                )
            )
            continue

        for value in _array_values(standard.fields.get("_MontagePartsDatas")):
            part_data = _resolve_node(standard.rsz, value, load_user_data)
            if part_data is None:
                continue
            parts_id = _guid(part_data.fields.get("_PartsId"))
            target = labels.get(parts_id)
            if target is None:
                diagnostics.append(
                    WotsMontageDiagnostic(
                        "wots_montage_unknown_parts_id",
                        f"Montage data references unknown PartsId {parts_id or '<empty>'}.",
                        source_object,
                        montage,
                    )
                )
                continue

            # Empty ModelData is the authored way to disable a part in this costume.
            model_ref = part_data.fields.get("_ModelData")
            if not int(getattr(model_ref, "value", 0) or 0) and not str(
                getattr(model_ref, "string", "") or ""
            ).strip().rstrip("\0"):
                continue
            model_data = _resolve_node(part_data.rsz, model_ref, load_user_data)
            if model_data is None:
                path = str(getattr(model_ref, "string", "") or "").rstrip("\0")
                diagnostics.append(
                    WotsMontageDiagnostic(
                        "wots_montage_missing_model_data",
                        f"Unable to resolve ModelData for PartsId {parts_id}.",
                        target[0],
                        target[1],
                        path,
                    )
                )
                continue
            mesh_path = _resource_path(model_data.fields.get("_Mesh"))
            material_path = _resource_path(model_data.fields.get("_Material"))
            if not mesh_path:
                diagnostics.append(
                    WotsMontageDiagnostic(
                        "wots_montage_missing_mesh",
                        f"ModelData for PartsId {parts_id} has no MESH resource.",
                        target[0],
                        target[1],
                    )
                )
                continue
            priority = int(_scalar(part_data.fields.get("_MergePriority"), 0) or 0)
            candidate = WotsMontagePart(
                target[0],
                target[1],
                parts_id,
                str(getattr(target[0], "name", "") or parts_id),
                mesh_path,
                material_path,
                holder_id,
                priority,
            )
            previous = resolved_parts.get(parts_id)
            if previous is None or candidate.merge_priority >= previous.merge_priority:
                resolved_parts[parts_id] = candidate

    if not montage_components:
        diagnostics.append(
            WotsMontageDiagnostic(
                "wots_montage_missing_component",
                "This PFB has MontageParts nodes but no Montage component.",
            )
        )
    return WotsMontageAssembly(
        tuple(resolved_parts.values()),
        tuple(diagnostics),
    )
