from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QBrush, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.resource_file_utils import (
    normalize_resource_path,
    resolve_handler_resource_data,
)

from .rsz_data_types import ArrayData, ObjectData, StructData, UserDataData
from .rsz_file import RszFile


BTABLE_TYPE = "ace.btable.user_data.BTable"
TABLE_TYPE = f"{BTABLE_TYPE}.cTableElement"
ROW_TYPE = f"{BTABLE_TYPE}.cRowElement"
ORDER_BANK_TYPE = "ace.btable.user_data.BTableOrderBank"
ORDER_LIST_TYPE = "ace.btable.user_data.BTableOrderList"
JUMP_ARGUMENT_TYPE = "ace.btable.cOperatorArgumentJump"


@dataclass(frozen=True, slots=True)
class BTableRow:
    index: int
    instance_id: int
    operator_hash: int
    operator_name: str
    command_hash: int
    command_name: str
    operator_argument: str = ""
    command_argument: str = ""
    jump_target_guid: str | None = None
    import_source_guid: str | None = None
    import_target_guid: str | None = None
    resolved_import: str = ""


@dataclass(frozen=True, slots=True)
class BTableNode:
    index: int
    instance_id: int
    guid: str
    rows: tuple[BTableRow, ...]
    semantic_title: str = "Behavior"
    semantic_summary: tuple[str, ...] = ()

    @property
    def title(self) -> str:
        return f"Table {self.index:03d}  {self.guid[:8]}"

    @property
    def display_title(self) -> str:
        return f"Table {self.index:03d} · {self.semantic_title}"

    @property
    def search_text(self) -> str:
        values = [self.title, self.guid]
        for row in self.rows:
            values.extend(
                (
                    row.operator_name,
                    row.command_name,
                    row.operator_argument,
                    row.command_argument,
                    row.resolved_import,
                )
            )
        return " ".join(values).casefold()


@dataclass(frozen=True, slots=True)
class BTableEdge:
    source_guid: str
    target_guid: str
    row_index: int
    label: str = ""


@dataclass(frozen=True, slots=True)
class BTableGraph:
    nodes: tuple[BTableNode, ...]
    edges: tuple[BTableEdge, ...]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BTableSemanticStep:
    row_index: int
    depth: int
    label: str
    detail: str = ""
    kind: str = "action"


_GUID_PAIR_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"\|"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_GUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

# Verified WOTS 1.0.1.0 action identifiers. Unknown GUIDs remain visible in Raw mode.
WOTS_ACTION_NAMES = {
    "4023715c-1734-4b6a-bde6-f718cdb1826e": "cChangeToOneHand",
    "029ed6ea-d2eb-4478-9b94-1c293edde1d1": "cTurnOneHandCombat",
    "05e14c01-4b4d-4399-b213-c74f49c308aa": "cAlertTargetLose",
    "224a6d6d-6ea3-4bef-90fc-f842a3b7a3a1": "cSetTypeIdle_Lookout",
    "666a1369-ffc1-4744-bb47-a56657aba7ab": "cSetTypeIdle_LookoutEnd",
    "a69eb05a-d083-4c73-82fb-260f7d57bbed": "cSetTypeIdle_GateAttack",
    "0ab12da0-835b-4c44-ab83-71b4d4282bd2": "cSetTypeWalkTurn",
    "8fa169d1-76e5-4722-86f8-8450de226e16": "cWaitAndSeeMinimumMoveOneHand",
}

_MOVE_ACTION_NAMES = {1: "IDLE", 2: "WALK", 4: "RUN", 5: "DASH", 6: "MINIMUM MOVE"}
_PERCEPTION_NAMES = {515639104: "CHARA", 789249792: "POS"}


def _argument_number(argument: str, field_name: str) -> float | None:
    match = re.search(
        rf"\b{re.escape(field_name)}=[^;(]*\((-?\d+(?:\.\d+)?)\)",
        argument,
    )
    return float(match.group(1)) if match else None


def _fmt_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:g}"


def _humanize(value: str) -> str:
    value = re.sub(r"^(Operator|Request|Check|Register)", "", value or "")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return value.strip() or "Action"


def _known_action(argument: str) -> str:
    for guid in _GUID_RE.findall(argument or ""):
        name = WOTS_ACTION_NAMES.get(guid.casefold())
        if name:
            return name
    return ""


def _condition_text(row: BTableRow) -> str:
    command = row.command_name
    argument = row.command_argument
    if command == "None":
        return "Else"
    if command == "TargetExistCheck":
        return "Target exists"
    if command == "TargetCombatZoneCheck":
        return "Inside combat zone"
    if command == "CheckTargetNpc":
        return "NPC target exists"
    if command == "IsExistTourPoint":
        return "Tour point exists"
    if command == "IsTwoHanded":
        return "Two-handed stance"
    if command == "CheckXZDistance":
        distance = _argument_number(argument, "Distance")
        method = _argument_number(argument, "CheckMethod")
        relation = "Farther than" if method == 1 else "Within"
        return f"{relation} {_fmt_number(distance)} m" if distance is not None else "XZ distance check"
    if command == "TargetDurationCheck":
        seconds = _argument_number(argument, "CheckTimeSec")
        return f"Target held for {_fmt_number(seconds)} s" if seconds is not None else "Target duration check"
    if command == "TargetPerceptionCheck":
        value = _argument_number(argument, "PerceptionType")
        name = _PERCEPTION_NAMES.get(int(value), str(int(value))) if value is not None else "unknown"
        return f"{name} perceived"
    if command == "CheckEnemyIdleFormType":
        value = _argument_number(argument, "CheckMethod")
        return f"Idle form = COMMON_{int(value)}" if value in (1, 2, 3) else "Idle form check"
    if command == "CheckEnemyTourFormType":
        value = _argument_number(argument, "CheckMethod")
        return f"Tour form = COMMON_{int(value)}" if value in (1, 2, 3) else "Tour form check"
    if command == "CheckAngle":
        width = _argument_number(argument, "AngleWidth")
        return f"Angle within {_fmt_number(width)}°" if width is not None else "Angle check"
    if command == "SetTourPoint":
        return "Tour point found"
    return _humanize(command)


def _action_text(row: BTableRow, target_titles: dict[str, str] | None = None) -> str:
    target_titles = target_titles or {}
    if row.resolved_import:
        return f"Call {row.resolved_import}"
    if row.jump_target_guid:
        target = target_titles.get(row.jump_target_guid, row.jump_target_guid[:8])
        return f"Go to {target}"
    action_name = _known_action(row.command_argument)
    if action_name:
        return action_name
    command = row.command_name
    argument = row.command_argument
    if command in ("None", ""):
        return ""
    if command.startswith("RequestMoveAction") or command == "RequestPatrolMoveActionArrival":
        move_value = _argument_number(argument, "MoveActionType")
        move = _MOVE_ACTION_NAMES.get(int(move_value), "MOVE") if move_value is not None else "MOVE"
        timeout = _argument_number(argument, "ActionTime")
        distance = _argument_number(argument, "ArrivalDist")
        height = _argument_number(argument, "ArrivalHeight")
        facts = [move]
        if timeout is not None:
            facts.append(f"{_fmt_number(timeout)} s")
        if distance is not None:
            facts.append(f"arrive {_fmt_number(distance)} m")
        if height is not None:
            facts.append(f"height {_fmt_number(height)} m")
        return " · ".join(facts)
    aliases = {
        "TargetReset": "Reset target",
        "EnemyExitPatrol": "Exit patrol",
        "SetTargetNpc": "Set NPC target",
        "SetTargetStartDir": "Use start direction",
        "RegisterMoveActionSetting": "Configure move target",
        "RegisterComboBreak": "Enable combo break",
    }
    return aliases.get(command, _humanize(command))


def _node_role(node: BTableNode) -> str:
    commands = {row.command_name for row in node.rows}
    operators = {row.operator_name for row in node.rows}
    if node.index == 0 and len(node.rows) <= 3 and "OperatorJump" in operators:
        return "Entry"
    if {"TargetExistCheck", "TargetCombatZoneCheck"} <= commands:
        return "Target routing"
    if {"TargetPerceptionCheck", "TargetDurationCheck"} <= commands:
        return "Approach target"
    if {"CheckTargetNpc", "CheckEnemyTourFormType"} <= commands:
        return "Patrol routing"
    if "CheckEnemyIdleFormType" in commands and any("Random" in name for name in operators):
        return "Alert idle selector"
    if {"IsExistTourPoint", "SetTourPoint"} <= commands:
        return "Tour point resolver"
    meaningful = next((row.command_name for row in node.rows if row.command_name not in ("", "None")), "")
    return _humanize(meaningful) if meaningful else "Behavior"


def _unique(values) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _node_summary(node: BTableNode, target_titles: dict[str, str]) -> tuple[str, ...]:
    role_summaries = {
        "Entry": ("Enter the local behavior flow",),
        "Target routing": (
            "Weapon state · target existence · combat zone",
            "Route to approach, patrol, or alert idle",
        ),
        "Approach target": (
            "5 m range · CHARA/POS perception · 2 s hold",
            "RUN/WALK arrival or reset target",
        ),
        "Patrol routing": ("NPC · distance · tour form", "Follow, resolve a tour point, or idle"),
        "Alert idle selector": ("Facing · weighted random · idle form", "10 / 20 / 70 and 10 / 90 branches"),
        "Tour point resolver": ("Find point · angle · patrol arrival", "180 s timeout · 1 m arrival"),
    }
    if node.semantic_title in role_summaries:
        return role_summaries[node.semantic_title]
    conditions = _unique(
        _condition_text(row)
        for row in node.rows
        if row.operator_name in ("OperatorIf", "OperatorORIf") and row.command_name != "None"
    )
    actions = _unique(
        _action_text(row, target_titles)
        for row in node.rows
        if row.operator_name in ("OperatorNext", "OperatorJump", "OperatorJumpToImportBTable")
    )
    lines = []
    if conditions:
        lines.append("Checks: " + " · ".join(conditions[:3]))
    if actions:
        lines.append("Actions: " + " · ".join(actions[:3]))
    return tuple(lines[:3]) or (f"{len(node.rows)} behavior rows",)


def _edge_labels(node: BTableNode) -> dict[int, str]:
    stack: list[tuple[str, str]] = []
    result = {}
    for row in node.rows:
        operator = row.operator_name
        if operator == "OperatorIf":
            stack.append(("if", _condition_text(row)))
        elif operator == "OperatorORIf" and stack and stack[-1][0] == "if":
            stack[-1] = ("if", stack[-1][1] + " OR " + _condition_text(row))
        elif operator == "OperatorElseIf":
            while stack and stack[-1][0] != "if":
                stack.pop()
            if stack:
                previous = stack[-1][1]
                stack[-1] = (
                    "if",
                    _condition_text(row)
                    if row.command_name != "None"
                    else f"Else ({previous})",
                )
        elif operator == "OperatorIfEnd":
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == "if":
                    del stack[index:]
                    break
        elif operator == "OperatorRandom":
            weight = _argument_number(row.operator_argument, "EditRandom")
            stack.append(("random", f"{_fmt_number(weight)}%" if weight is not None else "Random"))
        elif operator == "OperatorElseRandom":
            weight = _argument_number(row.operator_argument, "EditRandom")
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == "random":
                    stack[index] = ("random", f"{_fmt_number(weight)}%" if weight is not None else "Random")
                    del stack[index + 1 :]
                    break
        elif operator == "OperatorRandomEnd":
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == "random":
                    del stack[index:]
                    break
        if row.jump_target_guid:
            result[row.index] = " · ".join(value for _, value in stack[-2:]) or "Next"
    return result


def apply_btable_semantics(graph: BTableGraph) -> BTableGraph:
    titled_nodes = tuple(replace(node, semantic_title=_node_role(node)) for node in graph.nodes)
    target_titles = {node.guid: node.semantic_title for node in titled_nodes}
    nodes = tuple(
        replace(node, semantic_summary=_node_summary(node, target_titles))
        for node in titled_nodes
    )
    node_by_guid = {node.guid: node for node in nodes}
    edge_labels = {
        (node.guid, row_index): label
        for node in nodes
        for row_index, label in _edge_labels(node).items()
    }
    edges = tuple(
        replace(edge, label=edge_labels.get((edge.source_guid, edge.row_index), ""))
        for edge in graph.edges
        if edge.source_guid in node_by_guid and edge.target_guid in node_by_guid
    )
    return BTableGraph(nodes, edges, graph.diagnostics)


def build_semantic_steps(
    node: BTableNode,
    target_titles: dict[str, str],
) -> tuple[BTableSemanticStep, ...]:
    steps = []
    stack: list[str] = []
    for row in node.rows:
        operator = row.operator_name
        if operator == "OperatorIf":
            steps.append(
                BTableSemanticStep(
                    row.index,
                    len(stack),
                    f"IF {_condition_text(row)}",
                    row.command_argument,
                    "branch",
                )
            )
            stack.append("if")
        elif operator == "OperatorORIf":
            steps.append(
                BTableSemanticStep(
                    row.index,
                    max(0, len(stack) - 1),
                    f"OR {_condition_text(row)}",
                    row.command_argument,
                    "branch",
                )
            )
        elif operator == "OperatorElseIf":
            while stack and stack[-1] != "if":
                stack.pop()
            if stack:
                stack.pop()
            label = f"ELSE IF {_condition_text(row)}" if row.command_name != "None" else "ELSE"
            steps.append(BTableSemanticStep(row.index, len(stack), label, row.command_argument, "branch"))
            stack.append("if")
        elif operator == "OperatorIfEnd":
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] == "if":
                    del stack[index:]
                    break
        elif operator == "OperatorRandom":
            weight = _argument_number(row.operator_argument, "EditRandom")
            label = f"{_fmt_number(weight)}% CHANCE" if weight is not None else "RANDOM"
            steps.append(BTableSemanticStep(row.index, len(stack), label, "Weighted branch", "branch"))
            stack.append("random")
        elif operator == "OperatorElseRandom":
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] == "random":
                    del stack[index:]
                    break
            weight = _argument_number(row.operator_argument, "EditRandom")
            label = f"{_fmt_number(weight)}% CHANCE" if weight is not None else "RANDOM"
            steps.append(BTableSemanticStep(row.index, len(stack), label, "Weighted branch", "branch"))
            stack.append("random")
        elif operator == "OperatorRandomEnd":
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] == "random":
                    del stack[index:]
                    break
        elif operator in ("OperatorReturn", "OperatorExit"):
            steps.append(BTableSemanticStep(row.index, len(stack), _humanize(operator), "", "flow"))
        else:
            label = _action_text(row, target_titles)
            if label:
                detail = row.command_argument or row.operator_argument
                steps.append(BTableSemanticStep(row.index, len(stack), label, detail, "action"))
    return tuple(steps)


def _type_name(rsz: RszFile, instance_id: int) -> str:
    if not (0 <= instance_id < len(rsz.instance_infos)) or rsz.type_registry is None:
        return ""
    info = rsz.type_registry.get_type_info(rsz.instance_infos[instance_id].type_id)
    return str(info.get("name", "")) if info else ""


def _find_root(rsz: RszFile, expected_type: str) -> int | None:
    for instance_id in reversed(tuple(getattr(rsz, "object_table", ()) or ())):
        if _type_name(rsz, instance_id) == expected_type:
            return instance_id
    return None


def is_btable_document(rsz: RszFile | None) -> bool:
    return rsz is not None and _find_root(rsz, BTABLE_TYPE) is not None


def _objects(value) -> tuple[int, ...]:
    if isinstance(value, ObjectData):
        return (int(value.value),) if value.value else ()
    if not isinstance(value, ArrayData):
        return ()
    return tuple(
        int(item.value)
        for item in value.values
        if isinstance(item, ObjectData) and item.value
    )


def _scalar(value):
    if isinstance(value, UserDataData):
        return value.string.rstrip("\0") or int(value.value)
    if hasattr(value, "guid_str"):
        return str(value.guid_str)
    if hasattr(value, "value"):
        return value.value
    if all(hasattr(value, axis) for axis in ("x", "y")):
        components = [getattr(value, axis) for axis in ("x", "y", "z", "w") if hasattr(value, axis)]
        return "(" + ", ".join(f"{item:g}" for item in components) + ")"
    return value


def _short_type(type_name: str) -> str:
    value = type_name.rstrip("\0").rsplit(".", 1)[-1]
    if value.startswith("c") and len(value) > 1 and value[1].isupper():
        value = value[1:]
    return value or "None"


def _argument_summary(
    rsz: RszFile,
    instance_id: int,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> str:
    if not instance_id or depth > 3:
        return ""
    seen = set() if seen is None else seen
    if instance_id in seen:
        return "<cycle>"
    seen.add(instance_id)
    type_name = _type_name(rsz, instance_id)
    short_name = _short_type(type_name)
    if short_name.endswith("ArgumentNone") or short_name == "None":
        return ""
    fields = getattr(rsz, "parsed_elements", {}).get(instance_id, {})
    parts = []
    for field_name, value in fields.items():
        label = field_name.lstrip("_")
        if isinstance(value, ObjectData):
            rendered = _argument_summary(
                rsz,
                int(value.value),
                depth=depth + 1,
                seen=set(seen),
            )
        elif isinstance(value, ArrayData):
            items = list(value.values)
            shown = []
            for item in items[:4]:
                if isinstance(item, ObjectData):
                    shown.append(
                        _argument_summary(
                            rsz,
                            int(item.value),
                            depth=depth + 1,
                            seen=set(seen),
                        )
                    )
                else:
                    shown.append(str(_scalar(item)))
            rendered = "[" + ", ".join(shown) + (", …" if len(items) > 4 else "") + "]"
        elif isinstance(value, StructData):
            rendered = f"{len(value.values)} item(s)"
        else:
            rendered = str(_scalar(value)).rstrip("\0")
        if rendered:
            parts.append(rendered if label == "Value" and len(fields) == 1 else f"{label}={rendered}")
    if not parts:
        return short_name
    return f"{short_name}({'; '.join(parts)})"


def _collect_guids(rsz: RszFile, instance_id: int, seen: set[int] | None = None) -> list[str]:
    seen = set() if seen is None else seen
    if not instance_id or instance_id in seen:
        return []
    seen.add(instance_id)
    result = []
    for value in getattr(rsz, "parsed_elements", {}).get(instance_id, {}).values():
        if hasattr(value, "guid_str"):
            result.append(str(value.guid_str))
        elif isinstance(value, ObjectData):
            result.extend(_collect_guids(rsz, int(value.value), seen))
        elif isinstance(value, ArrayData):
            for item in value.values:
                if hasattr(item, "guid_str"):
                    result.append(str(item.guid_str))
                elif isinstance(item, ObjectData):
                    result.extend(_collect_guids(rsz, int(item.value), seen))
    return result


def _fallback_order_name(rsz: RszFile, argument_id: int, value_hash: int) -> str:
    if not value_hash:
        return "None"
    argument_type = _short_type(_type_name(rsz, argument_id))
    for suffix in ("ArgumentNone", "Argument"):
        if argument_type.endswith(suffix):
            argument_type = argument_type[: -len(suffix)]
    return argument_type or f"0x{value_hash:016X}"


def build_btable_graph(
    rsz: RszFile,
    operator_names: dict[int, str] | None = None,
    command_names: dict[int, str] | None = None,
    diagnostics: tuple[str, ...] = (),
) -> BTableGraph:
    root_id = _find_root(rsz, BTABLE_TYPE)
    if root_id is None:
        raise ValueError("RSZ document does not contain an ace.btable.user_data.BTable root")
    root_fields = rsz.parsed_elements.get(root_id, {})
    table_container = next(iter(_objects(root_fields.get("_Tables"))), 0)
    table_refs = _objects(rsz.parsed_elements.get(table_container, {}).get("_DataArray"))
    if not table_refs:
        return BTableGraph((), (), diagnostics + ("BTable contains no local tables",))

    operator_names = operator_names or {}
    command_names = command_names or {}
    raw_nodes = []
    guid_to_instance = {}
    for index, table_id in enumerate(table_refs):
        if _type_name(rsz, table_id) != TABLE_TYPE:
            continue
        fields = rsz.parsed_elements.get(table_id, {})
        guid = str(_scalar(fields.get("_Guid", "")))
        guid_to_instance[guid] = table_id
        raw_nodes.append((index, table_id, guid, _objects(fields.get("_Rows"))))

    nodes = []
    edges = []
    unresolved = 0
    for index, table_id, guid, row_refs in raw_nodes:
        rows = []
        for row_index, row_id in enumerate(row_refs):
            if _type_name(rsz, row_id) != ROW_TYPE:
                continue
            fields = rsz.parsed_elements.get(row_id, {})
            op_hash = int(_scalar(fields.get("_OperatorHash", 0)) or 0)
            cmd_hash = int(_scalar(fields.get("_CommandHash", 0)) or 0)
            op_ref = next(iter(_objects(fields.get("_OperatorArgument"))), 0)
            cmd_ref = next(iter(_objects(fields.get("_CommandArgument"))), 0)
            op_name = operator_names.get(op_hash) or _fallback_order_name(rsz, op_ref, op_hash)
            cmd_name = command_names.get(cmd_hash) or _fallback_order_name(rsz, cmd_ref, cmd_hash)
            op_summary = _argument_summary(rsz, op_ref)
            cmd_summary = _argument_summary(rsz, cmd_ref)
            import_match = _GUID_PAIR_RE.search(op_summary)
            import_source = import_match.group(1).casefold() if import_match else None
            import_target = import_match.group(2).casefold() if import_match else None
            jump_guid = None
            if _type_name(rsz, op_ref) == JUMP_ARGUMENT_TYPE or op_name == "OperatorJump":
                for candidate in _collect_guids(rsz, op_ref):
                    if candidate in guid_to_instance:
                        jump_guid = candidate
                        edges.append(BTableEdge(guid, candidate, row_index))
                        break
                if jump_guid is None:
                    unresolved += 1
            rows.append(
                BTableRow(
                    row_index,
                    row_id,
                    op_hash,
                    op_name,
                    cmd_hash,
                    cmd_name,
                    op_summary,
                    cmd_summary,
                    jump_guid,
                    import_source,
                    import_target,
                )
            )
        nodes.append(BTableNode(index, table_id, guid, tuple(rows)))
    graph_diagnostics = list(diagnostics)
    if unresolved:
        graph_diagnostics.append(f"{unresolved} jump target(s) could not be resolved")
    return apply_btable_semantics(
        BTableGraph(tuple(nodes), tuple(edges), tuple(graph_diagnostics))
    )


class BTableGraphLoader:
    def __init__(self, handler):
        self.handler = handler
        self._cache: dict[str, RszFile] = {}
        self.diagnostics: list[str] = []

    def load(self) -> BTableGraph:
        operator_names, command_names = self._load_order_names(self.handler.rsz_file)
        graph = build_btable_graph(
            self.handler.rsz_file,
            operator_names,
            command_names,
            tuple(self.diagnostics),
        )
        imported_names = self._load_imported_names(
            self.handler.rsz_file,
            operator_names,
            command_names,
        )
        unresolved = 0
        nodes = []
        for node in graph.nodes:
            rows = []
            for row in node.rows:
                resolved = ""
                if row.import_source_guid and row.import_target_guid:
                    resolved = imported_names.get(
                        (row.import_source_guid, row.import_target_guid),
                        "",
                    )
                    if not resolved:
                        unresolved += 1
                rows.append(replace(row, resolved_import=resolved))
            nodes.append(replace(node, rows=tuple(rows)))
        diagnostics = list(dict.fromkeys((*graph.diagnostics, *self.diagnostics)))
        if unresolved:
            diagnostics.append(f"{unresolved} imported call(s) could not be named")
        return apply_btable_semantics(
            BTableGraph(tuple(nodes), graph.edges, tuple(diagnostics))
        )

    def _load_imported_names(
        self,
        source: RszFile,
        operator_names: dict[int, str],
        command_names: dict[int, str],
    ) -> dict[tuple[str, str], str]:
        root_id = _find_root(source, BTABLE_TYPE)
        root = source.parsed_elements.get(root_id, {}) if root_id is not None else {}
        imports = root.get("_ImportBTableList")
        values = imports.values if isinstance(imports, ArrayData) else ()
        result = {}
        for value in values:
            if not isinstance(value, UserDataData) or not value.string:
                continue
            imported = self._load_resource(value.string.rstrip("\0"))
            if imported is None:
                self.diagnostics.append(
                    f"Imported BTable could not be loaded: {value.string.rstrip(chr(0))}"
                )
                continue
            imported_root_id = _find_root(imported, BTABLE_TYPE)
            if imported_root_id is None:
                continue
            imported_root = imported.parsed_elements.get(imported_root_id, {})
            source_guid = str(_scalar(imported_root.get("_ThisGuid", ""))).casefold()
            if not source_guid:
                continue
            imported_graph = build_btable_graph(
                imported,
                operator_names,
                command_names,
            )
            for node in imported_graph.nodes:
                result[(source_guid, node.guid.casefold())] = self._describe_imported_node(node)
        return result

    @staticmethod
    def _describe_imported_node(node: BTableNode) -> str:
        known_actions = _unique(_known_action(row.command_argument) for row in node.rows)
        if known_actions:
            return " + ".join(known_actions[:3])
        actions = _unique(
            _action_text(row)
            for row in node.rows
            if row.command_name not in (
                "",
                "None",
                "RegisterMoveActionSetting",
                "RegisterComboBreak",
            )
            and row.operator_name not in (
                "OperatorIf",
                "OperatorORIf",
                "OperatorElseIf",
            )
        )
        if actions:
            return " + ".join(actions[:2])
        conditions = _unique(
            _condition_text(row)
            for row in node.rows
            if row.operator_name in ("OperatorIf", "OperatorORIf")
        )
        return " / ".join(conditions[:2]) or node.title

    def _load_order_names(self, source: RszFile) -> tuple[dict[int, str], dict[int, str]]:
        root_id = _find_root(source, BTABLE_TYPE)
        root = source.parsed_elements.get(root_id, {}) if root_id is not None else {}
        order_ref = root.get("_OrderBank")
        order_path = order_ref.string if isinstance(order_ref, UserDataData) else ""
        if not order_path:
            self.diagnostics.append("BTable has no OrderBank reference; hashes remain numeric")
            return {}, {}
        order_bank = self._load_resource(order_path)
        if order_bank is None:
            self.diagnostics.append(f"OrderBank could not be loaded: {order_path}")
            return {}, {}
        bank_root = _find_root(order_bank, ORDER_BANK_TYPE)
        if bank_root is None:
            self.diagnostics.append(f"Referenced resource is not a BTableOrderBank: {order_path}")
            return {}, {}
        operator_names = {}
        command_names = {}
        bank_fields = order_bank.parsed_elements.get(bank_root, {})
        for item_id in _objects(bank_fields.get("_OrderLists")):
            item_fields = order_bank.parsed_elements.get(item_id, {})
            list_ref = item_fields.get("_OrderList")
            list_path = list_ref.string if isinstance(list_ref, UserDataData) else ""
            order_list = self._load_resource(list_path) if list_path else None
            if order_list is None:
                if list_path:
                    self.diagnostics.append(f"OrderList could not be loaded: {list_path}")
                continue
            self._consume_order_list(order_list, operator_names, command_names)
        return operator_names, command_names

    def _consume_order_list(self, rsz: RszFile, operators: dict[int, str], commands: dict[int, str]) -> None:
        root_id = _find_root(rsz, ORDER_LIST_TYPE)
        if root_id is None:
            return
        fields = rsz.parsed_elements.get(root_id, {})
        for factory_field, hash_field, target in (
            ("_OperatorFactories", "_OperatorHashList", operators),
            ("_CommandFactories", "_CommandHashList", commands),
        ):
            factory_ids = _objects(fields.get(factory_field))
            hashes_value = fields.get(hash_field)
            hashes = [int(_scalar(item)) for item in hashes_value.values] if isinstance(hashes_value, ArrayData) else []
            for value_hash, factory_id in zip(hashes, factory_ids):
                order_type = rsz.parsed_elements.get(factory_id, {}).get("_OrderType")
                full_name = str(_scalar(order_type) or "").rstrip("\0")
                if full_name:
                    target.setdefault(value_hash, _short_type(full_name))
                elif value_hash == 0:
                    target.setdefault(0, "None")

    def _load_resource(self, resource_path: str) -> RszFile | None:
        normalized = normalize_resource_path(resource_path)
        key = normalized.casefold()
        if key in self._cache:
            return self._cache[key]
        candidates = [normalized]
        if normalized.casefold().endswith(".user"):
            candidates.insert(0, normalized + ".3")
        hit = None
        for candidate in candidates:
            hit = resolve_handler_resource_data(
                self.handler,
                candidate,
                allow_selection_dialog=False,
            )
            if hit is not None:
                break
        if hit is None:
            hit = self._filesystem_hit(candidates)
        if hit is None:
            return None
        resolved_path, data = hit
        try:
            parsed = RszFile()
            parsed.filepath = resolved_path
            parsed.game_version = self.handler.game_version
            parsed.type_registry = self.handler.type_registry
            parsed.read(data)
        except (OSError, ValueError) as exc:
            self.diagnostics.append(f"Could not parse {normalized}: {exc}")
            return None
        self._cache[key] = parsed
        return parsed

    def _filesystem_hit(self, candidates: list[str]) -> tuple[str, bytes] | None:
        source = str(getattr(self.handler, "filepath", "") or "").replace("\\", "/")
        marker = source.casefold().find("/natives/")
        if marker < 0:
            return None
        root = Path(source[:marker])
        for candidate in candidates:
            normalized = normalize_resource_path(candidate)
            if not normalized.casefold().startswith(("natives/stm/", "natives/x64/")):
                normalized = "natives/stm/" + normalized
            path = root / Path(normalized)
            if path.is_file():
                return str(path), path.read_bytes()
        return None


class _GraphView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#171a21"))

    def wheelEvent(self, event):
        factor = 1.16 if event.angleDelta().y() > 0 else 1 / 1.16
        self.scale(factor, factor)
        event.accept()


class _NodeItem(QGraphicsRectItem):
    WIDTH = 330.0
    MAX_VISIBLE_ROWS = 7

    def __init__(self, node: BTableNode):
        super().__init__()
        self.node = node
        self.mode = "behavior"
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setData(0, node.guid)
        self.setBrush(QBrush(QColor("#252a34")))
        self.setPen(QPen(QColor("#586174"), 1.2))
        # Keep the Python wrapper alive. PySide otherwise destroys this child
        # when the constructor-local reference goes out of scope, leaving an
        # empty node rectangle in the scene.
        self._text = QGraphicsTextItem(self)
        self._text.setTextWidth(self.WIDTH - 20)
        self._text.setPos(10, 7)
        self._text.setHtml(self._html(node))
        self._resize_to_text()

    def _resize_to_text(self) -> None:
        height = max(70.0, self._text.document().size().height() + 14.0)
        self.setRect(QRectF(0, 0, self.WIDTH, height))

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._text.setHtml(self._html(self.node))
        self._resize_to_text()

    @staticmethod
    def _row_color(operator_name: str) -> str:
        name = operator_name.casefold()
        if "jump" in name:
            return "#c792ea"
        if "random" in name:
            return "#ffcb6b"
        if any(word in name for word in ("if", "switch", "case", "and", "or")):
            return "#89ddff"
        if any(word in name for word in ("return", "exit")):
            return "#f07178"
        return "#a6accd"

    def _html(self, node: BTableNode) -> str:
        if self.mode == "behavior":
            return self._semantic_html(node)
        return self._raw_html(node)

    @staticmethod
    def _semantic_html(node: BTableNode) -> str:
        lines = [
            "<div style='font-size:12pt;font-weight:600;color:#ffffff'>"
            f"{escape(node.display_title)}</div>",
            f"<div style='color:#7f8ba3;font-size:8pt'>{len(node.rows)} rows</div>",
            "<hr style='color:#40485a'>",
        ]
        for summary in node.semantic_summary[:3]:
            lines.append(f"<div style='color:#d8dee9;margin-bottom:3px'>{escape(summary)}</div>")
        return "".join(lines)

    def _raw_html(self, node: BTableNode) -> str:
        lines = [
            f"<div style='font-size:12pt;font-weight:600;color:#ffffff'>{escape(node.title)}</div>",
            f"<div style='color:#7f8ba3;font-size:8pt'>{escape(node.guid)}</div>",
            "<hr style='color:#40485a'>",
        ]
        for row in node.rows[: self.MAX_VISIBLE_ROWS]:
            color = self._row_color(row.operator_name)
            command = escape(row.command_name)
            args = escape(row.command_argument)
            suffix = f" <span style='color:#8b93a7'>{args}</span>" if args else ""
            lines.append(
                f"<div><span style='color:#657085'>{row.index:02d}</span> "
                f"<span style='color:{color}'>{escape(row.operator_name)}</span> "
                f"<span style='color:#82aaff'>→</span> <span style='color:#d8dee9'>{command}</span>{suffix}</div>"
            )
        if len(node.rows) > self.MAX_VISIBLE_ROWS:
            lines.append(f"<div style='color:#7f8ba3'>… {len(node.rows) - self.MAX_VISIBLE_ROWS} more row(s)</div>")
        return "".join(lines)

    def paint(self, painter, option, widget=None):
        self.setPen(QPen(QColor("#82aaff") if self.isSelected() else QColor("#586174"), 2.2 if self.isSelected() else 1.2))
        super().paint(painter, option, widget)


class _EdgeItem(QGraphicsPathItem):
    def __init__(self, source: _NodeItem, target: _NodeItem, row_index: int, label: str = ""):
        source_rect = source.sceneBoundingRect()
        target_rect = target.sceneBoundingRect()
        start = QPointF(source_rect.right(), source_rect.center().y())
        end = QPointF(target_rect.left(), target_rect.center().y())
        distance = max(80.0, abs(end.x() - start.x()) * 0.5)
        direction = 1.0 if end.x() >= start.x() else -1.0
        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x() + distance * direction, start.y()),
            QPointF(end.x() - distance * direction, end.y()),
            end,
        )
        super().__init__(path)
        self.setPen(QPen(QColor("#bb80e6"), 2.0))
        self.setZValue(-10)
        self.setToolTip(
            f"{label + ': ' if label else ''}jump from row {row_index} to {target.node.display_title}"
        )
        arrow = QPolygonF(
            [
                end,
                QPointF(end.x() - 10, end.y() - 5),
                QPointF(end.x() - 10, end.y() + 5),
            ]
        )
        marker = QGraphicsPolygonItem(arrow, self)
        marker.setPen(QPen(QColor("#bb80e6")))
        marker.setBrush(QBrush(QColor("#bb80e6")))
        self._label = QGraphicsTextItem(self)
        self._label.setDefaultTextColor(QColor("#a6accd"))
        self._label.setTextWidth(190.0)
        self._label.setPlainText(label)
        midpoint = path.pointAtPercent(0.5)
        self._label.setPos(midpoint + QPointF(5.0, -22.0))

    def set_mode(self, mode: str) -> None:
        self._label.setVisible(mode == "behavior" and bool(self._label.toPlainText()))


def _weak_components(graph: BTableGraph) -> list[list[str]]:
    adjacency = {node.guid: set() for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.source_guid].add(edge.target_guid)
        adjacency[edge.target_guid].add(edge.source_guid)
    result = []
    remaining = set(adjacency)
    order = {node.guid: node.index for node in graph.nodes}
    while remaining:
        start = min(remaining, key=order.get)
        queue = [start]
        remaining.remove(start)
        component = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(sorted(component, key=order.get))
    return sorted(result, key=lambda values: (-len(values), order[values[0]]))


def _layout_nodes(graph: BTableGraph, items: dict[str, _NodeItem]) -> None:
    directed = defaultdict(list)
    for edge in graph.edges:
        directed[edge.source_guid].append(edge.target_guid)
    cursor_x = cursor_y = row_height = 0.0
    pack_width = 5200.0
    for component in _weak_components(graph):
        component_set = set(component)
        indegree = {guid: 0 for guid in component}
        for source in component:
            for target in directed[source]:
                if target in component_set:
                    indegree[target] += 1
        roots = [guid for guid in component if indegree[guid] == 0] or component[:1]
        depth = {guid: 0 for guid in roots}
        queue = deque(roots)
        while queue:
            source = queue.popleft()
            for target in directed[source]:
                if target not in component_set or target in depth:
                    continue
                depth[target] = depth[source] + 1
                queue.append(target)
        for guid in component:
            depth.setdefault(guid, 0)
        layers = defaultdict(list)
        for guid in component:
            layers[depth[guid]].append(guid)
        local_positions = {}
        component_height = 0.0
        for layer, guids in layers.items():
            y = 0.0
            for guid in guids:
                local_positions[guid] = QPointF(layer * 480.0, y)
                y += items[guid].rect().height() + 45.0
            component_height = max(component_height, y)
        component_width = (max(layers) + 1) * 480.0
        if cursor_x and cursor_x + component_width > pack_width:
            cursor_x = 0.0
            cursor_y += row_height + 120.0
            row_height = 0.0
        for guid, pos in local_positions.items():
            items[guid].setPos(cursor_x + pos.x(), cursor_y + pos.y())
        cursor_x += component_width + 120.0
        row_height = max(row_height, component_height)


class BTableGraphWidget(QWidget):
    def __init__(self, graph: BTableGraph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._match_index = -1
        self._matches: list[_NodeItem] = []
        self._mode = "behavior"
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        tools = QHBoxLayout()
        self.summary = QLabel(
            self._summary_text()
        )
        tools.addWidget(self.summary)
        tools.addStretch(1)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        behavior_button = QPushButton("Behavior", self)
        raw_button = QPushButton("Raw", self)
        for button in (behavior_button, raw_button):
            button.setCheckable(True)
            self.mode_group.addButton(button)
        behavior_button.setChecked(True)
        behavior_button.setToolTip("Show semantic behavior summaries and a nested condition tree")
        raw_button.setToolTip("Show original rows, operators, arguments, and GUIDs")
        tools.addWidget(behavior_button)
        tools.addWidget(raw_button)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search behavior, commands, arguments, GUID…")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(280)
        next_button = QPushButton("Next", self)
        fit_button = QPushButton("Fit graph", self)
        tools.addWidget(self.search)
        tools.addWidget(next_button)
        tools.addWidget(fit_button)
        outer.addLayout(tools)

        self.scene = QGraphicsScene(self)
        self.view = _GraphView(self.scene, self)
        self.details = QTreeWidget(self)
        self.details.setHeaderLabels(("Behavior", "Source"))
        self.details.setAlternatingRowColors(True)
        self.details.setMinimumWidth(480)
        self.details.setColumnWidth(0, 300)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.view)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)
        if graph.diagnostics:
            warning = QLabel(" · ".join(graph.diagnostics), self)
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #d7ba7d; padding: 3px;")
            outer.addWidget(warning)

        self._items = {node.guid: _NodeItem(node) for node in graph.nodes}
        for item in self._items.values():
            self.scene.addItem(item)
        self._edge_items: list[_EdgeItem] = []
        self._rebuild_edges()
        self.scene.selectionChanged.connect(self._selection_changed)
        self.search.textChanged.connect(self._apply_search)
        self.search.returnPressed.connect(self._next_match)
        next_button.clicked.connect(self._next_match)
        fit_button.clicked.connect(self.fit_graph)
        behavior_button.clicked.connect(lambda: self._set_mode("behavior"))
        raw_button.clicked.connect(lambda: self._set_mode("raw"))
        if self._items:
            next(iter(self._items.values())).setSelected(True)
        self.fit_graph()

    def _summary_text(self, match_count: int | None = None) -> str:
        imported = sum(
            1
            for node in self.graph.nodes
            for row in node.rows
            if row.import_target_guid
        )
        text = (
            f"{len(self.graph.nodes)} tables · "
            f"{sum(len(node.rows) for node in self.graph.nodes)} rows · "
            f"{len(self.graph.edges)} local jumps · {imported} imported calls"
        )
        if match_count is not None:
            text += f" · {match_count} matches"
        return text

    def _rebuild_edges(self) -> None:
        for item in self._edge_items:
            self.scene.removeItem(item)
        self._edge_items.clear()
        _layout_nodes(self.graph, self._items)
        show_labels = len(self.graph.nodes) <= 30
        for edge in self.graph.edges:
            source = self._items.get(edge.source_guid)
            target = self._items.get(edge.target_guid)
            if source is None or target is None:
                continue
            item = _EdgeItem(source, target, edge.row_index, edge.label)
            item.set_mode(self._mode if show_labels else "raw")
            self.scene.addItem(item)
            self._edge_items.append(item)

    def _set_mode(self, mode: str) -> None:
        if mode not in ("behavior", "raw"):
            return
        self._mode = mode
        for item in self._items.values():
            item.set_mode(mode)
        self._rebuild_edges()
        self._selection_changed()
        self.fit_graph()

    def fit_graph(self):
        bounds = self.scene.itemsBoundingRect()
        if not bounds.isEmpty():
            self.view.fitInView(bounds.adjusted(-30, -30, 30, 30), Qt.AspectRatioMode.KeepAspectRatio)

    def _apply_search(self, text: str):
        query = text.strip().casefold()
        self._matches = []
        self._match_index = -1
        for item in self._items.values():
            matched = not query or query in item.node.search_text
            item.setOpacity(1.0 if matched else 0.16)
            item.setZValue(1 if matched and query else 0)
            if matched and query:
                self._matches.append(item)
        self.summary.setText(self._summary_text(len(self._matches) if query else None))

    def _next_match(self):
        if not self._matches:
            return
        self._match_index = (self._match_index + 1) % len(self._matches)
        item = self._matches[self._match_index]
        self.scene.clearSelection()
        item.setSelected(True)
        self.view.centerOn(item)

    def _selection_changed(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, _NodeItem)]
        if not selected:
            return
        node = selected[0].node
        self.details.clear()
        if self._mode == "behavior":
            self._populate_behavior_details(node)
        else:
            self._populate_raw_details(node)

    def _populate_behavior_details(self, node: BTableNode) -> None:
        self.details.setHeaderLabels((node.display_title, "Source"))
        target_titles = {
            item.guid: item.semantic_title
            for item in self.graph.nodes
        }
        branch_parents: dict[int, QTreeWidgetItem] = {}
        for step in build_semantic_steps(node, target_titles):
            item = QTreeWidgetItem((step.label, f"Row {step.row_index:02d}"))
            parent = branch_parents.get(step.depth - 1) if step.depth else None
            if parent is None:
                self.details.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for depth in tuple(branch_parents):
                if depth >= step.depth:
                    del branch_parents[depth]
            if step.kind == "branch":
                branch_parents[step.depth] = item
        self.details.expandAll()
        self.details.resizeColumnToContents(0)

    def _populate_raw_details(self, node: BTableNode) -> None:
        self.details.setHeaderLabels(("Row", node.title, node.guid))
        for row in node.rows:
            item = QTreeWidgetItem(
                (
                    str(row.index),
                    row.operator_name,
                    row.command_name,
                )
            )
            if row.operator_argument:
                item.addChild(QTreeWidgetItem(("", "Operator args", row.operator_argument)))
            if row.command_argument:
                item.addChild(QTreeWidgetItem(("", "Command args", row.command_argument)))
            if row.jump_target_guid:
                item.addChild(QTreeWidgetItem(("", "Jump target", row.jump_target_guid)))
            if row.import_target_guid:
                item.addChild(
                    QTreeWidgetItem(
                        (
                            "",
                            "Imported call",
                            row.resolved_import
                            or f"{row.import_source_guid}|{row.import_target_guid}",
                        )
                    )
                )
            self.details.addTopLevelItem(item)
        self.details.resizeColumnToContents(0)
        self.details.resizeColumnToContents(1)

    def cleanup(self):
        self.scene.clear()


def create_btable_preview(handler):
    rsz = getattr(handler, "rsz_file", None)
    if not is_btable_document(rsz):
        return None
    try:
        graph = BTableGraphLoader(handler).load()
        return BTableGraphWidget(graph)
    except Exception as exc:
        label = QLabel(f"BTable graph preview could not be created:\n{exc}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        return label
