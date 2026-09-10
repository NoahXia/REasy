from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from html import escape
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QBrush, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
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


@dataclass(frozen=True, slots=True)
class BTableNode:
    index: int
    instance_id: int
    guid: str
    rows: tuple[BTableRow, ...]

    @property
    def title(self) -> str:
        return f"Table {self.index:03d}  {self.guid[:8]}"

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
                )
            )
        return " ".join(values).casefold()


@dataclass(frozen=True, slots=True)
class BTableEdge:
    source_guid: str
    target_guid: str
    row_index: int


@dataclass(frozen=True, slots=True)
class BTableGraph:
    nodes: tuple[BTableNode, ...]
    edges: tuple[BTableEdge, ...]
    diagnostics: tuple[str, ...] = ()


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
                    _argument_summary(rsz, op_ref),
                    _argument_summary(rsz, cmd_ref),
                    jump_guid,
                )
            )
        nodes.append(BTableNode(index, table_id, guid, tuple(rows)))
    graph_diagnostics = list(diagnostics)
    if unresolved:
        graph_diagnostics.append(f"{unresolved} jump target(s) could not be resolved")
    return BTableGraph(tuple(nodes), tuple(edges), tuple(graph_diagnostics))


class BTableGraphLoader:
    def __init__(self, handler):
        self.handler = handler
        self._cache: dict[str, RszFile] = {}
        self.diagnostics: list[str] = []

    def load(self) -> BTableGraph:
        operator_names, command_names = self._load_order_names(self.handler.rsz_file)
        return build_btable_graph(
            self.handler.rsz_file,
            operator_names,
            command_names,
            tuple(self.diagnostics),
        )

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
    WIDTH = 410.0
    MAX_VISIBLE_ROWS = 12

    def __init__(self, node: BTableNode):
        super().__init__()
        self.node = node
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
        height = max(70.0, self._text.document().size().height() + 14.0)
        self.setRect(QRectF(0, 0, self.WIDTH, height))

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
    def __init__(self, source: _NodeItem, target: _NodeItem, row_index: int):
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
        self.setToolTip(f"Jump from row {row_index} to {target.node.guid}")
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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        tools = QHBoxLayout()
        self.summary = QLabel(
            f"{len(graph.nodes)} tables · {sum(len(node.rows) for node in graph.nodes)} rows · {len(graph.edges)} jumps"
        )
        tools.addWidget(self.summary)
        tools.addStretch(1)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search commands, arguments, GUID…")
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
        self.details.setHeaderLabels(("Row", "Operator", "Command / arguments"))
        self.details.setAlternatingRowColors(True)
        self.details.setMinimumWidth(430)
        self.details.setColumnWidth(0, 52)
        self.details.setColumnWidth(1, 150)
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
        _layout_nodes(graph, self._items)
        for edge in graph.edges:
            source = self._items.get(edge.source_guid)
            target = self._items.get(edge.target_guid)
            if source is not None and target is not None:
                self.scene.addItem(_EdgeItem(source, target, edge.row_index))
        self.scene.selectionChanged.connect(self._selection_changed)
        self.search.textChanged.connect(self._apply_search)
        self.search.returnPressed.connect(self._next_match)
        next_button.clicked.connect(self._next_match)
        fit_button.clicked.connect(self.fit_graph)
        if self._items:
            next(iter(self._items.values())).setSelected(True)
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
        self.summary.setText(
            f"{len(self.graph.nodes)} tables · {sum(len(node.rows) for node in self.graph.nodes)} rows · "
            f"{len(self.graph.edges)} jumps" + (f" · {len(self._matches)} matches" if query else "")
        )

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
