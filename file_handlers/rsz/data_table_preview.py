from __future__ import annotations

"""Read-only spreadsheet-style preview for tabular RSZ user data."""

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from utils.enum_manager import EnumManager
from utils.number_format import format_display_value

from .rsz_data_types import ArrayData, ObjectData, StructData, UserDataData


MAX_OBJECT_FLATTEN_DEPTH = 4


@dataclass(frozen=True, slots=True)
class RszTableDataset:
    path: str
    element_type: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def label(self) -> str:
        type_name = self.element_type.rsplit(".", 1)[-1] if self.element_type else "Values"
        return f"{self.path} · {type_name} · {len(self.rows)} rows"


def _field_label(name: str) -> str:
    return str(name or "Value").lstrip("_") or "Value"


def _instance_type(rsz, instance_id: int) -> str:
    infos = getattr(rsz, "instance_infos", ())
    registry = getattr(rsz, "type_registry", None)
    if not (0 < instance_id < len(infos)) or registry is None:
        return ""
    info = registry.get_type_info(infos[instance_id].type_id)
    return str((info or {}).get("name", "") or "")


def _enum_label(value) -> str:
    original_type = str(getattr(value, "orig_type", "") or "")
    raw = getattr(value, "value", None)
    if not original_type or not isinstance(raw, int) or isinstance(raw, bool):
        return ""
    members = EnumManager.instance().get_enum_values(original_type)
    if not isinstance(members, list):
        return ""
    matches = [str(member.get("name", "")) for member in members if member.get("value") == raw]
    return matches[0] if len(matches) == 1 else ""


def format_table_value(value, rsz=None) -> str:
    """Return a compact, stable cell representation for one parsed RSZ value."""
    if value is None:
        return ""
    if isinstance(value, ObjectData):
        instance_id = int(value.value or 0)
        if not instance_id:
            return "None"
        type_name = _instance_type(rsz, instance_id) if rsz is not None else ""
        return f"{type_name or value.orig_type or 'Object'} (ID: {instance_id})"
    if isinstance(value, UserDataData):
        if value.string:
            return value.string
        return f"{value.orig_type or 'UserData'} (ID: {value.value})" if value.value else "None"
    if isinstance(value, ArrayData):
        values = list(value.values)
        if not values:
            return "[]"
        if len(values) <= 8 and all(_is_scalar(item) for item in values):
            return ", ".join(format_table_value(item, rsz) for item in values)
        return f"[{len(values)} items]"
    if isinstance(value, StructData):
        return f"[{len(value.values)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} fields}}"

    enum_name = _enum_label(value)
    raw = getattr(value, "value", None)
    if enum_name:
        return f"{enum_name} ({raw})"
    if raw is not None:
        if isinstance(raw, bool):
            return "true" if raw else "false"
        return format_display_value(raw, 7)
    if hasattr(value, "guid_str"):
        return str(value.guid_str)
    if hasattr(value, "raw_bytes"):
        return bytes(value.raw_bytes).hex(" ")

    component_names = (
        ("x", "y", "z", "w"),
        ("r", "g", "b", "a"),
        ("min_x", "min_y", "max_x", "max_y"),
        ("width", "height"),
        ("min", "max"),
    )
    for names in component_names:
        present = [name for name in names if hasattr(value, name)]
        if len(present) >= 2:
            return ", ".join(
                f"{name}={format_display_value(getattr(value, name), 7)}"
                for name in present
            )
    return str(value)


def _is_scalar(value) -> bool:
    return not isinstance(value, (ArrayData, StructData, ObjectData, UserDataData, dict))


def _flatten_mapping(
    fields: dict,
    rsz,
    *,
    prefix: str = "",
    depth: int = 0,
    visited: frozenset[int] = frozenset(),
) -> dict[str, str]:
    """Flatten inline RSZ object references into dotted spreadsheet columns."""
    flattened: dict[str, str] = {}
    parsed_elements = getattr(rsz, "parsed_elements", {})
    for field_name, value in fields.items():
        label = _field_label(field_name)
        column = f"{prefix}.{label}" if prefix else label
        if isinstance(value, ObjectData):
            instance_id = int(value.value or 0)
            nested_fields = parsed_elements.get(instance_id)
            if (
                instance_id
                and isinstance(nested_fields, dict)
                and nested_fields
                and depth < MAX_OBJECT_FLATTEN_DEPTH
                and instance_id not in visited
            ):
                flattened.update(
                    _flatten_mapping(
                        nested_fields,
                        rsz,
                        prefix=column,
                        depth=depth + 1,
                        visited=visited | {instance_id},
                    )
                )
                continue
        flattened[column] = format_table_value(value, rsz)
    return flattened


def _mapping_dataset(path: str, entries, rsz) -> RszTableDataset | None:
    mappings: list[dict[str, str]] = []
    instance_ids: list[int | None] = []
    types: list[str] = []
    for entry in entries:
        if isinstance(entry, ObjectData):
            instance_id = int(entry.value or 0)
            fields = getattr(rsz, "parsed_elements", {}).get(instance_id)
            if not isinstance(fields, dict):
                continue
            mappings.append(_flatten_mapping(fields, rsz, visited=frozenset({instance_id})))
            instance_ids.append(instance_id)
            types.append(_instance_type(rsz, instance_id) or entry.orig_type)
        elif isinstance(entry, dict):
            mappings.append(_flatten_mapping(entry, rsz))
            instance_ids.append(None)
        else:
            return None
    if not mappings:
        return None

    field_names: list[str] = []
    for fields in mappings:
        for field_name in fields:
            if field_name not in field_names:
                field_names.append(field_name)
    headers = ["#"]
    if any(instance_id is not None for instance_id in instance_ids):
        headers.append("Instance ID")
    headers.extend(field_names)
    rows = []
    for index, (fields, instance_id) in enumerate(zip(mappings, instance_ids)):
        row = [str(index)]
        if len(headers) > len(field_names) + 1:
            row.append(str(instance_id or ""))
        row.extend(fields.get(name, "") for name in field_names)
        rows.append(tuple(row))
    element_type = next((type_name for type_name in types if type_name), "Struct")
    return RszTableDataset(path, element_type, tuple(headers), tuple(rows))


def _collection_dataset(path: str, value, rsz) -> RszTableDataset | None:
    entries = list(value.values)
    if not entries:
        return None
    mapped = _mapping_dataset(path, entries, rsz)
    if mapped is not None:
        return mapped
    if all(not isinstance(entry, (dict, ArrayData, StructData)) for entry in entries):
        return RszTableDataset(
            path,
            str(getattr(value, "orig_type", "") or "Values"),
            ("#", "Value"),
            tuple((str(index), format_table_value(entry, rsz)) for index, entry in enumerate(entries)),
        )
    return None


def build_data_table_datasets(rsz) -> tuple[RszTableDataset, ...]:
    """Discover table-shaped collections reachable from a USR root object."""
    if not getattr(rsz, "is_usr", False) or not getattr(rsz, "object_table", None):
        return ()
    root_id = rsz.object_table[0]
    root_fields = getattr(rsz, "parsed_elements", {}).get(root_id)
    if not isinstance(root_fields, dict):
        return ()

    datasets: list[RszTableDataset] = []
    visited_objects: set[int] = set()
    visited_collections: set[int] = set()

    def scan_mapping(fields: dict, path: str):
        for field_name, field_value in fields.items():
            child_path = f"{path}.{_field_label(field_name)}" if path else _field_label(field_name)
            scan_value(field_value, child_path)

    def scan_value(value, path: str):
        if isinstance(value, (ArrayData, StructData)):
            identity = id(value)
            if identity in visited_collections:
                return
            visited_collections.add(identity)
            dataset = _collection_dataset(path, value, rsz)
            if dataset is not None:
                datasets.append(dataset)
                # A tabular collection is already represented as one coherent
                # sheet. Keep nested collections as compact cells instead of
                # producing one duplicate sheet per row.
                return
            for index, entry in enumerate(value.values):
                if isinstance(entry, ObjectData):
                    scan_value(entry, f"{path}[{index}]")
                elif isinstance(entry, dict):
                    scan_mapping(entry, f"{path}[{index}]")
                elif isinstance(entry, (ArrayData, StructData)):
                    scan_value(entry, f"{path}[{index}]")
        elif isinstance(value, ObjectData):
            instance_id = int(value.value or 0)
            if not instance_id or instance_id in visited_objects:
                return
            visited_objects.add(instance_id)
            fields = getattr(rsz, "parsed_elements", {}).get(instance_id)
            if isinstance(fields, dict):
                scan_mapping(fields, path)

    scan_mapping(root_fields, "")
    root_summary = _mapping_dataset("Root", [ObjectData(root_id)], rsz)
    if root_summary is not None:
        datasets.append(root_summary)
    return tuple(datasets)


class RszTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset: RszTableDataset | None = None

    def set_dataset(self, dataset: RszTableDataset | None):
        self.beginResetModel()
        self.dataset = dataset
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.dataset is None else len(self.dataset.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.dataset is None else len(self.dataset.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or self.dataset is None:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self.dataset.columns):
            return self.dataset.columns[section]
        if orientation == Qt.Vertical:
            return section + 1
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.dataset is None:
            return None
        if not (0 <= index.row() < len(self.dataset.rows)):
            return None
        row = self.dataset.rows[index.row()]
        if not (0 <= index.column() < len(row)):
            return None
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            return row[index.column()]
        if role == Qt.TextAlignmentRole and index.column() < 2:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


class RszTableFilterProxy(QSortFilterProxyModel):
    @staticmethod
    def _number(text: str) -> Decimal | None:
        try:
            value = Decimal(text.strip())
        except (InvalidOperation, ValueError):
            return None
        return value if value.is_finite() else None

    def lessThan(self, left, right):
        model = self.sourceModel()
        left_text = str(model.data(left, Qt.DisplayRole) or "")
        right_text = str(model.data(right, Qt.DisplayRole) or "")
        left_number = self._number(left_text)
        right_number = self._number(right_text)
        if left_number is not None and right_number is not None:
            return left_number < right_number
        return left_text.casefold() < right_text.casefold()

    def filterAcceptsRow(self, source_row, source_parent):
        pattern = self.filterRegularExpression()
        if not pattern.pattern():
            return True
        model = self.sourceModel()
        for column in range(model.columnCount()):
            index = model.index(source_row, column, source_parent)
            if pattern.match(str(model.data(index, Qt.DisplayRole) or "")).hasMatch():
                return True
        return False


class DataTablePreviewWidget(QWidget):
    def __init__(self, datasets: tuple[RszTableDataset, ...], parent=None):
        super().__init__(parent)
        self.datasets = datasets
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel(self.tr("Table:")))
        self.dataset_combo = QComboBox(self)
        for dataset in datasets:
            self.dataset_combo.addItem(dataset.label)
        controls.addWidget(self.dataset_combo, 1)
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText(self.tr("Filter rows…"))
        self.search_edit.setClearButtonEnabled(True)
        controls.addWidget(self.search_edit)
        self.copy_button = QPushButton(self.tr("Copy"), self)
        self.export_button = QPushButton(self.tr("Export CSV…"), self)
        controls.addWidget(self.copy_button)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)

        self.model = RszTableModel(self)
        self.proxy = RszTableFilterProxy(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.table = QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.status_label = QLabel(self)
        layout.addWidget(self.status_label)

        self.dataset_combo.currentIndexChanged.connect(self._select_dataset)
        self.search_edit.textChanged.connect(self.proxy.setFilterFixedString)
        self.copy_button.clicked.connect(self.copy_selection)
        self.export_button.clicked.connect(self.export_csv)
        self._select_dataset(0)

    def _select_dataset(self, index: int):
        dataset = self.datasets[index] if 0 <= index < len(self.datasets) else None
        self.model.set_dataset(dataset)
        self.table.sortByColumn(0, Qt.AscendingOrder)
        self.table.resizeColumnsToContents()
        if dataset is not None:
            self.status_label.setText(
                self.tr("{rows} rows · {columns} columns · read-only").format(
                    rows=len(dataset.rows), columns=len(dataset.columns)
                )
            )

    def _selected_grid(self):
        indexes = self.table.selectionModel().selectedIndexes()
        if not indexes:
            return (), ()
        source_indexes = [self.proxy.mapToSource(index) for index in indexes]
        rows = sorted({index.row() for index in source_indexes})
        columns = sorted({index.column() for index in source_indexes})
        dataset = self.model.dataset
        if dataset is None:
            return (), ()
        return (
            tuple(dataset.columns[column] for column in columns),
            tuple(tuple(dataset.rows[row][column] for column in columns) for row in rows),
        )

    def copy_selection(self):
        headers, rows = self._selected_grid()
        if not rows:
            dataset = self.model.dataset
            if dataset is None:
                return
            headers, rows = dataset.columns, dataset.rows
        lines = ["\t".join(headers), *("\t".join(row) for row in rows)]
        QGuiApplication.clipboard().setText("\n".join(lines))

    def export_csv(self):
        dataset = self.model.dataset
        if dataset is None:
            return
        default_name = f"{Path(dataset.path).name or 'rsz_data'}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Data Table"), default_name, self.tr("CSV files (*.csv)")
        )
        if not filename:
            return
        with open(filename, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(dataset.columns)
            writer.writerows(dataset.rows)


def create_data_table_preview(handler) -> DataTablePreviewWidget | None:
    datasets = build_data_table_datasets(getattr(handler, "rsz_file", None))
    return DataTablePreviewWidget(datasets) if datasets else None
