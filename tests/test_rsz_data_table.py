from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from file_handlers.rsz.data_table_preview import (
    DataTablePreviewWidget,
    RszTableDataset,
    build_data_table_datasets,
    format_table_value,
)
from file_handlers.rsz.rsz_data_types import (
    ArrayData,
    BoolData,
    F32Data,
    Float3Data,
    ObjectData,
    S32Data,
    StringData,
    StructData,
)
from file_handlers.rsz.rsz_file import RszFile
from file_handlers.rsz.rsz_handler import RszHandler
from file_handlers.rsz.wots_registry import WotsTypeRegistry
from utils.type_registry import TypeRegistry


class _Registry:
    def __init__(self, names):
        self.names = names

    def get_type_info(self, type_id):
        name = self.names.get(type_id)
        return {"name": name} if name else None


def _fixture():
    return SimpleNamespace(
        is_usr=True,
        object_table=[1],
        instance_infos=[
            SimpleNamespace(type_id=0),
            SimpleNamespace(type_id=10),
            SimpleNamespace(type_id=11),
            SimpleNamespace(type_id=11),
            SimpleNamespace(type_id=12),
            SimpleNamespace(type_id=12),
        ],
        type_registry=_Registry(
            {
                10: "app.user_data.AttackReachPatternData",
                11: "app.user_data.AttackReachPatternData.cReachPatternInfo",
                12: "app.cKnockbackParam.cKnockbackData",
            }
        ),
        parsed_elements={
            1: {
                "_ReachPatternList": ArrayData(
                    [ObjectData(2), ObjectData(3)],
                    ObjectData,
                    "app.user_data.AttackReachPatternData.cReachPatternInfo",
                ),
                "_Empty": ArrayData([]),
                "_Weights": ArrayData([S32Data(2), S32Data(4)]),
            },
            2: {
                "_Name": StringData("Near"),
                "_Enabled": BoolData(True),
                "_Offset": Float3Data(1.0, 2.0, 3.0),
                "_Nested": ArrayData([S32Data(1), S32Data(2)]),
                "_KnockbackData": ObjectData(4),
            },
            3: {
                "_Name": StringData("Far"),
                "_Distance": S32Data(12),
                "_Nested": ArrayData([ObjectData(2)]),
                "_KnockbackData": ObjectData(5),
            },
            4: {"_UseCustom": BoolData(True), "ZStartSpeed": S32Data(8)},
            5: {"_UseCustom": BoolData(False), "ZStartSpeed": S32Data(3)},
        },
    )


class TestRszDataTable(unittest.TestCase):
    def test_object_array_becomes_rows_with_union_of_fields(self):
        datasets = build_data_table_datasets(_fixture())
        table = next(dataset for dataset in datasets if dataset.path == "ReachPatternList")
        self.assertEqual(
            table.columns,
            (
                "#",
                "Instance ID",
                "Name",
                "Enabled",
                "Offset",
                "Nested",
                "KnockbackData.UseCustom",
                "KnockbackData.ZStartSpeed",
                "Distance",
            ),
        )
        self.assertEqual(table.rows[0][2], "Near")
        self.assertEqual(table.rows[0][3], "true")
        self.assertEqual(table.rows[0][6:8], ("true", "8"))
        self.assertEqual(table.rows[1][8], "12")
        self.assertEqual(table.rows[1][3], "")
        root = next(dataset for dataset in datasets if dataset.path == "Root")
        self.assertEqual(len(root.rows), 1)
        self.assertIn("ReachPatternList", root.columns)
        self.assertFalse(
            any(dataset.path.startswith("ReachPatternList[") for dataset in datasets)
        )

    def test_scalar_and_struct_arrays_are_discovered(self):
        rsz = _fixture()
        rsz.parsed_elements[1]["_StructRows"] = StructData(
            [{"_Key": StringData("A"), "_Value": S32Data(7)}]
        )
        datasets = build_data_table_datasets(rsz)
        weights = next(dataset for dataset in datasets if dataset.path == "Weights")
        structs = next(dataset for dataset in datasets if dataset.path == "StructRows")
        self.assertEqual(weights.rows, (("0", "2"), ("1", "4")))
        self.assertEqual(structs.columns, ("#", "Key", "Value"))
        self.assertEqual(structs.rows[0], ("0", "A", "7"))
        self.assertFalse(any(dataset.path == "Empty" for dataset in datasets))

    def test_complex_values_use_compact_summaries(self):
        rsz = _fixture()
        self.assertEqual(format_table_value(ArrayData([S32Data(1), S32Data(2)]), rsz), "1, 2")
        self.assertEqual(format_table_value(ArrayData([ObjectData(2)]), rsz), "[1 items]")
        self.assertEqual(format_table_value(ObjectData(2), rsz), "app.user_data.AttackReachPatternData.cReachPatternInfo (ID: 2)")
        self.assertEqual(format_table_value(F32Data(13.800000190734863), rsz), "13.8")

    def test_non_usr_document_has_no_data_table(self):
        rsz = _fixture()
        rsz.is_usr = False
        self.assertEqual(build_data_table_datasets(rsz), ())

    def test_widget_switches_datasets_and_filters_rows(self):
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        datasets = build_data_table_datasets(_fixture())
        widget = DataTablePreviewWidget(datasets)
        self.assertEqual(widget.model.rowCount(), len(datasets[0].rows))
        widget.search_edit.setText("Far")
        self.assertEqual(widget.proxy.rowCount(), 1)
        widget.cleanup = lambda: None
        widget.close()
        app.processEvents()

    def test_table_defaults_to_numeric_ascending_row_order(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        dataset = RszTableDataset(
            "Values",
            "Value",
            ("#", "Value"),
            (("10", "ten"), ("2", "two"), ("100", "hundred"), ("1", "one")),
        )
        widget = DataTablePreviewWidget((dataset,))
        displayed = [
            widget.proxy.data(widget.proxy.index(row, 0), Qt.DisplayRole)
            for row in range(widget.proxy.rowCount())
        ]
        self.assertEqual(displayed, ["1", "2", "10", "100"])
        widget.close()
        app.processEvents()


@unittest.skipUnless(os.environ.get("REASY_WOTS_ASSET_ROOT"), "set REASY_WOTS_ASSET_ROOT")
class TestWotsRszDataTableAssets(unittest.TestCase):
    def test_attack_reach_pattern_is_tabular(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = root / (
            "GameDesign/Action/Player/Data/ParamPack/"
            "PlayerAttackReachPatternData.user.3"
        )
        registry_path = (
            Path(__file__).resolve().parents[1]
            / "resources/data/dumps/rszoniwots.json"
        )
        rsz = RszFile()
        rsz.filepath = str(path)
        rsz.game_version = "OnimushaWOTS"
        rsz.type_registry = TypeRegistry(str(registry_path))
        rsz.read(path.read_bytes(), validate_type_registry=True)

        tables = build_data_table_datasets(rsz)
        reach_patterns = next(
            table for table in tables if table.path == "ReachPatternList"
        )
        self.assertGreater(len(reach_patterns.rows), 0)
        self.assertIn("Instance ID", reach_patterns.columns)
        self.assertTrue(any(column != "" for column in reach_patterns.columns[2:]))

    def test_knockback_data_object_is_flattened_into_columns(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = root / "GameDesign/Action/knockbackpresetparamlist.user.3"
        registry_path = (
            Path(__file__).resolve().parents[1]
            / "resources/data/dumps/rszoniwots.json"
        )
        rsz = RszFile()
        rsz.filepath = str(path)
        rsz.game_version = "OnimushaWOTS"
        rsz.type_registry = TypeRegistry(str(registry_path))
        rsz.read(path.read_bytes(), validate_type_registry=True)

        preset = next(
            table for table in build_data_table_datasets(rsz)
            if table.path == "PresetList"
        )
        self.assertEqual(len(preset.rows), 2)
        self.assertIn("KnockbackData.UseCustom", preset.columns)
        self.assertIn("KnockbackData.ZStartSpeed", preset.columns)
        self.assertNotIn("KnockbackData", preset.columns)

    def test_player_attack_param_uses_wots_registry_overlay(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = root / (
            "GameDesign/Action/Player/Data/ParamPack/"
            "PlayerAttackParamList.user.3"
        )
        handler = RszHandler()
        handler.filepath = str(path)
        handler.resource_context = SimpleNamespace(game="OnimushaWOTS")
        handler.app = SimpleNamespace(
            settings={"game_version": "OnimushaWOTS", "rcol_json_path": ""},
            current_game="OnimushaWOTS",
            proj_dock=None,
        )
        handler.read(path.read_bytes(), validate_type_registry=True)

        self.assertIsInstance(handler.type_registry, WotsTypeRegistry)
        info, _ = handler.type_registry.find_type_by_name("app.cAttackParamDataPlayer")
        self.assertEqual(info["crc"], "b25b515f")
        self.assertIn("_AddSkill5Gauge", {field["name"] for field in info["fields"]})


if __name__ == "__main__":
    unittest.main()
