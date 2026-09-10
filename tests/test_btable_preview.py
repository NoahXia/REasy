from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from file_handlers.rsz.btable_preview import (
    BTableNode,
    BTableRow,
    BTableGraphWidget,
    BTableGraphLoader,
    _NodeItem,
    build_btable_graph,
    build_semantic_steps,
    is_btable_document,
)
from file_handlers.rsz.rsz_data_types import ArrayData, GuidData, ObjectData, U64Data
from file_handlers.rsz.rsz_file import RszFile
from file_handlers.rsz.rsz_handler import RszHandler, _registry_filename_for_game
from utils.type_registry import TypeRegistry


class _Registry:
    def __init__(self, names):
        self.names = names

    def get_type_info(self, type_id):
        name = self.names.get(type_id)
        return {"name": name} if name else None


def _fixture():
    names = {
        1: "ace.btable.cEditFieldTableGuid",
        2: "ace.btable.cOperatorArgumentJump",
        3: "ace.btable.cCommandArgumentNone",
        4: "ace.btable.user_data.BTable.cRowElement",
        5: "ace.btable.user_data.BTable.cTableElement",
        6: "ace.cInstanceGuidArray`1<ace.btable.user_data.BTable.cTableElement>",
        7: "ace.btable.user_data.BTable",
    }
    infos = [SimpleNamespace(type_id=0)] + [SimpleNamespace(type_id=i) for i in (1, 2, 3, 4, 5, 5, 6, 7)]
    source_guid = "11111111-1111-1111-1111-111111111111"
    target_guid = "22222222-2222-2222-2222-222222222222"
    parsed = {
        1: {"_Value": GuidData(target_guid)},
        2: {"_EditTableGuid": ObjectData(1)},
        3: {},
        4: {
            "_OperatorHash": U64Data(10),
            "_OperatorArgument": ObjectData(2),
            "_CommandHash": U64Data(0),
            "_CommandArgument": ObjectData(3),
        },
        5: {"_Rows": ArrayData([]), "_Guid": GuidData(target_guid)},
        6: {"_Rows": ArrayData([ObjectData(4)]), "_Guid": GuidData(source_guid)},
        7: {"_DataArray": ArrayData([ObjectData(6), ObjectData(5)])},
        8: {"_Tables": ObjectData(7)},
    }
    return SimpleNamespace(
        instance_infos=infos,
        parsed_elements=parsed,
        object_table=[8],
        type_registry=_Registry(names),
    )


class TestBTableGraph(unittest.TestCase):
    def test_wots_registry_filename_aliases(self):
        self.assertEqual(_registry_filename_for_game("OnimushaWOTS"), "rszoniwots.json")
        self.assertEqual(_registry_filename_for_game("ONIWOTS"), "rszoniwots.json")

    def test_wots_context_finds_bundled_registry_without_a_setting(self):
        handler = RszHandler()
        handler.app = SimpleNamespace(
            settings={"game_version": "RE4", "rcol_json_path": ""},
            current_game=None,
            proj_dock=None,
        )
        handler.resource_context = SimpleNamespace(game="OnimushaWOTS")
        resolved = Path(handler._resolve_type_registry_path())
        self.assertEqual(resolved.name, "rszoniwots.json")
        self.assertTrue(resolved.is_file())
        self.assertEqual(handler.game_version, "OnimushaWOTS")

    def test_node_retains_its_text_item(self):
        from PySide6.QtWidgets import QApplication, QGraphicsTextItem

        app = QApplication.instance() or QApplication([])
        node = build_btable_graph(
            _fixture(), {10: "OperatorJump"}, {0: "None"}
        ).nodes[0]
        item = _NodeItem(node)
        children = item.childItems()
        self.assertEqual(len(children), 1)
        self.assertIsInstance(children[0], QGraphicsTextItem)
        self.assertIn("Entry", children[0].toHtml())
        item.set_mode("raw")
        self.assertIn("OperatorJump", children[0].toHtml())
        app.processEvents()

    def test_extracts_tables_rows_and_jump_edges(self):
        rsz = _fixture()
        self.assertTrue(is_btable_document(rsz))
        graph = build_btable_graph(rsz, {10: "OperatorJump"}, {0: "None"})
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(sum(len(node.rows) for node in graph.nodes), 1)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].source_guid, graph.nodes[0].guid)
        self.assertEqual(graph.edges[0].target_guid, graph.nodes[1].guid)
        self.assertEqual(graph.nodes[0].rows[0].operator_name, "OperatorJump")
        self.assertEqual(graph.nodes[0].semantic_title, "Entry")
        self.assertEqual(graph.edges[0].label, "Next")

    def test_semantic_tree_collapses_else_and_end_markers(self):
        node = BTableNode(
            0,
            1,
            "11111111-1111-1111-1111-111111111111",
            (
                BTableRow(0, 1, 1, "OperatorIf", 1, "TargetExistCheck"),
                BTableRow(1, 2, 1, "OperatorNext", 1, "TargetReset"),
                BTableRow(2, 3, 1, "OperatorElseIf", 0, "None"),
                BTableRow(3, 4, 1, "OperatorNext", 1, "EnemyExitPatrol"),
                BTableRow(4, 5, 1, "OperatorIfEnd", 0, "None"),
            ),
        )
        labels = [step.label for step in build_semantic_steps(node, {})]
        self.assertEqual(
            labels,
            ["IF Target exists", "Reset target", "ELSE", "Exit patrol"],
        )
        self.assertNotIn("If End", labels)

    def test_known_wots_action_guid_gets_a_readable_name(self):
        node = BTableNode(
            0,
            1,
            "11111111-1111-1111-1111-111111111111",
            (
                BTableRow(
                    0,
                    1,
                    1,
                    "OperatorNext",
                    1,
                    "RequestFullBodyAction",
                    command_argument=(
                        "RequestFullBodyActionArg(ActionIndexBody="
                        "Guid>(05e14c01-4b4d-4399-b213-c74f49c308aa))"
                    ),
                ),
            ),
        )
        steps = build_semantic_steps(node, {})
        self.assertEqual(steps[0].label, "cAlertTargetLose")

    def test_widget_switches_between_behavior_and_raw_modes(self):
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        widget = BTableGraphWidget(
            build_btable_graph(_fixture(), {10: "OperatorJump"}, {0: "None"})
        )
        self.assertEqual(widget._mode, "behavior")
        self.assertEqual(widget.details.headerItem().text(0), "Table 000 · Entry")
        widget._set_mode("raw")
        self.assertEqual(widget.details.headerItem().text(0), "Row")
        widget._set_mode("behavior")
        self.assertEqual(widget.details.headerItem().text(0), "Table 000 · Entry")
        widget.cleanup()
        app.processEvents()


@unittest.skipUnless(os.environ.get("REASY_WOTS_ASSET_ROOT"), "set REASY_WOTS_ASSET_ROOT")
class TestWotsBTableAssets(unittest.TestCase):
    def _load_graph(self, relative_path: str):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = root / relative_path
        registry_path = Path(__file__).resolve().parents[1] / "resources/data/dumps/rszoniwots.json"
        rsz = RszFile()
        rsz.filepath = str(path)
        rsz.game_version = "ONIWOTS"
        rsz.type_registry = TypeRegistry(str(registry_path))
        rsz.read(path.read_bytes(), validate_type_registry=True)
        handler = SimpleNamespace(
            rsz_file=rsz,
            filepath=str(path),
            game_version="ONIWOTS",
            type_registry=rsz.type_registry,
        )
        return BTableGraphLoader(handler).load()

    def test_em100_action_graph_is_complete(self):
        graph = self._load_graph(
            "GameDesign/Action/Enemy/em100/00/btable/em100_00_action.user.3"
        )
        self.assertEqual(len(graph.nodes), 179)
        self.assertEqual(sum(len(node.rows) for node in graph.nodes), 1158)
        self.assertEqual(len(graph.edges), 59)
        self.assertFalse(graph.diagnostics)
        operator_names = {row.operator_name for node in graph.nodes for row in node.rows}
        command_names = {row.command_name for node in graph.nodes for row in node.rows}
        self.assertIn("OperatorJump", operator_names)
        self.assertIn("OperatorIf", operator_names)
        self.assertIn("RequestFullBodyAction", command_names)

    def test_em100_alert_main_has_semantic_nodes_and_import_names(self):
        graph = self._load_graph(
            "GameDesign/Action/Enemy/em100/00/btable/em100_00_alrt_main.user.3"
        )
        self.assertEqual(
            [node.semantic_title for node in graph.nodes],
            [
                "Entry",
                "Target routing",
                "Approach target",
                "Patrol routing",
                "Alert idle selector",
                "Tour point resolver",
            ],
        )
        imported_names = {
            row.resolved_import
            for node in graph.nodes
            for row in node.rows
            if row.resolved_import
        }
        self.assertIn("cChangeToOneHand", imported_names)
        self.assertIn("WALK · 10 s · arrive 0.5 m", imported_names)
        alert_steps = build_semantic_steps(graph.nodes[4], {})
        self.assertIn("10% CHANCE", {step.label for step in alert_steps})

    def test_action_file_recovers_a_relocated_wots_registry(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = root / "GameDesign/Action/Enemy/em100/00/btable/em100_00_action.user.3"
        handler = RszHandler()
        handler.filepath = str(path)
        handler.resource_context = SimpleNamespace(game="OnimushaWOTS")
        handler.app = SimpleNamespace(
            settings={
                "game_version": "RE4",
                "rcol_json_path": str(
                    Path("Z:/old/REasy/dist/resources/data/dumps/rszoniwots.json")
                ),
            },
            _rsz_type_registry_override=None,
        )
        handler.read(path.read_bytes(), validate_type_registry=True)
        self.assertIsNotNone(handler.type_registry)
        self.assertEqual(Path(handler.type_registry.json_path).name, "rszoniwots.json")
        self.assertEqual(handler.game_version, "OnimushaWOTS")
        self.assertTrue(is_btable_document(handler.rsz_file))


if __name__ == "__main__":
    unittest.main(verbosity=2)
