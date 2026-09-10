from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from file_handlers.rsz.btable_preview import (
    BTableGraphLoader,
    _NodeItem,
    build_btable_graph,
    is_btable_document,
)
from file_handlers.rsz.rsz_data_types import ArrayData, GuidData, ObjectData, U64Data
from file_handlers.rsz.rsz_file import RszFile
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


@unittest.skipUnless(os.environ.get("REASY_WOTS_ASSET_ROOT"), "set REASY_WOTS_ASSET_ROOT")
class TestWotsBTableAssets(unittest.TestCase):
    def test_em100_action_graph_is_complete(self):
        root = Path(os.environ["REASY_WOTS_ASSET_ROOT"])
        path = root / "GameDesign/Action/Enemy/em100/00/btable/em100_00_action.user.3"
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
        graph = BTableGraphLoader(handler).load()
        self.assertEqual(len(graph.nodes), 179)
        self.assertEqual(sum(len(node.rows) for node in graph.nodes), 1158)
        self.assertEqual(len(graph.edges), 59)
        self.assertFalse(graph.diagnostics)
        operator_names = {row.operator_name for node in graph.nodes for row in node.rows}
        command_names = {row.command_name for node in graph.nodes for row in node.rows}
        self.assertIn("OperatorJump", operator_names)
        self.assertIn("OperatorIf", operator_names)
        self.assertIn("RequestFullBodyAction", command_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
