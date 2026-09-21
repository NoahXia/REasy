from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QObject, QPoint, Signal
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QWidget

from file_handlers.mesh.material_session import MeshMaterialCollection
from file_handlers.motion.preview.event_timeline import MotionEventTimeline
from file_handlers.motion.preview.target import RigPreviewTarget
from file_handlers.motion.preview.widget import MotListPreviewWidget


class _FailingMaterialSession(QObject):
    reset = Signal()
    images_updated = Signal(object)
    status_changed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.get("parent"))
        self.loading = False
        self.closed = False

    def start(self) -> None:
        raise ValueError("bad MDF")

    def close(self) -> None:
        self.closed = True


class TestMotionPreviewUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_timeline_tooltip_accepts_qhelp_event(self):
        timeline = MotionEventTimeline()
        event = QHelpEvent(
            QEvent.Type.ToolTip,
            QPoint(12, 12),
            QPoint(12, 12),
        )

        timeline.event(event)

    def test_material_collection_rolls_back_failed_session(self):
        collection = MeshMaterialCollection(None)
        session = _FailingMaterialSession()

        with self.assertRaisesRegex(ValueError, "bad MDF"):
            collection.add("body", session)

        self.assertEqual(collection.sessions, ())
        self.assertTrue(session.closed)

    def test_target_keeps_geometry_when_one_material_fails(self):
        widget = MotListPreviewWidget.__new__(MotListPreviewWidget)
        QWidget.__init__(widget)
        widget.playback = SimpleNamespace(stop=lambda: None)
        widget._materials = MeshMaterialCollection(None, parent=widget)
        widget._target_material_session = None
        widget._target_material_sessions = {}
        widget._target_material_diagnostics = ()
        widget._weapon_attack_collision_diagnostics = ()
        widget._weapon_attack_collision_sources = {}
        widget._scene_renderer = SimpleNamespace(
            set_weapon_attack_collision_sources=lambda _sources: None,
        )
        widget._using_source_rig = True
        widget._motions = [object()]
        widget.target_rig_button = SimpleNamespace(setEnabled=lambda _enabled: None)
        widget._load_current_motion = lambda **_kwargs: None
        widget._material_parse_in_subprocess = False

        part = SimpleNamespace(
            label="Body",
            material_scope="target:0",
            handler=object(),
            explicit_mdf_path="body.mdf2.51",
        )
        target = RigPreviewTarget("Enemy", object(), parts=(part,))

        with patch(
            "file_handlers.motion.preview.widget.MeshMaterialSession",
            _FailingMaterialSession,
        ):
            diagnostics = widget.set_target(target)

        self.assertIs(widget._target, target)
        self.assertEqual(widget._target_material_sessions, {})
        self.assertIn("Material unavailable for Body: bad MDF", diagnostics)


if __name__ == "__main__":
    unittest.main()
