from __future__ import annotations

"""Read-only assembled character preview for WOTS PFB resources."""

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from ui.scene.scn_visibility_panel import ScnGameObjectVisibilityPanel

from .scn_scene_preview import ScnScenePreviewWidget


class WotsPfbPreviewWidget(QWidget):
    """Display the resolved PFB Montage assembly without a motion backend."""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self._initialized = False
        self._cleaned = False

        settings = getattr(getattr(handler, "app", None), "settings", None)
        self.scene = ScnScenePreviewWidget(
            self,
            settings=settings if isinstance(settings, dict) else None,
        )
        self.visibility = ScnGameObjectVisibilityPanel(self)
        self.visibility.visibility_changed.connect(self._apply_visibility)
        self.visibility.focus_keys_changed.connect(
            self.scene.set_focused_renderables
        )
        self.scene.renderables_changed.connect(self._sync_scene_objects)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.scene)
        splitter.addWidget(self.visibility)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([920, 240])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initialized and not self._cleaned:
            self._initialized = True
            QTimer.singleShot(0, self.scene.ensure_loaded)

    def _sync_scene_objects(self) -> None:
        if self._cleaned:
            return
        self.visibility.set_graphs(self.scene.graphs)
        self._apply_visibility()

    def _apply_visibility(self) -> None:
        if self._cleaned:
            return
        self.scene.set_user_renderable_visibility_overrides(
            self.visibility.user_overrides
        )

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.scene.cleanup()
        self.handler = None
