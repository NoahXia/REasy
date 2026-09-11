from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.scene.scene_preview import ScenePreviewWidget
from settings import save_settings
from utils.resource_file_utils import resolve_handler_resource_data
from utils.app_paths import resource_path
from utils.registry_manager import RegistryManager
from file_handlers.mesh.material_session import (
    MeshMaterialCollection,
    MeshMaterialSession,
)

from ..evaluation import (
    rig_from_motion_skeleton,
)
from ..mot.model import Motion
from .catalog import MotionPreviewCatalog
from .blend_shapes import mesh_blend_shape_targets
from .animation_browser import MotionEntryList
from .attack_collision import (
    attack_collision_detail_sections,
    load_attack_collision_resource,
    load_attack_collision_source,
    selected_attack_collision,
)
from .attack_parameters import (
    attack_parameter_detail_sections,
    load_attack_parameter_source,
)
from .controller import MotionPreviewController
from .controls import MotionPlaybackControls
from .editor_layout import MotionEditorPane, MotionEditorWorkspace
from .event_timeline import (
    EventDetailSection,
    MotionEventDetailsWidget,
    MotionEventTimeline,
    TimelineEventSelection,
)
from .model import (
    MotionPreviewError,
    snapshot_diagnostic_messages,
    snapshot_status_messages,
)
from .resolution import (
    MotionListDocument,
    PreviewMotionEntry,
    PreviewMotionOrigin,
)
from .renderer import MotionPreviewRenderer
from .support_registry import entity_motion_support_for_format
from .target import (
    RigPreviewTarget,
    WOTS_MESH_PREVIEW_PRESETS,
    load_re_engine_mesh_preset_target,
    load_re_engine_mesh_target,
)


ViewportFactory = Callable[..., QWidget]


def _wots_type_registry(handler):
    """Resolve the WOTS registry in both source and frozen distributions."""
    app = getattr(handler, "app", None)
    override = getattr(app, "_rsz_type_registry_override", None)
    if (
        override is not None
        and Path(str(getattr(override, "json_path", ""))).name.casefold()
        == "rszoniwots.json"
    ):
        return override

    settings = getattr(app, "settings", {}) or {}
    configured = Path(str(settings.get("rcol_json_path", "") or ""))
    candidates = []
    if configured.name.casefold() == "rszoniwots.json":
        candidates.append(configured)
    candidates.extend((
        resource_path("resources/data/dumps/rszoniwots.json"),
        resource_path("rszoniwots.json"),
    ))
    for candidate in candidates:
        if candidate.is_file():
            registry = RegistryManager.instance().get_registry(str(candidate.resolve()))
            if registry is not None:
                return registry
    return None


class MotListPreviewWidget(QWidget):
    """Interactive MOT preview backed by a game-specific evaluation profile."""

    modified_changed = Signal(bool)
    BONE_NAMES_SETTING = "motion_preview_show_bone_names"
    ATTACK_HITBOXES_SETTING = "motion_preview_show_attack_hitboxes"

    def __init__(
        self,
        handler,
        *,
        viewport_factory: ViewportFactory = ScenePreviewWidget,
    ):
        super().__init__()
        self.handler = handler
        format_codec = handler.motlist_file.codec
        support = entity_motion_support_for_format(format_codec)
        if support is None:
            raise ValueError(
                f"no preview support is registered for {format_codec.profile.name}"
            )
        self.evaluation_profile = support.evaluation
        self.controller = MotionPreviewController(self.evaluation_profile)
        self.playback = MotionPlaybackControls(self.controller, parent=self)
        self.playback.render_requested.connect(self._render)
        self._motions: list[PreviewMotionEntry] = []
        self._target: RigPreviewTarget | None = None
        self._target_material_session: MeshMaterialSession | None = None
        self._target_material_sessions: dict[str, MeshMaterialSession] = {}
        self._using_source_rig = True
        self._cleaned = False
        self._attack_collision_diagnostics: tuple[str, ...] = ()
        self._weapon_attack_collision_diagnostics: tuple[str, ...] = ()
        self._attack_collision_source = None
        self._weapon_attack_collision_sources = {}
        self._attack_parameter_source = None
        self._attack_parameter_diagnostics: tuple[str, ...] = ()
        self._attack_parameter_loaded = False
        root_path = str(getattr(handler, "filepath", "") or handler.model.name)
        self._root_path = root_path
        self._catalog = MotionPreviewCatalog(
            MotionListDocument(root_path, handler.model),
            support.tree_references,
            format_codec,
            app=getattr(handler, "app", None),
            selection_parent=self,
            resource_context=getattr(handler, "resource_context", None),
        )
        self._build_ui(viewport_factory)
        self.play_pause_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Space),
            self,
        )
        self.play_pause_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.play_pause_shortcut.activated.connect(self.playback.toggle)
        self.playback.play_button.setToolTip(self.tr("Play / pause (Space)"))
        self.playback.set_frame_driver(
            self.viewport.set_frame_callback,
        )
        self.viewport.render_failure.connect(
            self._on_render_failure,
            Qt.ConnectionType.QueuedConnection,
        )
        self._materials = MeshMaterialCollection(self.viewport, parent=self)
        self.viewport.texture_quality_changed.connect(
            self._materials.set_texture_quality
        )
        self._scene_renderer = MotionPreviewRenderer(self.viewport)
        type_registry = _wots_type_registry(handler)
        attack_source, self._attack_collision_diagnostics = load_attack_collision_source(
            root_path,
            self._catalog.resources.resource_data,
            type_registry=type_registry,
        )
        self._attack_collision_source = attack_source
        self._scene_renderer.set_attack_collision_source(attack_source)
        self._scene_renderer.set_attack_hitboxes_enabled(
            self.attack_hitboxes_toggle.isChecked()
        )
        self._populate_motions()

    def _build_ui(self, viewport_factory: ViewportFactory) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        settings = getattr(getattr(self.handler, "app", None), "settings", None)
        self.workspace = MotionEditorWorkspace(
            self.tr("MOTLIST Editor  ·  {name}").format(
                name=self.handler.model.name or self.tr("Untitled")
            ),
            settings=settings if isinstance(settings, dict) else None,
            parent=self,
        )
        root.addWidget(self.workspace)

        self.animation_pane = MotionEditorPane(self.tr("Animations"), self)
        self.animation_pane.setMinimumWidth(300)
        self.animation_pane.setMaximumWidth(470)
        self.motion_browser = MotionEntryList(self.animation_pane)
        self.motion_browser.selection_changed.connect(self._on_motion_changed)
        self.motion_browser.entry_activated.connect(self._play_selected_animation)
        self.animation_pane.add_widget(self.motion_browser, 1)
        self.animation_pane.add_widget(self.playback)

        self.viewport_pane = MotionEditorPane(self.tr("Viewport"), self)
        self.viewport_pane.setProperty("role", "viewport")
        self.status_label = QLabel()
        self.status_label.setObjectName("motionStatusBar")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.rig_pane = MotionEditorPane(self.tr("Preview rig"), self)
        self.rig_pane.setMinimumWidth(250)
        self.rig_pane.setMaximumWidth(390)
        source_title = QLabel(self.tr("MOTION SOURCE"))
        source_title.setObjectName("motionInspectorLabel")
        self.rig_pane.add_widget(source_title)
        self.motion_source_label = QLabel()
        self.motion_source_label.setWordWrap(True)
        self.motion_source_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.rig_pane.add_widget(self.motion_source_label)
        rig_title = QLabel(self.tr("ACTIVE RIG"))
        rig_title.setObjectName("motionInspectorLabel")
        self.rig_pane.add_widget(rig_title)
        self.rig_label = QLabel(self.tr("MOT source skeleton"))
        self.rig_label.setWordWrap(True)
        self.rig_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.rig_pane.add_widget(self.rig_label)
        preset_title = QLabel(self.tr("MODEL PRESET"))
        preset_title.setObjectName("motionInspectorLabel")
        self.rig_pane.add_widget(preset_title)
        self.model_preset_combo = QComboBox(self.rig_pane)
        for preset in WOTS_MESH_PREVIEW_PRESETS:
            self.model_preset_combo.addItem(self.tr(preset.label), preset.key)
        self.model_preset_combo.setToolTip(
            self.tr("Load all character mesh parts in the selected preview preset")
        )
        self.rig_pane.add_widget(self.model_preset_combo)
        self.load_preset_button = QPushButton(self.tr("Load Model Preset"))
        self.load_preset_button.clicked.connect(self._load_model_preset)
        self.rig_pane.add_widget(self.load_preset_button)
        self.load_resource_button = QPushButton(self.tr("Load Mesh Resource…"))
        self.load_resource_button.clicked.connect(self._load_mesh_resource)
        self.rig_pane.add_widget(self.load_resource_button)
        self.source_rig_button = QPushButton(self.tr("Use MOT Skeleton"))
        self.source_rig_button.clicked.connect(self.use_source_rig)
        self.rig_pane.add_widget(self.source_rig_button)
        self.target_rig_button = QPushButton(self.tr("Use Loaded Mesh"))
        self.target_rig_button.setEnabled(False)
        self.target_rig_button.clicked.connect(self.use_target_rig)
        self.rig_pane.add_widget(self.target_rig_button)
        self.export_gltf_button = QPushButton(self.tr("Export selected animation…"))
        self.export_gltf_button.clicked.connect(self._export_gltf)
        self.rig_pane.add_widget(self.export_gltf_button)
        self.bone_names_toggle = QCheckBox(self.tr("Show bone names"))
        self.bone_names_toggle.setChecked(
            bool(settings.get(self.BONE_NAMES_SETTING, False))
            if isinstance(settings, dict)
            else False
        )
        self.bone_names_toggle.setToolTip(
            self.tr("Display each animated joint name in the viewport")
        )
        self.bone_names_toggle.toggled.connect(self._on_bone_names_toggled)
        self.rig_pane.add_widget(self.bone_names_toggle)
        self.attack_hitboxes_toggle = QCheckBox(self.tr("Show attack hitboxes"))
        self.attack_hitboxes_toggle.setChecked(
            bool(settings.get(self.ATTACK_HITBOXES_SETTING, True))
            if isinstance(settings, dict)
            else True
        )
        self.attack_hitboxes_toggle.setToolTip(
            self.tr(
                "Draw active AttackCollision request sets from the actor and "
                "loaded preset weapons in the animated pose."
            )
        )
        self.attack_hitboxes_toggle.toggled.connect(
            self._on_attack_hitboxes_toggled
        )
        self.rig_pane.add_widget(self.attack_hitboxes_toggle)
        self.blend_shapes_toggle = QCheckBox(self.tr("Blend shapes"))
        self.blend_shapes_toggle.setChecked(True)
        self.blend_shapes_toggle.setEnabled(False)
        self.blend_shapes_toggle.hide()
        self.blend_shapes_toggle.setToolTip(
            self.tr("Enable MOT-driven mesh blend shapes")
        )
        self.blend_shapes_toggle.toggled.connect(self._on_blend_shapes_toggled)
        self.rig_pane.add_widget(self.blend_shapes_toggle)
        self.event_details = MotionEventDetailsWidget(self.rig_pane)
        self.event_details.setMinimumHeight(150)
        self.rig_pane.add_widget(self.event_details, 1)

        self.viewport = viewport_factory(
            self.viewport_pane,
            controls="motion",
            settings=settings if isinstance(settings, dict) else None,
            initial_distance=3.0,
        )
        self.viewport.setMinimumHeight(320)
        self.viewport.set_bone_name_labels_visible(
            self.bone_names_toggle.isChecked()
        )
        self.viewport_pane.add_widget(self.viewport, 1)
        self.event_timeline = MotionEventTimeline(self.viewport_pane)
        self.event_timeline.frame_requested.connect(self.playback.seek)
        self.event_timeline.scrub_started.connect(self.playback.stop)
        self.event_timeline.details_requested.connect(self._show_event_details)
        self.event_timeline_scroll = QScrollArea(self.viewport_pane)
        self.event_timeline_scroll.setWidgetResizable(True)
        self.event_timeline_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.event_timeline_scroll.setMinimumHeight(150)
        self.event_timeline_scroll.setMaximumHeight(280)
        self.event_timeline_scroll.setWidget(self.event_timeline)
        self.viewport_pane.add_widget(self.event_timeline_scroll)
        self.viewport_pane.add_widget(self.status_label)

        self.workspace.add_pane(self.animation_pane, 0)
        self.workspace.add_pane(self.viewport_pane, 1)
        self.workspace.add_pane(self.rig_pane, 0)
        self.workspace.splitter.setSizes([330, 900, 290])

    def _populate_motions(self) -> None:
        previous = self.current_entry
        previous_key = self._entry_key(previous) if previous is not None else None
        resolution = self._catalog.refresh()
        self._motions = list(resolution.entries)
        selected = next(
            (
                index
                for index, entry in enumerate(self._motions)
                if self._entry_key(entry) == previous_key
            ),
            0,
        )
        self.motion_browser.set_entries(self._motions, selected)
        enabled = bool(self._motions)
        self.source_rig_button.setEnabled(enabled)
        self.load_resource_button.setEnabled(enabled)
        self.load_preset_button.setEnabled(enabled)
        origins = Counter(entry.origin for entry in self._motions)
        summary = self.tr(
            "{total} playable · {embedded} embedded · {inherited} inherited"
        ).format(
            total=len(self._motions),
            embedded=origins[PreviewMotionOrigin.EMBEDDED],
            inherited=origins[PreviewMotionOrigin.INHERITED],
        )
        self.motion_source_label.setText(summary)
        if not enabled:
            if resolution.unresolved_bank_ids:
                banks = ", ".join(
                    str(value) for value in resolution.unresolved_bank_ids
                )
                message = self.tr(
                    "This MOTLIST has no embedded or explicitly inherited MOT payloads. "
                    "Its MotTree references BankID value(s) {banks}; resolve those "
                    "through the owning MOTBANK/PFB preview."
                ).format(banks=banks)
            else:
                message = self.tr(
                    "This MOTLIST contains no resolvable MOT payloads to preview."
                )
            if self._catalog.messages:
                message = f"{message}  {'  '.join(self._catalog.messages)}"
            self._clear_scene(message)
        else:
            self._load_current_motion(reset_camera=True)

    @staticmethod
    def _entry_key(entry: PreviewMotionEntry) -> tuple[str, int, int | None]:
        path = entry.source_path.replace("\\", "/").lower()
        marker = path.find("natives/")
        if marker >= 0:
            path = path[marker:]
        return path, entry.motion_id, entry.bank_id

    @property
    def current_motion(self) -> Motion | None:
        entry = self.current_entry
        return entry.resolve_motion() if entry is not None else None

    @property
    def current_entry(self) -> PreviewMotionEntry | None:
        index = self.motion_browser.current_index
        if index < 0 or index >= len(self._motions):
            return None
        return self._motions[index]

    def _on_motion_changed(self, _index: int) -> None:
        self.playback.stop()
        self._load_current_motion(reset_camera=True)

    def _play_selected_animation(self) -> None:
        if self.controller.ready and not self.controller.playing:
            self.playback.toggle()

    def _on_blend_shapes_toggled(self, enabled: bool) -> None:
        self.controller.set_deformation_enabled(enabled)
        self._render()

    def _on_bone_names_toggled(self, enabled: bool) -> None:
        self.viewport.set_bone_name_labels_visible(enabled)
        app = getattr(self.handler, "app", None)
        settings = getattr(app, "settings", None)
        if isinstance(settings, dict):
            settings[self.BONE_NAMES_SETTING] = bool(enabled)
            save_settings(settings)

    def _on_attack_hitboxes_toggled(self, enabled: bool) -> None:
        self._scene_renderer.set_attack_hitboxes_enabled(enabled)
        app = getattr(self.handler, "app", None)
        settings = getattr(app, "settings", None)
        if isinstance(settings, dict):
            settings[self.ATTACK_HITBOXES_SETTING] = bool(enabled)
            save_settings(settings)
        self._render()

    def _show_event_details(
        self,
        selection: TimelineEventSelection | None,
    ) -> None:
        if selection is None:
            self.event_details.set_event(None)
            return
        event = selected_attack_collision(
            selection.track_type,
            selection.properties,
            selection.sample_frame,
            selection.start_frame,
            selection.end_frame,
        )
        if event is None:
            self.event_details.set_event(selection)
            return

        if event.track_type == "AttackCollision_Wp":
            collision_source = self._weapon_attack_collision_sources.get(
                int(event.collision_type or 0)
            )
            collision_diagnostics = self._weapon_attack_collision_diagnostics or (
                self.tr(
                    "Weapon attack RCOL is not loaded; load a model preset with "
                    "the matching weapon collision resource."
                ),
            )
        else:
            collision_source = self._attack_collision_source
            collision_diagnostics = self._attack_collision_diagnostics
        section_data = list(attack_collision_detail_sections(
            collision_source,
            event,
            collision_diagnostics,
        ))

        if not self._attack_parameter_loaded:
            self._attack_parameter_loaded = True
            (
                self._attack_parameter_source,
                self._attack_parameter_diagnostics,
            ) = load_attack_parameter_source(
                self._root_path,
                self._catalog.resources.resource_data,
                type_registry=_wots_type_registry(self.handler),
            )
        section_data.extend(attack_parameter_detail_sections(
            self._attack_parameter_source,
            event.attack_param_id,
            self._attack_parameter_diagnostics,
        ))
        self.event_details.set_event(
            selection,
            tuple(EventDetailSection(title, rows) for title, rows in section_data),
        )

    def _load_current_motion(self, *, reset_camera: bool) -> None:
        deformation_targets = self._mesh_deformation_targets()
        self.blend_shapes_toggle.setVisible(bool(deformation_targets))
        self.blend_shapes_toggle.setEnabled(
            not self._using_source_rig and bool(deformation_targets)
        )
        motion = self.current_motion
        if motion is None:
            self._clear_scene(self.tr("No motion is selected."))
            return
        self.event_timeline.set_motion(motion)
        self.controller.clear()
        try:
            if self._using_source_rig or self._target is None:
                rig = rig_from_motion_skeleton(
                    motion,
                    scale=self.evaluation_profile.source_preview_scale,
                )
                scale = ", ".join(
                    f"{value:g}"
                    for value in self.evaluation_profile.source_preview_scale
                )
                rig_description = self.tr(
                    "MOT source skeleton (profile scale {scale})"
                ).format(scale=scale)
            else:
                rig = self._target.rig
                rig_description = self._target.label
            self.rig_label.setText(rig_description)
            if not self.controller.load(
                motion,
                rig,
                deformation_targets=(
                    deformation_targets if not self._using_source_rig else ()
                ),
            ):
                self._clear_scene(self.controller.error_message, clear_timeline=False)
                return
            self.playback.configure()
            self._render(reset_camera=reset_camera)
        except (MotionPreviewError, ValueError) as exc:
            self._clear_scene(str(exc), clear_timeline=False)

    def set_target(
        self,
        target: RigPreviewTarget,
        *,
        weapon_attack_sources=None,
        weapon_attack_diagnostics: tuple[str, ...] = (),
    ) -> None:
        self._materials.clear()
        self._target_material_session = None
        self._target_material_sessions.clear()
        self._target = target
        self._weapon_attack_collision_diagnostics = tuple(
            weapon_attack_diagnostics
        )
        self._scene_renderer.set_weapon_attack_collision_sources(
            weapon_attack_sources
        )
        self._weapon_attack_collision_sources = dict(weapon_attack_sources or {})
        self._using_source_rig = False
        for index, part in enumerate(target.render_parts):
            key = part.material_scope or "target"
            session = MeshMaterialSession(
                part.handler,
                material_scope=part.material_scope,
                texture_quality=self._materials.texture_quality,
                parent=self._materials,
            )
            self._target_material_sessions[key] = session
            self._materials.add(key, session)
            if index == 0:
                self._target_material_session = session
        self.target_rig_button.setEnabled(bool(self._motions))
        self.playback.stop()
        self._load_current_motion(reset_camera=True)

    def use_source_rig(self) -> None:
        self._using_source_rig = True
        for key in self._target_material_sessions:
            self._materials.set_enabled(key, False)
        self.playback.stop()
        self._load_current_motion(reset_camera=True)

    def use_target_rig(self) -> None:
        if self._target is None:
            return
        self._using_source_rig = False
        for key in self._target_material_sessions:
            self._materials.set_enabled(key, True)
        self.playback.stop()
        self._load_current_motion(reset_camera=True)

    def load_target_mesh(self, filepath: str, data: bytes) -> None:
        target = load_re_engine_mesh_target(
            filepath,
            data,
            app=getattr(self.handler, "app", None),
            resource_context=getattr(self.handler, "resource_context", None),
        )
        self.set_target(target)

    def _load_mesh_resource(self) -> None:
        resource_path, accepted = QInputDialog.getText(
            self,
            self.tr("Load Target Mesh from Project or PAK"),
            self.tr("Mesh resource path (including numeric version suffix):"),
        )
        if not accepted or not resource_path.strip():
            return
        hit = resolve_handler_resource_data(
            self.handler,
            resource_path.strip(),
            self,
        )
        if hit is None:
            self._show_error(
                self.tr(
                    "The target mesh was not found in the project, PAKs, or unpacked files."
                )
            )
            return
        try:
            filepath, data = hit
            self.load_target_mesh(filepath, data)
        except ValueError as exc:
            self._show_error(
                self.tr("Could not load target mesh: {error}").format(error=exc)
            )

    def _load_model_preset(self) -> None:
        preset_key = str(self.model_preset_combo.currentData() or "")
        preset = next(
            (item for item in WOTS_MESH_PREVIEW_PRESETS if item.key == preset_key),
            None,
        )
        if preset is None:
            self._show_error(self.tr("No model preset is selected."))
            return
        resources = []
        missing = []
        for resource_path in preset.resource_paths:
            hit = resolve_handler_resource_data(self.handler, resource_path, self)
            if hit is None:
                missing.append(resource_path)
            else:
                resources.append(hit)
        if missing:
            self._show_error(
                self.tr("Model preset is missing resource(s): {paths}").format(
                    paths=", ".join(missing)
                )
            )
            return
        try:
            target = load_re_engine_mesh_preset_target(
                preset,
                tuple(resources),
                app=getattr(self.handler, "app", None),
                resource_context=getattr(self.handler, "resource_context", None),
            )
            weapon_sources = {}
            weapon_diagnostics = []
            type_registry = _wots_type_registry(self.handler)
            for collision_type, resource_path in preset.weapon_attack_resources:
                source, diagnostics = load_attack_collision_resource(
                    resource_path,
                    self._catalog.resources.resource_data,
                    type_registry=type_registry,
                )
                weapon_diagnostics.extend(diagnostics)
                if source is not None:
                    weapon_sources[int(collision_type)] = source
            self.set_target(
                target,
                weapon_attack_sources=weapon_sources,
                weapon_attack_diagnostics=tuple(weapon_diagnostics),
            )
        except ValueError as exc:
            self._show_error(
                self.tr("Could not load model preset: {error}").format(error=exc)
            )

    def _export_gltf(self) -> None:
        motion = self.current_motion
        rig = self.controller.rig
        if motion is None or rig is None or not self.controller.ready:
            self._show_error(self.tr("Load a playable animation before exporting."))
            return
        default = f"{motion.name or 'animation'}.glb"
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export selected animation"),
            default,
            self.tr("glTF Binary (*.glb);;glTF JSON (*.gltf)"),
        )
        if not path:
            return
        try:
            from file_handlers.gltf_export import export_gltf

            mesh = None
            if not self._using_source_rig and self._target is not None:
                mesh = self._target.mesh
            result = export_gltf(
                path,
                mesh=mesh,
                rig=rig,
                motion=motion,
                evaluation_profile=self.evaluation_profile,
                material_handler=(
                    self._target.handler
                    if mesh is not None and self._target is not None
                    else None
                ),
                resolved_mdf=(
                    self._target_material_session.resolved_mdf
                    if mesh is not None and self._target_material_session is not None
                    else None
                ),
            )
            QMessageBox.information(
                self,
                self.tr("Export selected animation"),
                self.tr("Exported to:\n{path}").format(path=result),
            )
        except Exception as exc:
            QMessageBox.critical(self, self.tr("glTF Export Failed"), str(exc))

    def _render(self, reset_camera: bool = False) -> None:
        if not self.controller.ready:
            return
        try:
            snapshot = self.controller.sample()
        except MotionPreviewError as exc:
            self._clear_scene(str(exc), clear_timeline=False)
            return
        target = None if self._using_source_rig else self._target
        try:
            self._scene_renderer.present(
                snapshot,
                target,
                reset_camera=reset_camera,
                motion=self.current_motion,
            )
        except (ValueError, RuntimeError) as exc:
            self._clear_scene(str(exc), clear_timeline=False)
            return
        status = self._status_text(snapshot)
        self.event_timeline.set_current_frame(snapshot.frame)
        if status != self.status_label.text():
            self.status_label.setText(status)

    def _status_text(self, snapshot) -> str:
        messages = snapshot_status_messages(
            snapshot,
            self.controller.frames_per_second,
            self.tr,
        )
        if any(abs(weight - 1.0) > 1e-4 for weight in snapshot.node_weights):
            messages.append(self.tr("Orange joints have non-unit MOT weights."))
        motion = self.current_motion
        entry = self.current_entry
        if entry is not None and entry.origin is PreviewMotionOrigin.INHERITED:
            messages.append(
                self.tr("Motion payload inherited from {path}.").format(
                    path=entry.source_path
                )
            )
        if motion is not None and motion.character_path:
            messages.append(
                self.tr(
                    "Character/JMAP expressions are not evaluated in this skeleton preview."
                )
            )
        messages.extend(snapshot_diagnostic_messages(snapshot))
        messages.extend(self._catalog.messages)
        collision_status = self._scene_renderer.attack_collision_status(
            motion,
            snapshot.frame,
        )
        if collision_status:
            messages.append(collision_status)
        if collision_status and self._attack_collision_diagnostics:
            messages.extend(self._attack_collision_diagnostics)
        if collision_status and self._weapon_attack_collision_diagnostics:
            messages.extend(self._weapon_attack_collision_diagnostics)
        return "  ".join(messages)

    def _mesh_deformation_targets(self):
        if self._target is None:
            return ()
        targets = []
        seen = set()
        for part in self._target.render_parts:
            for target in mesh_blend_shape_targets(
                part.mesh,
                self.evaluation_profile.property_name_hash,
                motion_name_key=(
                    self.evaluation_profile.joint_binding.motion_name_key
                ),
            ):
                key = (target.binding_key, target.name, target.property_hash)
                if key not in seen:
                    seen.add(key)
                    targets.append(target)
        return tuple(targets)

    def _show_error(self, message: str) -> None:
        self.status_label.setText(message)

    def _clear_scene(self, message: str, *, clear_timeline: bool = True) -> None:
        self.playback.clear()
        if clear_timeline:
            self.event_timeline.clear()
        self._scene_renderer.clear(reset_camera=True)
        self.status_label.setText(message)

    def _on_render_failure(self, message: str) -> None:
        if not self._cleaned:
            self._clear_scene(
                self.tr("Material preview failed: {error}").format(error=message),
                clear_timeline=False,
            )

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.playback.cleanup()
        self._materials.clear()
        self.viewport.cleanup()

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)
