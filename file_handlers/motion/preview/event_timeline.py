from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from ..mot.model import Motion
from ..mot_clip.model import ClipNode, ClipProperty


_CATEGORY_COLORS = {
    "GAME": QColor("#d99145"),
    "SOUND": QColor("#55a7df"),
    "VFX": QColor("#ba76dc"),
    "MOTION_SYNC": QColor("#65b98a"),
    "PHYSICS": QColor("#d4676c"),
}


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    start: float
    end: float
    state: str


@dataclass(frozen=True, slots=True)
class TimelineLane:
    category: str
    name: str
    total_frame: float
    intervals: tuple[tuple[float, float], ...]
    markers: tuple[float, ...]
    details: str
    kind: str = "generic"
    segments: tuple[TimelineSegment, ...] = ()


_CANCEL_NAMES = {
    "StandIdleGroupCancel": "Stand Idle",
    "DodgeGroupCancel": "Dodge",
    "AttackGroupCancel": "Attack",
    "DefenceGroupCancel": "Defence",
    "IssenGroupCancel": "Issen",
    "ParryCancel": "Parry",
    "JustDodgeCancel": "Just Dodge",
    "GrabCounterCancel": "Grab Counter",
    "IssenCounterCancel": "Issen Counter",
    "IssenBreakCancel": "Issen Break",
    "GrappleBlowCancel": "Grapple Blow",
    "AbsorbCancel": "Absorb",
    "BowCancel": "Bow",
    "DashAttack": "Dash Attack",
    "BattleIdleCancel": "Battle Idle",
}

_PHASE_TEXT = {
    "PRE": "input buffer; the command is retained until ACTUAL begins",
    "ACTUAL": "the animation can cancel into this action immediately",
    "NONE": "cancel is disabled",
    "IGNORE": "the cancel condition is ignored",
}

_PHASE_COLORS = {
    "PRE": QColor("#d6a646"),
    "ACTUAL": QColor("#57b879"),
    "NONE": QColor("#69717d"),
    "IGNORE": QColor("#9272d5"),
}


def motion_timeline_lanes(motion: Motion | None) -> tuple[TimelineLane, ...]:
    if motion is None:
        return ()
    lanes: list[TimelineLane] = []
    for sequence in motion.sequences:
        category = sequence.category.name
        total = max(float(sequence.clip.total_frame), 0.0)
        for node in sequence.clip.root.children:
            track_name = _short_track_name(node)
            if track_name == "PlayerCommandCancel":
                lanes.extend(_cancel_timeline_lanes(category, node, total))
                continue
            properties = tuple(_walk_properties(node.properties))
            intervals = {
                (max(0.0, prop.start_frame), max(0.0, prop.end_frame))
                for prop in properties
                if _is_visible_interval(prop, total)
            }
            markers = {
                max(0.0, key.frame)
                for prop in properties
                for key in prop.keys
                if 0.0 <= key.frame <= total
            }
            if not intervals and markers:
                intervals = {(frame, frame) for frame in markers}
            details = _raw_track_details(category, node)
            kind = "generic"
            display_name = track_name
            if track_name == "AttackCollision_Wp":
                kind = "attack_collision_weapon"
                display_name = "Attack Collision · Weapon"
                on_box = _find_property(node.properties, "_OnClipBox")
                if on_box is not None:
                    intervals = {(on_box.start_frame, on_box.end_frame)}
                    markers = {on_box.start_frame, on_box.end_frame}
                details = _attack_collision_details(category, node)
            lanes.append(
                TimelineLane(
                    category,
                    display_name,
                    total,
                    tuple(sorted(intervals)),
                    tuple(sorted(markers)),
                    details,
                    kind,
                )
            )
    return tuple(lanes)


def _cancel_timeline_lanes(
    category: str,
    node: ClipNode,
    total: float,
) -> list[TimelineLane]:
    lanes: list[TimelineLane] = []
    for group in node.properties:
        phase = _find_property(group.children, "_Phase")
        if phase is None or not phase.keys:
            continue
        segments = _phase_segments(phase, total)
        if not segments:
            continue
        key = group.name.removeprefix("_")
        command_name = _CANCEL_NAMES.get(key, _humanize_identifier(key))
        active = tuple(
            (segment.start, segment.end)
            for segment in segments
            if segment.state != "NONE"
        )
        lines = [
            "DESIGN SUMMARY",
            f"Cancel target: {command_name}",
            "Phases: PRE = input buffer, ACTUAL = immediate cancel, NONE = disabled",
            "",
            "PHASES",
        ]
        lines.extend(
            f"Frames {segment.start:g}–{segment.end:g}  {segment.state} · "
            f"{_PHASE_TEXT.get(segment.state, 'unknown phase')}"
            for segment in segments
        )
        extra = [prop for prop in group.children if prop is not phase]
        if extra:
            lines.extend(("", "PARAMETERS"))
            lines.extend(_semantic_property_line(prop) for prop in extra)
        lines.extend(("", _raw_track_details(category, node, (group,))))
        lanes.append(
            TimelineLane(
                category,
                f"Cancel · {command_name}",
                total,
                active,
                tuple(sorted({key.frame for key in phase.keys})),
                "\n".join(lines),
                "command_cancel",
                segments,
            )
        )
    return lanes


def _phase_segments(prop: ClipProperty, total: float) -> tuple[TimelineSegment, ...]:
    keys = sorted(prop.keys, key=lambda key: key.frame)
    # WOTS stores a repeated key at SequenceData.total_frame as a closing
    # sentinel. It does not represent another authored state change.
    if (
        len(keys) > 1
        and keys[-1].frame >= total
        and keys[-1].value == keys[-2].value
    ):
        keys = keys[:-1]
    result: list[TimelineSegment] = []
    for index, key in enumerate(keys):
        start = max(0.0, float(key.frame))
        end = float(keys[index + 1].frame) if index + 1 < len(keys) else total
        end = max(start, min(end, total))
        if end > start:
            result.append(TimelineSegment(start, end, str(key.value)))
    return tuple(result)


def _attack_collision_details(category: str, node: ClipNode) -> str:
    on_box = _find_property(node.properties, "_OnClipBox")
    request_set = _property_value(_find_property(node.properties, "_RequestSetID"))
    attack_param = _property_value(_find_property(node.properties, "_AttackParamID"))
    hit_group = _property_value(_find_property(node.properties, "_HitGroup"))
    collision_type = _property_value(_find_property(node.properties, "_CollisionType"))
    lines = [
        "DESIGN SUMMARY",
        "Type: Attack Collision (Weapon)",
    ]
    if on_box is not None:
        lines.append(
            f"Active frames: {on_box.start_frame:g}–{on_box.end_frame:g}"
        )
    if attack_param is not None:
        lines.append(f"Attack parameter: AttackParam ID {attack_param}")
    if hit_group is not None:
        lines.append(f"Hit group: {hit_group}")
    if collision_type is not None:
        lines.append(f"Collision type: {collision_type} (raw enum value)")
    if request_set is not None:
        lines.append(f"Request set: {request_set}")
    lines.extend(
        (
            "Note: _OnClipBox is the authoritative active range. Its two On "
            "keys mark the range boundaries; they do not enable it twice.",
            "",
            _raw_track_details(category, node),
        )
    )
    return "\n".join(lines)


def _find_property(
    properties: list[ClipProperty] | tuple[ClipProperty, ...],
    name: str,
) -> ClipProperty | None:
    for prop in properties:
        if prop.name == name:
            return prop
        found = _find_property(prop.children, name)
        if found is not None:
            return found
    return None


def _property_value(prop: ClipProperty | None):
    if prop is None or not prop.keys:
        return None
    return prop.keys[0].value


def _humanize_identifier(name: str) -> str:
    result: list[str] = []
    for char in name:
        if result and char.isupper() and result[-1][-1].islower():
            result.append(char)
        elif result:
            result[-1] += char
        else:
            result.append(char)
    return " ".join(result)


def _semantic_property_line(prop: ClipProperty) -> str:
    name = prop.name.removeprefix("_")
    value = _property_value(prop)
    return f"{_humanize_identifier(name)}: {_format_value(value)}"


def _raw_track_details(
    category: str,
    node: ClipNode,
    roots: tuple[ClipProperty, ...] | None = None,
) -> str:
    lines = ["RAW TRACK DATA", f"{category} · {node.name}"]
    for prop in roots if roots is not None else tuple(node.properties):
        lines.extend(_raw_property_lines(prop))
    return "\n".join(lines)


def _raw_property_lines(prop: ClipProperty, depth: int = 0) -> list[str]:
    prefix = "  " * depth
    values = _compact_key_values(prop)
    suffix = f" = {values}" if values else ""
    lines = [
        f"{prefix}{prop.name}  [{prop.start_frame:g}–{prop.end_frame:g}]{suffix}"
    ]
    for child in prop.children:
        lines.extend(_raw_property_lines(child, depth + 1))
    return lines


def _compact_key_values(prop: ClipProperty) -> str:
    if not prop.keys:
        return ""
    keys = list(prop.keys)
    values = [_format_value(key.value) for key in keys]
    if len(values) > 1 and len(set(values)) == 1:
        frames = ", ".join(f"{key.frame:g}" for key in keys)
        return f"{values[0]} (constant; boundary keys {frames})"
    if (
        len(keys) > 1
        and keys[-1].frame >= prop.end_frame
        and keys[-1].value == keys[-2].value
    ):
        keys = keys[:-1]
    return ", ".join(
        f"{key.frame:g}: {_format_value(key.value)}" for key in keys
    )


def timeline_lane_details(
    lane: TimelineLane,
    frame: float,
    selected_interval: tuple[float, float] | None,
) -> str:
    heading = [f"SELECTED FRAME: {frame:g}"]
    if lane.kind == "command_cancel":
        segment = next(
            (
                item
                for item in lane.segments
                if item.start <= frame < item.end
                or (frame == lane.total_frame and item.end == lane.total_frame)
            ),
            None,
        )
        if segment is None:
            heading.append("CURRENT STATE: Not configured at this frame")
        else:
            heading.append(
                f"CURRENT STATE: {segment.state} · "
                f"{_PHASE_TEXT.get(segment.state, 'unknown phase')}"
            )
    elif lane.kind == "attack_collision_weapon":
        active = any(start <= frame <= end for start, end in lane.intervals)
        heading.append(
            "CURRENT STATE: ACTIVE · attack collision enabled"
            if active
            else "CURRENT STATE: INACTIVE · attack collision disabled"
        )
    else:
        heading.append(
            f"EVENT WINDOW: {selected_interval[0]:g}–{selected_interval[1]:g}"
            if selected_interval is not None
            else "EVENT WINDOW: No event at this frame"
        )
    return "\n".join((*heading, "", lane.details))


def _timeline_interval_is_active(
    frame: float,
    start: float,
    end: float,
    total_frame: float,
) -> bool:
    if abs(end - start) < 1e-6:
        return abs(frame - start) < 0.5
    return start <= frame < end or (frame == total_frame and end == total_frame)


def _walk_properties(properties):
    for prop in properties:
        yield prop
        yield from _walk_properties(prop.children)


def _is_visible_interval(prop: ClipProperty, total: float) -> bool:
    if prop.start_frame < 0.0 or prop.end_frame < prop.start_frame:
        return False
    # Full-span scalar values are configuration, not event windows. Nested
    # containers and non-full ranges carry the authored activation interval.
    return bool(prop.children) or prop.start_frame > 0.0 or prop.end_frame < total


def _short_track_name(node: ClipNode) -> str:
    name = node.name.rsplit(".", 1)[-1]
    return name.removesuffix("Track") or name


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "On" if value else "Off"
    return str(value)


class MotionEventTimeline(QWidget):
    """Compact, read-only overview of WOTS SequenceData event tracks."""

    frame_requested = Signal(float)
    scrub_started = Signal()
    details_requested = Signal(str)
    LABEL_WIDTH = 230
    HEADER_HEIGHT = 24
    ROW_HEIGHT = 24

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._lanes: tuple[TimelineLane, ...] = ()
        self._end_frame = 0.0
        self._current_frame = 0.0
        self._selected_row = -1
        self._selected_interval: tuple[float, float] | None = None
        self._dragging = False
        self.setMouseTracking(True)
        self.setMinimumHeight(self.HEADER_HEIGHT + self.ROW_HEIGHT)

    def set_motion(self, motion: Motion | None) -> None:
        self._lanes = motion_timeline_lanes(motion)
        self._end_frame = max(
            [float(motion.end_frame) if motion is not None else 0.0]
            + [lane.total_frame for lane in self._lanes]
        )
        self._current_frame = 0.0
        self._selected_row = -1
        self._selected_interval = None
        self._dragging = False
        self.details_requested.emit("")
        self.setMinimumHeight(
            self.HEADER_HEIGHT + max(1, len(self._lanes)) * self.ROW_HEIGHT
        )
        self.updateGeometry()
        self.update()

    def clear(self) -> None:
        self.set_motion(None)

    def set_current_frame(self, frame: float) -> None:
        frame = max(0.0, min(float(frame), self._end_frame))
        if abs(frame - self._current_frame) < 0.01:
            return
        self._current_frame = frame
        if 0 <= self._selected_row < len(self._lanes):
            lane = self._lanes[self._selected_row]
            self.details_requested.emit(
                timeline_lane_details(lane, frame, self._selected_interval)
            )
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        painter.fillRect(0, 0, self.width(), self.HEADER_HEIGHT, palette.alternateBase())
        painter.setPen(palette.text().color())
        painter.drawText(
            8,
            0,
            self.LABEL_WIDTH - 12,
            self.HEADER_HEIGHT,
            Qt.AlignmentFlag.AlignVCenter,
            self.tr("EVENT TIMELINE · FRAMES"),
        )

        timeline_left = self.LABEL_WIDTH
        timeline_width = max(1, self.width() - timeline_left - 8)
        self._draw_ticks(painter, timeline_left, timeline_width)
        if not self._lanes:
            painter.setPen(palette.placeholderText().color())
            painter.drawText(
                timeline_left + 8,
                self.HEADER_HEIGHT,
                timeline_width - 8,
                self.ROW_HEIGHT,
                Qt.AlignmentFlag.AlignVCenter,
                self.tr("No embedded event tracks"),
            )
            return

        for row, lane in enumerate(self._lanes):
            top = self.HEADER_HEIGHT + row * self.ROW_HEIGHT
            if row % 2:
                painter.fillRect(0, top, self.width(), self.ROW_HEIGHT, palette.alternateBase())
            if row == self._selected_row:
                selected = QColor(palette.highlight().color())
                selected.setAlpha(45)
                painter.fillRect(0, top, self.width(), self.ROW_HEIGHT, selected)
            color = _CATEGORY_COLORS.get(lane.category, QColor("#8b96a8"))
            lane_active = any(
                self._interval_is_active(start, end)
                and not (
                    lane.segments
                    and lane.segments[index].state == "NONE"
                )
                for index, (start, end) in enumerate(
                    tuple((item.start, item.end) for item in lane.segments)
                    if lane.segments
                    else lane.intervals
                )
            )
            painter.setPen(QColor("#ffe36e") if lane_active else color)
            painter.drawText(
                8,
                top,
                self.LABEL_WIDTH - 12,
                self.ROW_HEIGHT,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"{lane.category}  {lane.name}",
            )
            drawn_intervals = (
                tuple((segment.start, segment.end) for segment in lane.segments)
                if lane.segments
                else lane.intervals
            )
            for interval_index, (start, end) in enumerate(drawn_intervals):
                x1 = self._frame_x(start, timeline_left, timeline_width)
                x2 = self._frame_x(end, timeline_left, timeline_width)
                if abs(x2 - x1) < 3:
                    x2 = x1 + 3
                segment = lane.segments[interval_index] if lane.segments else None
                interval_active = self._interval_is_active(start, end) and not (
                    segment is not None and segment.state == "NONE"
                )
                fill = QColor(
                    _PHASE_COLORS.get(segment.state, color)
                    if segment is not None
                    else color
                )
                fill.setAlpha(
                    235
                    if interval_active
                    else 65
                    if segment is not None and segment.state == "NONE"
                    else 145
                )
                segment_rect = QRectF(
                    x1, top + 5, x2 - x1, self.ROW_HEIGHT - 10
                )
                painter.fillRect(segment_rect, fill)
                if interval_active:
                    painter.setPen(QPen(QColor("#ffe36e"), 2))
                    painter.drawRect(segment_rect.adjusted(1, 1, -1, -1))
                if segment is not None and segment_rect.width() >= 38:
                    painter.setPen(palette.text().color())
                    painter.drawText(
                        segment_rect,
                        Qt.AlignmentFlag.AlignCenter,
                        segment.state,
                    )
                if row == self._selected_row and (start, end) == self._selected_interval:
                    painter.setPen(QPen(palette.highlight().color(), 2))
                    painter.drawRect(
                        QRectF(x1, top + 4, x2 - x1, self.ROW_HEIGHT - 8)
                    )
            painter.setPen(QPen(color.lighter(135), 1))
            for frame in lane.markers:
                x = self._frame_x(frame, timeline_left, timeline_width)
                painter.drawLine(x, top + 3, x, top + self.ROW_HEIGHT - 3)

        playhead = self._frame_x(self._current_frame, timeline_left, timeline_width)
        painter.setPen(QPen(QColor("#ff5c5c"), 1))
        painter.drawLine(playhead, self.HEADER_HEIGHT, playhead, self.height())

    def _draw_ticks(self, painter: QPainter, left: int, width: int) -> None:
        if self._end_frame <= 0.0:
            return
        divisions = max(2, min(10, width // 90))
        palette = self.palette()
        for index in range(divisions + 1):
            frame = self._end_frame * index / divisions
            x = left + round(width * index / divisions)
            painter.setPen(palette.mid().color())
            painter.drawLine(x, self.HEADER_HEIGHT - 4, x, self.height())
            painter.setPen(palette.text().color())
            painter.drawText(
                x + 3,
                0,
                58,
                self.HEADER_HEIGHT,
                Qt.AlignmentFlag.AlignVCenter,
                f"{frame:.0f}",
            )

    def _frame_x(self, frame: float, left: int, width: int) -> int:
        if self._end_frame <= 0.0:
            return left
        return left + round(width * max(0.0, min(frame, self._end_frame)) / self._end_frame)

    def _interval_is_active(self, start: float, end: float) -> bool:
        return _timeline_interval_is_active(
            self._current_frame,
            start,
            end,
            self._end_frame,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._select_row_at(event.position().x(), event.position().y())
            if event.position().x() >= self.LABEL_WIDTH:
                self._dragging = True
                self.scrub_started.emit()
                self._request_frame(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self._request_frame(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._request_frame(event.position().x())
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _request_frame(self, x: float) -> None:
        width = max(1, self.width() - self.LABEL_WIDTH - 8)
        frame = (x - self.LABEL_WIDTH) * self._end_frame / width
        self.frame_requested.emit(max(0.0, min(frame, self._end_frame)))

    def _select_row_at(self, x: float, y: float) -> None:
        row = int((y - self.HEADER_HEIGHT) // self.ROW_HEIGHT)
        if not 0 <= row < len(self._lanes):
            return
        lane = self._lanes[row]
        self._selected_row = row
        frame = self._frame_at_x(x)
        tolerance = self._end_frame * 5.0 / max(
            1, self.width() - self.LABEL_WIDTH - 8
        )
        selectable_intervals = (
            tuple((segment.start, segment.end) for segment in lane.segments)
            if lane.segments
            else lane.intervals
        )
        matches = [
            interval
            for interval in selectable_intervals
            if interval[0] - tolerance <= frame <= interval[1] + tolerance
        ]
        self._selected_interval = min(
            matches,
            key=lambda interval: (interval[1] - interval[0], interval[0]),
            default=None,
        )
        self.details_requested.emit(
            timeline_lane_details(lane, frame, self._selected_interval)
        )
        self.update()

    def _frame_at_x(self, x: float) -> float:
        width = max(1, self.width() - self.LABEL_WIDTH - 8)
        return max(
            0.0,
            min((x - self.LABEL_WIDTH) * self._end_frame / width, self._end_frame),
        )

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            row = (event.position().y() - self.HEADER_HEIGHT) // self.ROW_HEIGHT
            if 0 <= row < len(self._lanes):
                lane = self._lanes[row]
                QToolTip.showText(
                    event.globalPos(),
                    timeline_lane_details(lane, self._current_frame, None),
                    self,
                )
                return True
            QToolTip.hideText()
        return super().event(event)
