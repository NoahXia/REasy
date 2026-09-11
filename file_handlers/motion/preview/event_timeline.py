from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QGridLayout,
    QHeaderView,
    QLabel,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..mot.model import Motion
from ..mot_clip.model import ClipNode, ClipProperty
from .attack_collision import collision_type_label


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
    properties: tuple[ClipProperty, ...] = ()
    track_type: str = ""


@dataclass(frozen=True, slots=True)
class TimelineEventSelection:
    event_name: str
    start_frame: float
    end_frame: float
    properties: tuple[ClipProperty, ...]
    description: str = ""
    track_type: str = ""
    sample_frame: float = 0.0

    @property
    def frame_count(self) -> float:
        return max(0.0, self.end_frame - self.start_frame)


@dataclass(frozen=True, slots=True)
class EventDetailSection:
    title: str
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class EventSemanticProfile:
    display_name: str
    description: str
    timing_properties: tuple[str, ...] = ()
    marker_only: bool = False


_EVENT_PROFILES = {
    "AttackCollision_Wp": EventSemanticProfile(
        "Attack Collision · Weapon",
        "Weapon hitbox window. RequestSetID selects the collision set and "
        "AttackParamID selects the damage definition.",
        ("OnClipBox",),
    ),
    "AttackCollision_Body": EventSemanticProfile(
        "Attack Collision · Body",
        "Body hitbox window used by unarmed, body-contact, and grapple attacks.",
        ("OnClipBox",),
    ),
    "DamageCollision": EventSemanticProfile(
        "Damage Collision",
        "Enables the selected incoming-damage collision set.",
        ("OnClipBox",),
    ),
    "PressCollision": EventSemanticProfile(
        "Push Collision",
        "Enables push collision and optionally interpolates its scale.",
        ("OnClipBox",),
    ),
    "SensorCollision": EventSemanticProfile(
        "Sensor Collision",
        "Enables a non-damaging detection collision set.",
        ("On",),
    ),
    "PlayerNoHit": EventSemanticProfile(
        "Player Invulnerability",
        "Controls the player's no-hit window and filtering level.",
        ("On",),
    ),
    "EnemyNoHit": EventSemanticProfile(
        "Enemy Invulnerability",
        "Controls the enemy's no-hit window and filtering level.",
        ("On",),
    ),
    "PlayerSuperArmor": EventSemanticProfile(
        "Player Super Armor",
        "Prevents eligible hit reactions while the armor window is active.",
        ("On",),
    ),
    "EnemySuperArmor": EventSemanticProfile(
        "Enemy Super Armor",
        "Prevents eligible hit reactions while the armor window is active.",
        ("On",),
    ),
    "CharacterHitStop": EventSemanticProfile(
        "Hit Stop",
        "Applies the configured stop rate and duration during impact freeze.",
        ("HitStopAll", "On"),
    ),
    "CharacterWorkRateChange": EventSemanticProfile(
        "Animation Speed",
        "Changes the character work rate; Rate is the playback multiplier.",
        ("WorkRateSetting", "WorkRateSettingTrigger"),
    ),
    "PlayerInputRotate": EventSemanticProfile(
        "Input Rotation",
        "Allows player input to rotate the character during this window.",
        ("Enable",),
    ),
    "PlayerTrackingTarget": EventSemanticProfile(
        "Target Tracking",
        "Controls target-facing correction, distance limits, and movement rate.",
        ("Enable",),
    ),
    "AIControl": EventSemanticProfile(
        "AI Control",
        "Allows AI thinking or absolute thinking to cancel the current action.",
        ("EnableThinkCancel", "AbsThinkCancel"),
        True,
    ),
    "EnemyAttackGuide": EventSemanticProfile(
        "Enemy Attack Guide",
        "Controls whether and how the enemy attack warning is displayed.",
        ("IsDisplay",),
    ),
    "EnemyAtemi": EventSemanticProfile(
        "Enemy Counter Window",
        "Defines accepted, failed, and ignored counter types for this window.",
        ("IsAtemiSuccessAccept", "IsAtemiFailuerAccept"),
    ),
    "EnemyActionAcception": EventSemanticProfile(
        "Enemy Action Acceptance",
        "Selects which external action requests the enemy accepts.",
        ("AcceptionTypeInt",),
    ),
    "CharacterGrapple": EventSemanticProfile(
        "Grapple",
        "Controls grapple ownership and fixed-phase behavior.",
        ("PhaseFixed",),
    ),
    "AttackInformation": EventSemanticProfile(
        "Attack Information",
        "Provides attack-direction metadata used by reactions and effects.",
        ("AttackVecDir",),
    ),
    "SoulAbsorption": EventSemanticProfile(
        "Soul Absorption",
        "Selects the soul-absorption behavior active in this interval.",
        ("AbsorptionType",),
    ),
    "SpawnShell": EventSemanticProfile(
        "Spawn Shell",
        "Spawns the configured shell or gameplay projectile at a keyed frame.",
        ("Spawn",),
        True,
    ),
    "PlayerSoulBoostSpawnShell": EventSemanticProfile(
        "Soul Boost Shell",
        "Spawns the configured Soul Boost shell and its attack parameters.",
        ("Spawn",),
        True,
    ),
    "SoundTriggerTracksApp": EventSemanticProfile(
        "Sound Trigger",
        "Fires the configured Wwise trigger on each enabled Trigger key.",
        ("Trigger",),
        True,
    ),
    "SoundRegisteredContainerTriggerTracks": EventSemanticProfile(
        "Registered Sound Trigger",
        "Fires a Wwise trigger on the selected registered sound container.",
        ("Trigger",),
        True,
    ),
    "SoundRegisteredContainerLoopTriggerTracks": EventSemanticProfile(
        "Registered Loop Sound",
        "Starts or stops a looping sound on the selected registered container.",
        ("Trigger",),
        True,
    ),
    "SoundLoopTriggerTracksApp": EventSemanticProfile(
        "Loop Sound Trigger",
        "Starts or stops the configured looping Wwise trigger.",
        ("Trigger",),
        True,
    ),
    "SoundMaterialTriggerTracks": EventSemanticProfile(
        "Material Sound Trigger",
        "Fires a material-dependent sound using the configured raycast joint.",
        ("Trigger",),
        True,
    ),
    "SoundJunctionTracks": EventSemanticProfile(
        "Sound Junction",
        "Changes the sound junction state for this animation.",
        ("JunctionTriggerState",),
    ),
    "VFXRange": EventSemanticProfile(
        "VFX Range",
        "Plays EffectId on the configured joint for the authored range.",
        ("EffectId",),
    ),
    "AfterImage": EventSemanticProfile(
        "Afterimage VFX",
        "Enables the configured afterimage preset for the authored range.",
        ("PresetID",),
    ),
    "HitVfxOverwrite": EventSemanticProfile(
        "Hit VFX Override",
        "Overrides hit-effect rotation or attachment parameters.",
        ("On",),
    ),
    "VfxDrawOff": EventSemanticProfile(
        "VFX Draw Off",
        "Stops drawing the selected VFX element.",
        ("DrawOff",),
        True,
    ),
    "CameraEvent": EventSemanticProfile(
        "Camera Event",
        "Controls camera shake, camera work, auto-follow, or post effects.",
    ),
    "AppPadVibration": EventSemanticProfile(
        "Controller Vibration",
        "Plays the selected vibration preset for the configured target.",
        ("Enable",),
    ),
    "TwoSideSyncPoint": EventSemanticProfile(
        "Two-Side Sync Point",
        "Pairs left and right motion synchronization points.",
        ("LeftPoint", "RightPoint"),
    ),
    "MotionSyncPoint": EventSemanticProfile(
        "Motion Sync Point",
        "Marks a synchronization point used when blending motions.",
        ("Point",),
        True,
    ),
    "CharacterWeapon": EventSemanticProfile(
        "Weapon State",
        "Selects weapon-chain or weapon-state behavior for the action.",
        ("ChainPresetId", "Condition"),
    ),
    "NoiseRequest": EventSemanticProfile(
        "Noise Request",
        "Emits the configured gameplay noise request.",
        ("On",),
    ),
    "LookAtBlendRate": EventSemanticProfile(
        "Look-at Blend Rate",
        "Controls how strongly the character follows its look-at target.",
        ("BlendRate",),
    ),
    "LookAtParameterSetSelector": EventSemanticProfile(
        "Look-at Parameter Set",
        "Selects the look-at limits and response parameter set.",
        ("ParameterSetIndex",),
    ),
    "JointConstraintsLayer": EventSemanticProfile(
        "Joint Constraint Layer",
        "Selects and blends a joint-constraint layer.",
        ("BlendRate", "LayerIndex"),
    ),
    "CharacterBasic": EventSemanticProfile(
        "Character Motion",
        "Controls general character-motion properties such as MotionSpeed.",
        ("MotionSpeed",),
    ),
    "HumanoidBoneBasic": EventSemanticProfile(
        "Humanoid Bone Control",
        "Controls foot grounding and humanoid-bone locking values.",
        ("FootGroundAdjustRate", "FootLockL", "FootLockR"),
    ),
    "PlayerBasic": EventSemanticProfile(
        "Player State",
        "Applies player action flags and action-end movement behavior.",
    ),
    "SheathHold": EventSemanticProfile(
        "Sheath Hold",
        "Controls sheath holding and its body-state mode.",
        ("HoldOn",),
    ),
    "CharacterGrappleTilt": EventSemanticProfile(
        "Grapple Tilt / IK",
        "Blends body tilt, limb IK, and slope adjustment during grapples.",
        ("TiltBlendRate", "IKBlendRate"),
    ),
    "EnemyBasic": EventSemanticProfile(
        "Enemy State",
        "Applies enemy motion state such as target rotation or weapon condition.",
    ),
    "CharacterAnimationBasic": EventSemanticProfile(
        "Animation State",
        "Publishes animation lifecycle state such as MotionEnd.",
        ("MotionEnd",),
        True,
    ),
    "PlayerCloakControl": EventSemanticProfile(
        "Player Cloak Control",
        "Overrides the cloak simulation or display state.",
        ("OverwriteState",),
    ),
    "CharacterStanceDetail": EventSemanticProfile(
        "Character Posture",
        "Selects the authored posture for this motion range.",
        ("Posture",),
    ),
    "IkLegSwitch": EventSemanticProfile(
        "Leg IK Switch",
        "Blends leg IK behavior for grounded or lying poses.",
        ("LieDownRate",),
    ),
    "EnemyAIControl": EventSemanticProfile(
        "Enemy AI Control",
        "Animation-synchronized enemy AI control marker.",
    ),
    "JointOffsetController": EventSemanticProfile(
        "Joint Offset Controller",
        "Selects the active joint-offset parameter set.",
        ("<ActiveParameterSetIndex>k__BackingField",),
    ),
    "SoulBoostTrailVfx": EventSemanticProfile(
        "Soul Boost Trail VFX",
        "Enables or disables the Soul Boost trail effect.",
        ("DrawOn",),
    ),
    "IkLegBlendRate": EventSemanticProfile(
        "Leg IK Blend Rate",
        "Controls the contribution of leg inverse kinematics.",
        ("<BlendRate>k__BackingField",),
    ),
    "CharacterAxisXTargetAim": EventSemanticProfile(
        "Vertical Target Aim",
        "Controls vertical target aiming, limits, and blend rate.",
        ("BlendRate", "IsTargetUpdate"),
    ),
    "CharacterDirectDamage": EventSemanticProfile(
        "Direct Damage",
        "Applies configured damage directly without a normal attack hitbox.",
        ("Damage",),
    ),
    "EnemyDieFadeout": EventSemanticProfile(
        "Enemy Death Fade",
        "Enables the enemy death fade-out state.",
        ("On",),
    ),
}

_CATEGORY_DESCRIPTIONS = {
    "GAME": "Gameplay event evaluated with the animation.",
    "SOUND": "Audio event synchronized to animation frames.",
    "VFX": "Visual-effect event synchronized to animation frames.",
    "MOTION_SYNC": "Motion synchronization metadata.",
    "EXTRA_0": "Auxiliary animation-control data.",
    "PHYSICS": "Physics state synchronized to the animation.",
}

_PROPERTY_HELP = {
    "RequestSetID": "Collision request-set identifier.",
    "AttackParamID": (
        "Actor attack-parameter row. It selects damage, guard/armor damage, "
        "hit reaction, hit stop, knockback, parry/dodge rules, and related "
        "combat effects; it does not select hitbox geometry."
    ),
    "OnClipBox": "Authored activation window for the collision shape.",
    "On": "Enables this event or state during its authored range.",
    "Enable": "Enables this behavior during its authored range.",
    "Spawn": "Triggers creation at the keyed frame.",
    "ShellID": "Shell or gameplay-projectile definition identifier.",
    "HitGroup": "Hit-group channel used to separate simultaneous hitboxes.",
    "CollisionType": (
        "Weapon collision owner slot, not a primitive shape type: 0 NONE, "
        "1 SUB1_WEAPON, 2 MAIN_WEAPON, and 3–10 SUB2_WEAPON–SUB9_WEAPON."
    ),
    "HitStopID": "Hit-stop parameter identifier.",
    "StopRate": "Animation rate applied during hit stop.",
    "StopFrame": "Configured hit-stop duration in frames.",
    "NoHitLevel": "Invulnerability/filtering level.",
    "SuperArmorLevel_Fixed": "Super-armor reaction threshold identifier.",
    "Rate": "Playback or blend multiplier over the property range.",
    "TargetType": "Target category affected by the event.",
    "TriggerId": "Wwise trigger hash.",
    "Trigger": "Frames on which the audio trigger fires.",
    "TargetContainerId": "Registered sound-container hash.",
    "EffectId": "Effect resource or effect-element identifier.",
    "JointName": "Joint used as the attachment origin.",
    "DataContainerId": "Effect data-container identifier.",
    "Phase": "Cancel state: PRE buffers input, ACTUAL cancels immediately, NONE disables it.",
    "BlendRate": "Blend contribution or transition rate.",
    "ParameterSetIndex": "Selects a predefined parameter set.",
    "IsAtemiSuccessAccept": "Enables successful counter/deflection acceptance.",
    "IsAtemiFailuerAccept": "Enables failed counter/deflection handling.",
    "AtemiSuccessTypeBit": "Accepted successful counter/deflection categories.",
    "AtemiIgnoreTypeBit": "Counter/deflection categories ignored by this event.",
    "AbsorptionType": "Soul-absorption behavior selected for the range.",
    "MotionSpeed": "Character animation-speed multiplier.",
    "FootGroundAdjustRate": "Amount of procedural foot-to-ground adjustment.",
    "FootLockL": "Left-foot locking value.",
    "FootLockR": "Right-foot locking value.",
    "HoldOn": "Enables sheath holding.",
    "SheathBodyStateType": "Body-state mode used while holding the sheath.",
    "Posture": "Authored character posture.",
    "MotionEnd": "Signals that the gameplay action may treat the motion as ended.",
    "DrawOn": "Enables drawing for the effect.",
    "OverwriteState": "Replacement cloak state.",
    "ActiveParameterSetIndex": "Selects the active parameter preset.",
    "<ActiveParameterSetIndex>k__BackingField": "Selects the active parameter preset.",
    "<BlendRate>k__BackingField": "Blend contribution for this controller.",
    "Damage": "Direct damage amount or damage-setting value.",
    "AttackVecDir": "Authored attack-direction selector.",
}


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
            profile = _EVENT_PROFILES.get(track_name)
            intervals, markers = _event_timing(properties, total, profile)
            display_name = (
                profile.display_name
                if profile is not None
                else _humanize_identifier(track_name)
            )
            details = (
                profile.description
                if profile is not None
                else _CATEGORY_DESCRIPTIONS.get(
                    category,
                    "Animation event with game-specific properties.",
                )
            )
            lanes.append(
                TimelineLane(
                    category,
                    display_name,
                    total,
                    tuple(sorted(intervals)),
                    tuple(sorted(markers)),
                    details,
                    "generic",
                    properties=tuple(node.properties),
                    track_type=track_name,
                )
            )
    return tuple(lanes)


def _event_timing(
    properties: tuple[ClipProperty, ...],
    total: float,
    profile: EventSemanticProfile | None,
) -> tuple[set[tuple[float, float]], set[float]]:
    selected = properties
    if profile is not None and profile.timing_properties:
        names = set(profile.timing_properties)
        matched = tuple(
            prop for prop in properties if prop.name.lstrip("_") in names
        )
        if matched:
            selected = matched

    markers = {
        max(0.0, key.frame)
        for prop in selected
        for key in prop.keys
        if 0.0 <= key.frame <= total
        and not (isinstance(key.value, bool) and not key.value)
    }
    if profile is not None and profile.marker_only:
        return ({(frame, frame) for frame in markers}, markers)

    intervals = {
        (max(0.0, prop.start_frame), max(0.0, prop.end_frame))
        for prop in selected
        if _is_visible_interval(prop, total)
    }
    if not intervals and markers:
        intervals = {(frame, frame) for frame in markers}
    return intervals, markers


def _timeline_end_frame(
    motion: Motion | None,
    lanes: tuple[TimelineLane, ...],
) -> float:
    """Use the playable MOT duration instead of a reused event-clip length."""
    motion_end = float(motion.end_frame) if motion is not None else 0.0
    if isfinite(motion_end) and motion_end > 0.0:
        return motion_end
    return max(
        (
            lane.total_frame
            for lane in lanes
            if isfinite(lane.total_frame) and lane.total_frame > 0.0
        ),
        default=0.0,
    )


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
        lanes.append(
            TimelineLane(
                category,
                f"Cancel · {command_name}",
                total,
                active,
                tuple(sorted({key.frame for key in phase.keys})),
                "PRE buffers the command; ACTUAL permits an immediate cancel; "
                "NONE disables the cancel target.",
                "command_cancel",
                segments,
                tuple(group.children),
                "PlayerCommandCancel",
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


def _humanize_identifier(name: str) -> str:
    text = name.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return " ".join(text.split())


def timeline_lane_details(
    lane: TimelineLane,
    frame: float,
    selected_interval: tuple[float, float] | None,
) -> str:
    selection = timeline_event_selection(lane, frame, selected_interval)
    lines = [
        f"EventName: {selection.event_name}",
        f"StartFrame: {selection.start_frame:g}",
        f"EndFrame: {selection.end_frame:g}",
        f"FrameCount: {selection.frame_count:g}",
    ]
    if selection.description:
        lines.extend(("", selection.description))
    return "\n".join(lines)


def timeline_event_selection(
    lane: TimelineLane,
    frame: float,
    selected_interval: tuple[float, float] | None,
) -> TimelineEventSelection:
    interval = selected_interval
    if interval is None:
        candidates = (
            tuple((segment.start, segment.end) for segment in lane.segments)
            if lane.segments
            else lane.intervals
        )
        active = [
            item
            for item in candidates
            if _timeline_interval_is_active(frame, item[0], item[1], lane.total_frame)
        ]
        if active:
            interval = min(active, key=lambda item: (item[1] - item[0], item[0]))
        elif candidates:
            interval = (
                min(item[0] for item in candidates),
                max(item[1] for item in candidates),
            )
        else:
            interval = (0.0, max(0.0, lane.total_frame))
    return TimelineEventSelection(
        event_name=lane.name,
        start_frame=max(0.0, float(interval[0])),
        end_frame=max(0.0, float(interval[1])),
        properties=lane.properties,
        description=lane.details,
        track_type=lane.track_type,
        sample_frame=max(0.0, float(frame)),
    )


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


def _format_frame(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


class MotionEventTimeline(QWidget):
    """Compact, read-only overview of WOTS SequenceData event tracks."""

    frame_requested = Signal(float)
    scrub_started = Signal()
    details_requested = Signal(object)
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
        self._end_frame = _timeline_end_frame(motion, self._lanes)
        self._current_frame = 0.0
        self._selected_row = -1
        self._selected_interval = None
        self._dragging = False
        self.details_requested.emit(None)
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
                timeline_event_selection(lane, frame, self._selected_interval)
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
                if start > self._end_frame or end < 0.0:
                    continue
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
                if frame < 0.0 or frame > self._end_frame:
                    continue
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
            timeline_event_selection(lane, frame, self._selected_interval)
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


class MotionEventDetailsWidget(QWidget):
    """Read-only property inspector for the selected timeline event."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        event_title = QLabel(self.tr("EVENT NAME"), self)
        event_title.setObjectName("motionInspectorLabel")
        layout.addWidget(event_title)
        self.event_name = QLabel(self.tr("No event selected"), self)
        self.event_name.setWordWrap(True)
        self.event_name.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.event_name)

        metrics = QGridLayout()
        metrics.setContentsMargins(0, 2, 0, 2)
        metrics.setHorizontalSpacing(12)
        for column, title in enumerate(("StartFrame", "EndFrame", "FrameCount")):
            label = QLabel(self.tr(title), self)
            label.setObjectName("motionInspectorLabel")
            metrics.addWidget(label, 0, column)
        self.start_frame = QLabel("—", self)
        self.end_frame = QLabel("—", self)
        self.frame_count = QLabel("—", self)
        for column, label in enumerate(
            (self.start_frame, self.end_frame, self.frame_count)
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            metrics.addWidget(label, 1, column)
        layout.addLayout(metrics)

        self.description = QLabel(self)
        self.description.setWordWrap(True)
        self.description.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.description.setVisible(False)
        layout.addWidget(self.description)

        self.properties = QTreeWidget(self)
        self.properties.setHeaderLabels((self.tr("Property"), self.tr("Value")))
        self.properties.setAlternatingRowColors(True)
        self.properties.setRootIsDecorated(True)
        header = self.properties.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.properties, 1)

    def set_event(
        self,
        selection: TimelineEventSelection | None,
        sections: tuple[EventDetailSection, ...] = (),
    ) -> None:
        self.properties.clear()
        if selection is None:
            self.event_name.setText(self.tr("No event selected"))
            self.start_frame.setText("—")
            self.end_frame.setText("—")
            self.frame_count.setText("—")
            self.description.clear()
            self.description.setVisible(False)
            return
        self.event_name.setText(selection.event_name)
        self.start_frame.setText(_format_frame(selection.start_frame))
        self.end_frame.setText(_format_frame(selection.end_frame))
        self.frame_count.setText(_format_frame(selection.frame_count))
        self.description.setText(selection.description)
        self.description.setVisible(bool(selection.description))
        for prop in selection.properties:
            self._add_property(None, prop)
        for section in sections:
            if not section.rows:
                continue
            section_item = QTreeWidgetItem((section.title, ""))
            section_font = section_item.font(0)
            section_font.setBold(True)
            section_item.setFont(0, section_font)
            section_item.setExpanded(True)
            self.properties.addTopLevelItem(section_item)
            for name, value in section.rows:
                section_item.addChild(QTreeWidgetItem((name, value)))
        self.properties.expandAll()

    def _add_property(
        self,
        parent: QTreeWidgetItem | None,
        prop: ClipProperty,
    ) -> None:
        item = QTreeWidgetItem(
            (_display_property_name(prop.name), _property_display_value(prop))
        )
        help_text = _PROPERTY_HELP.get(prop.name.lstrip("_"), "")
        if help_text:
            item.setToolTip(0, help_text)
            item.setToolTip(1, help_text)
        if parent is None:
            self.properties.addTopLevelItem(item)
        else:
            parent.addChild(item)
        for child in prop.children:
            self._add_property(item, child)


def _property_display_value(prop: ClipProperty) -> str:
    if not prop.keys:
        return ""
    property_name = _display_property_name(prop.name)
    values = [
        (
            f"{int(key.value)} ({collision_type_label(int(key.value))})"
            if property_name == "CollisionType"
            and isinstance(key.value, (bool, int, float))
            else _format_value(key.value)
        )
        for key in prop.keys
    ]
    compact: list[str] = []
    for value in values:
        if not compact or value != compact[-1]:
            compact.append(value)
    return " → ".join(compact)


def _display_property_name(name: str) -> str:
    cleaned = name.lstrip("_") or name
    backing_field = re.fullmatch(r"<(.+)>k__BackingField", cleaned)
    return backing_field.group(1) if backing_field is not None else cleaned
