from __future__ import annotations

"""Read-only paired-motion previews for WOTS interaction resources."""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import math

import numpy as np

from PySide6.QtCore import (
    QElapsedTimer,
    QSignalBlocker,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from file_handlers.motion.motlist_handler import MotListHandler
from file_handlers.motion.preview.widget import MotListPreviewWidget
from file_handlers.motbank.motbank_file import MotbankFile
from file_handlers.rsz.rsz_file import RszFile
from ui.scene.scene_preview import ScenePreviewWidget
from utils.resource_file_utils import resolve_handler_resource_data

from .rsz_data_types import ArrayData, ObjectData


_ROLE = int(Qt.ItemDataRole.UserRole)
_TARGET_COUNT_ROLE = _ROLE + 1
_SLIDER_SCALE = 100


@dataclass(frozen=True, slots=True)
class InteractionMotionRef:
    set_id: int
    bank_id: int | None
    motion_id: int | None

    @property
    def available(self) -> bool:
        return (
            self.set_id not in (0xFFFFFFFF, _INVALID_MOTION_GROUP_ID)
            and self.bank_id is not None
            and self.motion_id is not None
        )

    @property
    def label(self) -> str:
        if self.bank_id is not None and self.motion_id is None:
            return f"0x{self.set_id:016X} · Bank {self.bank_id} · unresolved"
        if not self.available:
            return "Not configured"
        return (
            f"0x{self.set_id:08X} · Bank {self.bank_id} · "
            f"Motion {self.motion_id}"
        )


@dataclass(frozen=True, slots=True)
class InteractionNextAction:
    action_type: int
    action_guid: str
    action_name: str
    motion_group_id: int
    reference: InteractionMotionRef
    resource_path: str = ""
    diagnostic: str = ""
    preview_kind: str = ""


@dataclass(frozen=True, slots=True)
class _MotionSegment:
    label: str
    reference: InteractionMotionRef
    start: float
    duration: float
    root_start: np.ndarray
    root_end: np.ndarray
    root_correction: np.ndarray


def _transform_motion_snapshot(snapshot, matrix: np.ndarray):
    """Apply one row-vector actor transform to every rendered pose surface."""
    transform = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    if np.array_equal(transform, np.identity(4, dtype=np.float32)):
        return snapshot

    def transformed(value):
        source = np.asarray(value, dtype=np.float32).reshape(4, 4)
        return tuple(float(item) for item in (source @ transform).reshape(-1))

    world_matrices = tuple(
        transformed(value) for value in snapshot.pose.world_matrices
    )
    skin_matrices = tuple(
        None if value is None else transformed(value)
        for value in snapshot.pose.skin_matrices
    )
    linear = transform[:3, :3]
    translation = transform[3, :3]
    points = np.asarray(snapshot.joint_positions, dtype=np.float32).reshape(-1, 3)
    joint_positions = tuple(
        tuple(float(value) for value in point)
        for point in (points @ linear + translation)
    )
    return replace(
        snapshot,
        pose=replace(
            snapshot.pose,
            world_matrices=world_matrices,
            skin_matrices=skin_matrices,
        ),
        joint_positions=joint_positions,
    )


@dataclass(frozen=True, slots=True)
class InteractionReaction:
    group_index: int
    reaction_index: int
    trigger_groups: tuple[int, ...]
    grapple_type: int
    trigger_frame: int
    condition_flags: int
    const_type: int
    attacker_style: int
    next_action_type: int
    fixed_object_type: int
    chance_level: int
    use_system_param: bool
    disable_next_frame_control: bool
    motion_data_id: int
    rikido_break_data_id: int
    attacker_motion: InteractionMotionRef
    defender_motion: InteractionMotionRef
    defender_start_motion: InteractionMotionRef
    next_action: InteractionNextAction | None = None


@dataclass(frozen=True, slots=True)
class InteractionDocument:
    source_path: str
    attacker_role: str
    defender_role: str
    reactions: tuple[InteractionReaction, ...]
    group_count: int


@dataclass(frozen=True, slots=True)
class IssenMotionPhase:
    label: str
    action_name: str
    action_guid: str
    motion_group_id: int
    reference: InteractionMotionRef


@dataclass(frozen=True, slots=True)
class IssenPattern:
    index: int
    title: str
    grapple_type: int
    pattern_value: int
    player_phases: tuple[IssenMotionPhase, ...]
    enemy_phases: tuple[IssenMotionPhase, ...]
    fixed_object_type: int = 0


@dataclass(frozen=True, slots=True)
class IssenDocument:
    source_path: str
    patterns: tuple[IssenPattern, ...]
    attacker_role: str = "Player"
    defender_role: str = "Enemy"


@dataclass(frozen=True, slots=True)
class MotionBankCandidate:
    bank_id: int
    resource_path: str


def _scalar(value, default=0):
    raw = getattr(value, "value", value)
    return default if raw is None else raw


def _field(rsz, instance_id: int, name: str):
    return rsz.parsed_elements.get(instance_id, {}).get(name)


def _object_id(value) -> int:
    return int(value.value) if isinstance(value, ObjectData) else 0


def _object_ids(value) -> tuple[int, ...]:
    if not isinstance(value, ArrayData):
        return ()
    return tuple(
        int(item.value)
        for item in value.values
        if isinstance(item, ObjectData) and int(item.value) > 0
    )


def _motion_ref(rsz, owner_id: int, field_name: str) -> InteractionMotionRef:
    reference_id = _object_id(_field(rsz, owner_id, field_name))
    set_id = int(_scalar(_field(rsz, reference_id, "_SetID"), 0xFFFFFFFF))
    if set_id < 0:
        set_id &= 0xFFFFFFFF
    if set_id == 0xFFFFFFFF:
        return InteractionMotionRef(set_id, None, None)
    return InteractionMotionRef(set_id, set_id >> 12, set_id & 0xFFF)


def _root_type_name(rsz) -> str:
    if not getattr(rsz, "object_table", None) or rsz.type_registry is None:
        return ""
    instance_id = int(rsz.object_table[0])
    if not (0 < instance_id < len(rsz.instance_infos)):
        return ""
    info = rsz.type_registry.get_type_info(rsz.instance_infos[instance_id].type_id)
    return str((info or {}).get("name", "") or "")


def parse_wots_interaction_document(rsz, source_path: str = "") -> InteractionDocument:
    """Convert one JustGuardConditionMap RSZ object graph to stable semantics."""
    root_type = _root_type_name(rsz)
    if root_type != "app.user_data.JustGuardConditionMap":
        raise ValueError(f"unsupported interaction root type {root_type or '(unknown)'!r}")
    root_id = int(rsz.object_table[0])
    motion_data = {}
    for instance_id in _object_ids(_field(rsz, root_id, "_GrappleMotionDataList")):
        data_id = int(_scalar(_field(rsz, instance_id, "_DataId"), -1))
        motion_data[data_id] = (
            _motion_ref(rsz, instance_id, "_AttackerGrappleMotion"),
            _motion_ref(rsz, instance_id, "_DefenderGrappleMotion"),
            _motion_ref(rsz, instance_id, "_DefenderGrappleStartMotion"),
        )

    reactions = []
    group_ids = _object_ids(_field(rsz, root_id, "_GrappleList"))
    missing = InteractionMotionRef(0xFFFFFFFF, None, None)
    for group_index, group_id in enumerate(group_ids):
        trigger_groups = tuple(
            int(_scalar(_field(rsz, item_id, "_GroupSetID")))
            for item_id in _object_ids(_field(rsz, group_id, "_TriggerMotionList"))
        )
        for reaction_index, reaction_id in enumerate(
            _object_ids(_field(rsz, group_id, "_ReactionList"))
        ):
            data_id = int(
                _scalar(_field(rsz, reaction_id, "_GrappleMotionDataId"), -1)
            )
            attacker, defender, defender_start = motion_data.get(
                data_id,
                (missing, missing, missing),
            )
            reactions.append(InteractionReaction(
                group_index=group_index,
                reaction_index=reaction_index,
                trigger_groups=trigger_groups,
                grapple_type=int(_scalar(_field(rsz, reaction_id, "_GrappleType"))),
                trigger_frame=int(_scalar(_field(rsz, reaction_id, "_TriggerFrame"))),
                condition_flags=int(_scalar(_field(rsz, reaction_id, "_ConditionFlag"))),
                const_type=int(_scalar(_field(rsz, reaction_id, "_ConstType"))),
                attacker_style=int(
                    _scalar(_field(rsz, reaction_id, "_AttackerGrappleStyle"))
                ),
                next_action_type=int(
                    _scalar(_field(rsz, reaction_id, "_NextActionType"))
                ),
                fixed_object_type=int(
                    _scalar(_field(rsz, reaction_id, "_FixedObjType"))
                ),
                chance_level=int(_scalar(_field(rsz, reaction_id, "_ChanceLevel"))),
                use_system_param=bool(
                    _scalar(_field(rsz, reaction_id, "_UseSystemParam"), False)
                ),
                disable_next_frame_control=bool(
                    _scalar(
                        _field(rsz, reaction_id, "_DisableNextFrameCtrl"),
                        False,
                    )
                ),
                motion_data_id=data_id,
                rikido_break_data_id=int(
                    _scalar(
                        _field(rsz, reaction_id, "_RikidoBreakBlockIssenDataId"),
                        -1,
                    )
                ),
                attacker_motion=attacker,
                defender_motion=defender,
                defender_start_motion=defender_start,
            ))

    lowered = str(source_path).replace("\\", "/").casefold()
    if "attack_player_justguard" in lowered:
        roles = ("Enemy", "Player")
    elif "defend_player_justguard" in lowered:
        roles = ("Player", "Enemy")
    else:
        roles = ("Attacker", "Defender")
    return InteractionDocument(
        str(source_path),
        roles[0],
        roles[1],
        tuple(reactions),
        len(group_ids),
    )


_INVALID_MOTION_GROUP_ID = 0xFFFFFFFFFFFFFFFF
_PLAYER_KATATE_ATTACK_MOTLIST_PATH = (
    "natives/stm/Motion/Player/Weapon/plw_KatateAttack/plw_KatateAttack.motlist"
)
_PLAYER_RYOTE_ATTACK_MOTLIST_PATH = (
    "natives/stm/Motion/Player/Weapon/plw_RyoteAttack/plw_RyoteAttack.motlist"
)


def _player_justguard_followups() -> tuple[InteractionNextAction, ...]:
    """Return verified, input-dependent player actions after a parry.

    These are not selected by enemy NEXT_ACT_TYPE.  The player action state
    machine chooses one only when the corresponding attack branch is entered.
    Group and motion IDs were verified against the WOTS 1.0.1.0 runtime
    ActionParam and the adjacent MEX resources.
    """
    rows = (
        (
            "Attack After Parry · Front",
            0x04E2A825EDCC3B7A,
            40,
            _PLAYER_KATATE_ATTACK_MOTLIST_PATH,
        ),
        (
            "Attack After Parry · Far",
            0x04E2AC669006C141,
            44,
            _PLAYER_KATATE_ATTACK_MOTLIST_PATH,
        ),
        (
            "Attack After Parry · Front · Oni Change",
            0x04E2ACC90DEDBA79,
            46,
            _PLAYER_KATATE_ATTACK_MOTLIST_PATH,
        ),
        (
            "Attack After Parry · Far · Oni Change",
            0x04E2A7F6076A2FAF,
            47,
            _PLAYER_KATATE_ATTACK_MOTLIST_PATH,
        ),
        (
            "Attack Finish After Parry · Front",
            0x04E2BEF8EA181904,
            60,
            _PLAYER_RYOTE_ATTACK_MOTLIST_PATH,
        ),
        (
            "Attack Finish After Parry · Far",
            0x04E2BC669006C141,
            61,
            _PLAYER_RYOTE_ATTACK_MOTLIST_PATH,
        ),
        (
            "Attack Finish After Parry · Front · Oni Change",
            0x04E2BCC90DEDBA79,
            62,
            _PLAYER_RYOTE_ATTACK_MOTLIST_PATH,
        ),
        (
            "Attack Finish After Parry · Far · Oni Change",
            0x04E2B7F6076A2FAF,
            63,
            _PLAYER_RYOTE_ATTACK_MOTLIST_PATH,
        ),
    )
    followups = tuple(
        InteractionNextAction(
            0,
            "",
            label,
            group_id,
            InteractionMotionRef(group_id, group_id >> 44, motion_id),
            resource_path=resource_path,
            diagnostic="Input-dependent player action; not selected by enemy NEXT_ACT_TYPE.",
        )
        for label, group_id, motion_id, resource_path in rows
    )
    # cMultipleParry has no serialized MotionGroupID of its own. Runtime
    # inspection shows that cMultipleJustGuardBase obtains ADJUST/JUST_GUARD
    # motion details from the active GrappleMotionTable, so the widget binds
    # this marker to the currently selected reaction rather than a fixed MOT.
    return followups + (
        InteractionNextAction(
            0,
            "",
            "Multiple Parry · Repeat Current Reaction",
            _INVALID_MOTION_GROUP_ID,
            InteractionMotionRef(_INVALID_MOTION_GROUP_ID, None, None),
            diagnostic=(
                "Runtime cMultipleParry reuses the selected JustGuard "
                "GrappleMotionTable. The nearby target placement and repeated "
                "target selection are simulated."
            ),
            preview_kind="multiple_parry",
        ),
    )


_BREAK_ISSEN_COMBO_ADJUST_GUID = "074b37a4ce154b18a40c017904d10f08"
_BREAK_ISSEN_COMBO_CONST_GUID = "7082cf4496d540e0b1f456639c760356"
_BREAK_ISSEN_COMBO_DEFAULT_SELECT_FRAMES = 180.0
# app.GrappleTableDef.PATTERN_FATAL_BLOW_Fixed.  The runtime combo manager
# advances through these three authored grapple variants in order.
_BREAK_ISSEN_COMBO_PATTERN_VALUES = (
    4728653,     # COMMON_PAT_10
    1449708544,  # COMMON_PAT_11
    958406400,   # COMMON_PAT_12
)
_BREAK_ISSEN_COMBO_PATTERN_NAMES = (
    "COMMON_PAT_10",
    "COMMON_PAT_11",
    "COMMON_PAT_12",
)
_BREAK_ISSEN_MOTION_BANK_ID = 20031
_BREAK_ISSEN_MOTLIST_PATH = (
    "natives/stm/Motion/Player/Weapon/plw_Issen/plw_Issen.motlist"
)
# Verified from the WOTS 1.0.1.0 runtime TypeDB and plw_Issen_mex.user.3.
# Each row is one subsequent target; columns follow F, B, L, R.
_BREAK_ISSEN_NEXT_TURN_MOTIONS = (
    (
        (352402014515322549, 533),
        (352405801135085221, 534),
        (352393451123000886, 535),
        (352405088304742692, 536),
    ),
    (
        (352400910373180611, 538),
        (352396275015885347, 539),
        (352401101460717324, 540),
        (352396012774523291, 541),
    ),
    (
        (352405270068959620, 543),
        (352392040753985171, 544),
        (352393296696579374, 545),
        (352396668684862077, 546),
    ),
)
_BREAK_ISSEN_DIRECTIONS = ("F", "B", "L", "R")

# GrappleMotionTable stores the partner side as action GUIDs instead of motion
# groups.  These are the verified Em100 WOTS 1.0.1.0 action-to-IssenEm links.
# Valid group IDs are still resolved from the adjacent MEX at run time.
_ISSEN_ACTIONS: dict[str, tuple[str, tuple[int, int] | None]] = {
    _BREAK_ISSEN_COMBO_ADJUST_GUID: ("Break Issen Combo Adjust", None),
    _BREAK_ISSEN_COMBO_CONST_GUID: ("Break Issen Combo Const", None),
    "29cd4efa8c6d45c2a65725a57d12824e": ("Block Issen Adjust", None),
    "085f8d9ba6c94c9781164c168b0b3bad": ("Block Issen Const", None),
    "7e0fa006c2ee46aab33a990cc1e09928": (
        "Block Issen Enemy Const",
        (40031, 401),
    ),
    "67474a87483f4ffa956f785928118844": ("Counter Issen Adjust", None),
    "0230d2261f294ec094097f869e19a972": ("Counter Issen Const", None),
    "1dd502b449654130ab76fa3a27048974": ("Counter Issen Release", None),
    "78eda3b63555400cb4bfa0632f3ce36f": (
        "Counter Issen Enemy Const",
        (40031, 189),
    ),
    "472a08a42fac4488ba73752b6291707b": (
        "Counter Issen Enemy Release",
        (40031, 204),
    ),
    "9433f04aaea941f483135e5581a3a8bd": ("Chain Issen Finish Start", None),
    "9668bdce9cb5427db1f9f75581308d4f": ("Chain Issen Run Up", None),
    "5079ee0c185f40a5931a56b1367e8502": ("Chain Issen Finish Attack", None),
    "b42b09db7c154376ba90d1c674ebec6e": (
        "Enemy Match Position",
        (40031, 240),
    ),
    "d8745c26fd044248b60f42f8caf44ff3": ("Enemy Chain Issen", (40031, 250)),
    "e7d06a8a08b642af909cce360872aa05": (
        "Oni Change Finish Adjust",
        None,
    ),
    "b16539ba3bc041a58b21e1b92c933c17": (
        "Oni Change Finish Const",
        None,
    ),
    "d82872b64b654eb8aabf012b2a60ffb5": (
        "Oni Change Finish Release",
        None,
    ),
    "e39f3b0e387041e98474b50ad22333be": (
        "Enemy Fully Chain Issen Start",
        (40031, 270),
    ),
    "82019f29935e41c2bc80944cf01ba3f1": (
        "Enemy Fully Chain Issen Const",
        (40031, 271),
    ),
    "adb33f23844747c7a037f01ce169edc3": (
        "Enemy Fully Chain Issen End",
        (40031, 272),
    ),
}


def _guid_key(value) -> str:
    raw = getattr(value, "guid_str", None)
    if raw is None:
        raw = _scalar(value, "")
    return (
        str(raw)
        .replace("-", "")
        .replace("{", "")
        .replace("}", "")
        .casefold()
    )


def _issen_phase(rsz, instance_id: int, fallback_label: str) -> IssenMotionPhase:
    guid = _guid_key(_field(rsz, instance_id, "ActionGuid"))
    action_name, fallback = _ISSEN_ACTIONS.get(
        guid,
        (fallback_label, None),
    )
    group_id = int(
        _scalar(
            _field(rsz, instance_id, "MotionGroupID"),
            _INVALID_MOTION_GROUP_ID,
        )
    ) & _INVALID_MOTION_GROUP_ID
    if group_id != _INVALID_MOTION_GROUP_ID:
        reference = InteractionMotionRef(group_id, group_id >> 44, None)
    elif fallback is not None:
        bank_id, motion_id = fallback
        reference = InteractionMotionRef(
            (int(bank_id) << 12) | int(motion_id),
            int(bank_id),
            int(motion_id),
        )
    else:
        reference = InteractionMotionRef(0xFFFFFFFF, None, None)
    return IssenMotionPhase(
        fallback_label,
        action_name,
        guid,
        group_id,
        reference,
    )


def _is_break_issen_combo_pattern(pattern: IssenPattern) -> bool:
    """Return whether a grapple pattern is the paired Break Issen finisher."""
    action_guids = {phase.action_guid for phase in pattern.player_phases}
    return {
        _BREAK_ISSEN_COMBO_ADJUST_GUID,
        _BREAK_ISSEN_COMBO_CONST_GUID,
    }.issubset(action_guids)


def _break_issen_combo_patterns(
    document: IssenDocument,
) -> tuple[IssenPattern, ...]:
    """Return COMMON_PAT_10/11/12 in the runtime combo order."""
    by_value = {
        int(pattern.pattern_value): pattern
        for pattern in document.patterns
        if _is_break_issen_combo_pattern(pattern)
    }
    return tuple(
        by_value[value]
        for value in _BREAK_ISSEN_COMBO_PATTERN_VALUES
        if value in by_value
    )


def _break_issen_next_turn_phase(
    target_number: int,
    direction: str = "F",
) -> IssenMotionPhase:
    """Return one verified directional NextTurn motion for a chained target."""
    if not 1 <= int(target_number) <= len(_BREAK_ISSEN_NEXT_TURN_MOTIONS):
        raise ValueError(f"Break Issen target number is out of range: {target_number}")
    normalized_direction = str(direction).strip().upper()
    try:
        direction_index = _BREAK_ISSEN_DIRECTIONS.index(normalized_direction)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported Break Issen direction: {direction!r}"
        ) from exc
    group_set_id, motion_id = _BREAK_ISSEN_NEXT_TURN_MOTIONS[
        int(target_number) - 1
    ][direction_index]
    ordinal = ("1st", "2nd", "3rd")[int(target_number) - 1]
    action_name = f"Chain Break Issen {ordinal} Move {normalized_direction} Start"
    return IssenMotionPhase(
        label="Next Turn",
        action_name=action_name,
        action_guid="",
        motion_group_id=group_set_id,
        reference=InteractionMotionRef(
            group_set_id,
            _BREAK_ISSEN_MOTION_BANK_ID,
            motion_id,
        ),
    )


def _bind_break_issen_enemy_motion(
    player: tuple[IssenMotionPhase, ...],
    enemy: tuple[IssenMotionPhase, ...],
) -> tuple[IssenMotionPhase, ...]:
    """Bind the generic enemy death action to the paired player Const group.

    Break Issen Combo enemy actions select their concrete motion from the
    current grapple pattern at runtime, so their serialized action entry has no
    MotionGroupID of its own.  The matching enemy MEX contains the same group
    ID as the player's Const phase; carrying that ID across reproduces the
    engine's runtime selection without guessing a motion number.
    """
    const_phase = next(
        (
            phase
            for phase in reversed(player)
            if phase.action_guid == _BREAK_ISSEN_COMBO_CONST_GUID
            and phase.motion_group_id != _INVALID_MOTION_GROUP_ID
        ),
        None,
    )
    if const_phase is None:
        return enemy
    group_id = const_phase.motion_group_id
    inherited = InteractionMotionRef(group_id, group_id >> 44, None)
    return tuple(
        replace(phase, motion_group_id=group_id, reference=inherited)
        if phase.motion_group_id == _INVALID_MOTION_GROUP_ID
        else phase
        for phase in enemy
    )


def _issen_pattern_title(source_path: str, index: int, phase_count: int) -> str:
    lowered = str(source_path).replace("\\", "/").casefold()
    if "blowgrap" in lowered:
        return f"Blow Grapple {index + 1:02d}"
    if "fatalblow" in lowered:
        return f"Fatal Blow {index + 1:02d}"
    if "counterissen" in lowered:
        return f"Counter Issen {chr(ord('A') + index)}"
    if "blockissen" in lowered:
        return f"Block Issen {chr(ord('A') + index)}"
    if "chainissen" in lowered:
        if phase_count >= 3 and index >= 2:
            return "Fully Chain Issen"
        return f"Chain Issen {chr(ord('A') + index)}"
    return f"Issen Pattern {index + 1}"


def parse_wots_issen_document(rsz, source_path: str = "") -> IssenDocument:
    """Convert a WOTS Issen GrappleMotionTable into paired phase sequences."""
    root_type = _root_type_name(rsz)
    if root_type != "app.user_data.GrappleMotionTable":
        raise ValueError(f"unsupported Issen root type {root_type or '(unknown)'!r}")
    root_id = int(rsz.object_table[0])
    patterns = []
    for index, list_id in enumerate(_object_ids(_field(rsz, root_id, "_List"))):
        player_ids = _object_ids(_field(rsz, list_id, "_OwnerMotionList"))
        enemy_ids = _object_ids(_field(rsz, list_id, "_PartnerMotionList"))
        player = tuple(
            _issen_phase(rsz, instance_id, f"Player Phase {phase_index + 1}")
            for phase_index, instance_id in enumerate(player_ids)
        )
        enemy = tuple(
            _issen_phase(rsz, instance_id, f"Enemy Phase {phase_index + 1}")
            for phase_index, instance_id in enumerate(enemy_ids)
        )
        pattern = IssenPattern(
            index=index,
            title=_issen_pattern_title(source_path, index, len(player)),
            grapple_type=int(_scalar(_field(rsz, list_id, "_GrappleType"))),
            pattern_value=int(_scalar(_field(rsz, list_id, "_Pattern"))),
            player_phases=player,
            enemy_phases=enemy,
        )
        if _is_break_issen_combo_pattern(pattern):
            pattern = replace(
                pattern,
                enemy_phases=_bind_break_issen_enemy_motion(player, enemy),
            )
        patterns.append(pattern)
    return IssenDocument(str(source_path), tuple(patterns))


def parse_wots_mex_motion_map(rsz) -> dict[int, int]:
    """Return exact MotionGroupSetID -> MotionID links from one WOTS MEX."""
    result = {}
    for fields in getattr(rsz, "parsed_elements", {}).values():
        group = fields.get("_MotionGroupSetID")
        motion = fields.get("_MotionID")
        if group is None or motion is None:
            continue
        result[int(_scalar(group)) & _INVALID_MOTION_GROUP_ID] = int(_scalar(motion))
    return result


def _instance_type_name(rsz, instance_id: int) -> str:
    registry = getattr(rsz, "type_registry", None)
    infos = getattr(rsz, "instance_infos", ())
    if registry is None or not (0 < instance_id < len(infos)):
        return ""
    info = registry.get_type_info(infos[instance_id].type_id)
    return str((info or {}).get("name", "") or "")


def _referenced_object_ids(value) -> tuple[int, ...]:
    if isinstance(value, ObjectData):
        instance_id = int(value.value)
        return (instance_id,) if instance_id > 0 else ()
    if isinstance(value, ArrayData):
        return tuple(
            int(item.value)
            for item in value.values
            if isinstance(item, ObjectData) and int(item.value) > 0
        )
    return ()


def _motion_groups_below(rsz, root_id: int) -> tuple[int, ...]:
    """Collect MotionGroupIDs owned by one ActionParam entry."""
    pending = [int(root_id)]
    visited: set[int] = set()
    groups: list[int] = []
    while pending:
        instance_id = pending.pop(0)
        if instance_id <= 0 or instance_id in visited:
            continue
        visited.add(instance_id)
        fields = rsz.parsed_elements.get(instance_id, {})
        value = fields.get("_MotionGroupID")
        if value is not None:
            group_id = int(_scalar(value, _INVALID_MOTION_GROUP_ID)) & _INVALID_MOTION_GROUP_ID
            if group_id != _INVALID_MOTION_GROUP_ID and group_id not in groups:
                groups.append(group_id)
        for field_value in fields.values():
            pending.extend(_referenced_object_ids(field_value))
    return tuple(groups)


def parse_wots_action_motion_map(action_ids_rsz, action_params_rsz):
    """Map ActionID instance GUIDs to names and MotionGroupIDs.

    WOTS keeps the stable action GUIDs in an ActionID resource and the actual
    action objects in a parallel ActionParam array.  Their array positions are
    the join key; the motion module is nested below each action object.
    """
    id_root = int(action_ids_rsz.object_table[0])
    param_root = int(action_params_rsz.object_table[0])
    id_array_id = _object_id(_field(action_ids_rsz, id_root, "_ActionIDArray"))
    id_entries = _object_ids(_field(action_ids_rsz, id_array_id, "_DataArray"))
    action_entries = _object_ids(
        _field(action_params_rsz, param_root, "_ActionClassList")
    )
    result = {}
    for index, id_entry in enumerate(id_entries):
        if index >= len(action_entries):
            break
        guid = _guid_key(_field(action_ids_rsz, id_entry, "_InstanceGuid"))
        if not guid:
            continue
        action_entry = action_entries[index]
        groups = _motion_groups_below(action_params_rsz, action_entry)
        if not groups:
            continue
        full_name = _instance_type_name(action_params_rsz, action_entry)
        action_name = full_name.rsplit(".", 1)[-1] or f"Action {index}"
        result[guid] = (action_name, groups)
    return result


def _loose_stm_root(source_path: str) -> Path | None:
    path = Path(source_path)
    if not path.is_absolute():
        return None
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index].casefold() == "natives" and parts[index + 1].casefold() == "stm":
            return Path(*parts[: index + 2])
    return None


@lru_cache(maxsize=8)
def _scan_loose_motion_banks(stm_root: str) -> tuple[MotionBankCandidate, ...]:
    root = Path(stm_root)
    motion_root = root / "Motion"
    if not motion_root.is_dir():
        return ()
    candidates = []
    for path in motion_root.rglob("*.motbank.*"):
        try:
            bank = MotbankFile()
            bank.read(path.read_bytes())
        except (OSError, ValueError):
            continue
        for item in bank.items:
            if not item.path:
                continue
            resource_path = str(item.path).replace("\\", "/").lstrip("/")
            if not resource_path.casefold().startswith("natives/stm/"):
                resource_path = f"natives/stm/{resource_path}"
            candidates.append(MotionBankCandidate(int(item.bank_id), resource_path))

    # Actor-specific Grapple MOTLISTs can share a bank ID even when only one
    # path is registered by a MOTBANK.  Their adjacent MEX stores the same
    # MotionBankID, so include the sibling MOTLIST as another role candidate.
    known_bank_ids = {item.bank_id for item in candidates}
    for mex_path in motion_root.rglob("*_mex.user.*"):
        try:
            data = mex_path.read_bytes()
        except OSError:
            continue
        matching_ids = [
            bank_id
            for bank_id in known_bank_ids
            if int(bank_id).to_bytes(4, "little", signed=True) in data
        ]
        if not matching_ids:
            continue
        stem = mex_path.name.split("_mex.user.", 1)[0]
        try:
            motlists = sorted(mex_path.parent.glob(f"{stem}.motlist.*"))
        except OSError:
            motlists = []
        if not motlists:
            continue
        relative_file = motlists[0].relative_to(root)
        relative = (relative_file.parent / f"{stem}.motlist").as_posix()
        resource_path = f"natives/stm/{relative}"
        for bank_id in matching_ids:
            candidate = MotionBankCandidate(int(bank_id), resource_path)
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _context_resource(context, resource_path: str) -> tuple[str, bytes] | None:
    if context is None:
        return None
    try:
        return context.resolve(
            resource_path,
            allow_selection_dialog=False,
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _context_known_paths(context) -> tuple[str, ...]:
    reader = getattr(context, "pak_cached_reader", None)
    if reader is None:
        return ()
    cached_known_paths = getattr(reader, "cached_known_paths", None)
    try:
        paths = (
            cached_known_paths()
            if callable(cached_known_paths)
            else reader.cached_paths(include_unknown=False)
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return ()
    return tuple(str(path).replace("\\", "/").lstrip("/") for path in paths)


def _motion_actor_scope(source_path: str) -> str:
    parts = str(source_path).replace("\\", "/").casefold().split("/")
    for index, part in enumerate(parts[:-2]):
        if part == "enemy" and index > 0 and parts[index - 1] == "action":
            return (
                f"natives/stm/motion/enemy/{parts[index + 1]}/"
                f"{parts[index + 2]}/"
            )
    return ""


def _scan_context_motion_banks(
    context,
    bank_ids: set[int],
    source_path: str,
) -> tuple[MotionBankCandidate, ...]:
    """Resolve MOTBANK and adjacent MEX metadata from an owning PAK context."""
    known_paths = _context_known_paths(context)
    if not known_paths:
        return ()

    candidates: list[MotionBankCandidate] = []
    actor_scope = _motion_actor_scope(source_path)
    for path in known_paths:
        lowered = path.casefold()
        if "/motion/" not in lowered or ".motbank." not in lowered:
            continue
        if actor_scope and not lowered.startswith(actor_scope):
            continue
        hit = _context_resource(context, path)
        if hit is None:
            continue
        try:
            bank = MotbankFile()
            bank.read(hit[1])
        except (OSError, ValueError):
            continue
        for item in bank.items:
            if int(item.bank_id) not in bank_ids or not item.path:
                continue
            resource_path = str(item.path).replace("\\", "/").lstrip("/")
            if not resource_path.casefold().startswith("natives/stm/"):
                resource_path = f"natives/stm/{resource_path}"
            candidate = MotionBankCandidate(int(item.bank_id), resource_path)
            if candidate not in candidates:
                candidates.append(candidate)

    # A WOTS Grapple MOTBANK commonly registers the player list while the
    # enemy list carries the same MotionBankID only in its adjacent MEX.  Use
    # the PAK's known path index to find those sibling pairs without requiring
    # an unpacked game directory.
    companion_roots = tuple({
        candidate.resource_path.rsplit("/", 2)[0].casefold().rstrip("/") + "/"
        for candidate in candidates
        if candidate.resource_path.count("/") >= 2
    })
    if not companion_roots:
        return tuple(candidates)

    motlists = {
        path.casefold(): path
        for path in known_paths
        if ".motlist." in path.casefold()
    }
    encoded_bank_ids = {
        bank_id: int(bank_id).to_bytes(4, "little", signed=True)
        for bank_id in bank_ids
    }
    for mex_path in known_paths:
        lowered = mex_path.casefold()
        if "_mex.user." not in lowered:
            continue
        if not any(lowered.startswith(root) for root in companion_roots):
            continue
        hit = _context_resource(context, mex_path)
        if hit is None:
            continue
        basename = mex_path.rsplit("/", 1)[-1]
        marker = basename.casefold().find("_mex.user.")
        if marker < 0:
            continue
        stem = basename[:marker]
        parent = mex_path.rsplit("/", 1)[0]
        motlist_prefix = f"{parent}/{stem}.motlist".casefold()
        motlist_path = next(
            (
                original
                for key, original in motlists.items()
                if key == motlist_prefix or key.startswith(motlist_prefix + ".")
            ),
            None,
        )
        if motlist_path is None:
            continue
        resource_path = motlist_path[: motlist_path.casefold().find(".motlist") + 8]
        for bank_id, encoded in encoded_bank_ids.items():
            if encoded not in hit[1]:
                continue
            candidate = MotionBankCandidate(bank_id, resource_path)
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def discover_motion_bank_candidates(
    source_path: str,
    bank_ids: set[int],
    resource_context=None,
) -> dict[int, tuple[MotionBankCandidate, ...]]:
    result = {bank_id: [] for bank_id in bank_ids}
    roots: set[Path] = set()
    direct_root = _loose_stm_root(source_path)
    if direct_root is not None:
        roots.add(direct_root)
    for base in (
        getattr(resource_context, "project_dir", ""),
        getattr(resource_context, "unpacked_dir", ""),
    ):
        if not base:
            continue
        root = Path(base) / Path(
            str(getattr(resource_context, "path_prefix", "natives/stm"))
        )
        if root.is_dir():
            roots.add(root)
    resolved_source = _context_resource(resource_context, source_path)
    if resolved_source is not None:
        root = _loose_stm_root(resolved_source[0])
        if root is not None:
            roots.add(root)

    discovered = []
    for root in roots:
        discovered.extend(_scan_loose_motion_banks(str(root)))
    discovered.extend(
        _scan_context_motion_banks(resource_context, bank_ids, source_path)
    )
    actor_scope = _motion_actor_scope(source_path)
    scoped = [
        candidate
        for candidate in discovered
        if candidate.resource_path.casefold().startswith(actor_scope)
    ] if actor_scope else []
    if scoped:
        discovered = scoped
    for candidate in discovered:
        existing = result.get(candidate.bank_id)
        if existing is None:
            continue
        resource_key = candidate.resource_path.casefold()
        if any(item.resource_path.casefold() == resource_key for item in existing):
            continue
        existing.append(candidate)
    return {key: tuple(value) for key, value in result.items()}


def preferred_motion_bank_candidate(
    candidates: tuple[MotionBankCandidate, ...],
    role: str,
) -> MotionBankCandidate | None:
    if not candidates:
        return None
    normalized_role = role.casefold()

    def role_penalty(item: MotionBankCandidate) -> int:
        path = "/" + item.resource_path.casefold()
        basename = path.rsplit("/", 1)[-1]
        looks_player = "/motion/player/" in path or basename.startswith("plw_")
        looks_enemy = "/motion/enemy/" in path and not basename.startswith("plw_")
        if normalized_role.startswith("player"):
            return 0 if looks_player else 1
        if normalized_role.startswith("enemy"):
            return 0 if looks_enemy else 1
        return 0

    return min(
        candidates,
        key=lambda item: (
            role_penalty(item),
            "grapple" not in item.resource_path.casefold(),
            len(item.resource_path),
            item.resource_path.casefold(),
        ),
    )


def _resolve_interaction_resource(handler, resource_path: str, parent):
    hit = resolve_handler_resource_data(
        handler,
        resource_path,
        parent,
        allow_selection_dialog=False,
    )
    if hit is not None:
        return hit
    root = _loose_stm_root(str(getattr(handler, "filepath", "") or ""))
    if root is None:
        return None
    normalized = resource_path.replace("\\", "/").lstrip("/")
    prefix = "natives/stm/"
    relative = normalized[len(prefix):] if normalized.casefold().startswith(prefix) else normalized
    candidate = root / Path(relative)
    if candidate.is_file():
        return str(candidate), candidate.read_bytes()
    try:
        matches = sorted(
            item
            for item in candidate.parent.iterdir()
            if item.is_file()
            and (
                item.name.casefold() == candidate.name.casefold()
                or item.name.casefold().startswith(candidate.name.casefold() + ".")
            )
        )
    except OSError:
        matches = []
    return (str(matches[0]), matches[0].read_bytes()) if matches else None


def _enemy_action_resource_pairs(source_path: str) -> tuple[tuple[str, str], ...]:
    """Return ActionID/ActionParam pairs that own an enemy grapple action."""
    parts = str(source_path).replace("\\", "/").split("/")
    lowered = [part.casefold() for part in parts]
    try:
        enemy_index = next(
            index
            for index, part in enumerate(lowered[:-2])
            if part == "enemy" and index > 0 and lowered[index - 1] == "action"
        )
    except StopIteration:
        return ()
    actor = parts[enemy_index + 1]
    variant = parts[enemy_index + 2]
    actor_stem = actor[:1].upper() + actor[1:]
    prefix = f"natives/stm/GameDesign/Action/Enemy/{actor}/{variant}/Data/Action"
    return (
        (
            "natives/stm/GameDesign/Action/Enemy/CommonData/EmCommon_FullBody.user",
            f"{prefix}/{actor_stem}_{variant}_CommonFullBodyActionParam.user",
        ),
        (
            f"{prefix}/{actor_stem}_{variant}_FullBody.user",
            f"{prefix}/{actor_stem}_{variant}_FullBodyActionParam.user",
        ),
    )


def _load_wots_rsz_resource(handler, resource_path: str, parent=None):
    hit = _resolve_interaction_resource(handler, resource_path, parent)
    if hit is None:
        return None
    filepath, data = hit
    source_rsz = getattr(handler, "rsz_file", None)
    registry = getattr(source_rsz, "type_registry", None)
    if registry is None:
        return None
    try:
        parsed = RszFile()
        parsed.filepath = filepath
        parsed.game_version = str(
            getattr(source_rsz, "game_version", "OnimushaWOTS")
        )
        parsed.type_registry = registry
        # Several shipped WOTS ActionParam resources have registry CRCs that
        # differ from the public dump while retaining the same field layout.
        # The owning GrappleMotionTable was already validated by the handler;
        # keep this secondary, read-only lookup tolerant of those known CRCs.
        parsed.read(data, validate_type_registry=False)
        return parsed
    except (OSError, TypeError, ValueError):
        return None


def _first_guid_below(rsz, root_id: int) -> str:
    pending = [int(root_id)]
    visited: set[int] = set()
    while pending:
        instance_id = pending.pop(0)
        if instance_id <= 0 or instance_id in visited:
            continue
        visited.add(instance_id)
        for value in rsz.parsed_elements.get(instance_id, {}).values():
            if hasattr(value, "guid_str"):
                guid = _guid_key(value)
                if guid and guid != "0" * 32:
                    return guid
            pending.extend(_referenced_object_ids(value))
    return ""


def parse_wots_next_action_guid_map(rsz) -> dict[int, str]:
    """Resolve JustGuard NEXT_ACT_TYPE values to requested FullBody GUIDs.

    WOTS stores this dispatch in the actor ACTION BTable.  A branch normally
    requests the action directly; ACT_25 in Em100 jumps to a small helper
    table first, so local table jumps are followed as well.
    """
    if _root_type_name(rsz) != "ace.btable.user_data.BTable":
        return {}
    root_id = int(rsz.object_table[0])
    table_container = _object_id(_field(rsz, root_id, "_Tables"))
    table_ids = _object_ids(_field(rsz, table_container, "_DataArray"))
    table_rows = {
        _guid_key(_field(rsz, table_id, "_Guid")): _object_ids(
            _field(rsz, table_id, "_Rows")
        )
        for table_id in table_ids
    }

    def requested_guid(rows: tuple[int, ...], row_index: int) -> str:
        if not 0 <= row_index < len(rows):
            return ""
        row_fields = rsz.parsed_elements.get(rows[row_index], {})
        command_id = _object_id(row_fields.get("_CommandArgument"))
        command_type = _instance_type_name(rsz, command_id)
        if command_type.endswith("cRequestFullBodyActionArg"):
            return _first_guid_below(rsz, command_id)
        operator_id = _object_id(row_fields.get("_OperatorArgument"))
        operator_type = _instance_type_name(rsz, operator_id)
        if "OperatorArgumentJump" in operator_type:
            target_guid = _first_guid_below(rsz, operator_id)
            target_rows = table_rows.get(target_guid, ())
            for target_index in range(min(3, len(target_rows))):
                guid = requested_guid(target_rows, target_index)
                if guid:
                    return guid
        return ""

    result: dict[int, str] = {}
    for rows in table_rows.values():
        for row_index, row_id in enumerate(rows):
            fields = rsz.parsed_elements.get(row_id, {})
            command_id = _object_id(fields.get("_CommandArgument"))
            if not _instance_type_name(rsz, command_id).endswith(
                "cCheckJustGuardAfterActionTypeArg"
            ):
                continue
            check_method = _object_id(_field(rsz, command_id, "_CheckMethod"))
            action_type = int(_scalar(_field(rsz, check_method, "_Value"), 0))
            guid = requested_guid(rows, row_index + 1)
            if action_type > 0 and guid:
                result[action_type] = guid
    return result


def _enemy_action_btable_resource(source_path: str) -> str:
    parts = str(source_path).replace("\\", "/").split("/")
    lowered = [part.casefold() for part in parts]
    try:
        enemy_index = next(
            index
            for index, part in enumerate(lowered[:-2])
            if part == "enemy" and index > 0 and lowered[index - 1] == "action"
        )
    except StopIteration:
        return ""
    actor = parts[enemy_index + 1]
    variant = parts[enemy_index + 2]
    actor_stem = actor[:1].upper() + actor[1:]
    return (
        f"natives/stm/GameDesign/Action/Enemy/{actor}/{variant}/Btable/"
        f"{actor_stem}_{variant}_ACTION.user"
    )


def resolve_wots_justguard_next_actions(
    handler,
    document: InteractionDocument,
    parent=None,
) -> InteractionDocument:
    """Attach statically verifiable post-grapple enemy actions to reactions."""
    requested_types = {
        reaction.next_action_type
        for reaction in document.reactions
        if reaction.next_action_type > 0
    }
    if not requested_types:
        return document
    btable_path = _enemy_action_btable_resource(document.source_path)
    btable = _load_wots_rsz_resource(handler, btable_path, parent) if btable_path else None
    guid_map = parse_wots_next_action_guid_map(btable) if btable is not None else {}

    action_map = {}
    for id_path, param_path in _enemy_action_resource_pairs(document.source_path):
        action_ids = _load_wots_rsz_resource(handler, id_path, parent)
        action_params = _load_wots_rsz_resource(handler, param_path, parent)
        if action_ids is None or action_params is None:
            continue
        action_map.update(parse_wots_action_motion_map(action_ids, action_params))

    def resolved(reaction: InteractionReaction) -> InteractionReaction:
        action_type = int(reaction.next_action_type)
        if action_type <= 0:
            return reaction
        guid = guid_map.get(action_type, "")
        binding = action_map.get(guid)
        if not guid:
            diagnostic = (
                f"ACT_{action_type:02d} has no static FullBody branch in the "
                "owning ACTION BTable."
            )
            transition = InteractionNextAction(
                action_type,
                "",
                "Runtime-selected action",
                _INVALID_MOTION_GROUP_ID,
                InteractionMotionRef(_INVALID_MOTION_GROUP_ID, None, None),
                diagnostic=diagnostic,
            )
        elif binding is None:
            transition = InteractionNextAction(
                action_type,
                guid,
                "Unresolved FullBody action",
                _INVALID_MOTION_GROUP_ID,
                InteractionMotionRef(_INVALID_MOTION_GROUP_ID, None, None),
                diagnostic=f"Action GUID {guid} was not found in the owning ActionParam.",
            )
        else:
            action_name, groups = binding
            group_id = int(groups[0]) if groups else _INVALID_MOTION_GROUP_ID
            reference = (
                InteractionMotionRef(group_id, group_id >> 44, None)
                if group_id != _INVALID_MOTION_GROUP_ID
                else InteractionMotionRef(group_id, None, None)
            )
            transition = InteractionNextAction(
                action_type,
                guid,
                action_name,
                group_id,
                reference,
                diagnostic=(
                    "Action uses the previous motion at runtime."
                    if group_id == _INVALID_MOTION_GROUP_ID
                    else ""
                ),
            )
        return replace(reaction, next_action=transition)

    return replace(
        document,
        reactions=tuple(resolved(reaction) for reaction in document.reactions),
    )


def resolve_wots_grapple_action_phases(
    handler,
    document: IssenDocument,
    parent=None,
) -> IssenDocument:
    """Resolve partner ActionGuid phases through WOTS ActionID resources."""
    unresolved = {
        phase.action_guid
        for pattern in document.patterns
        for phase in pattern.enemy_phases
        if phase.action_guid
        and phase.motion_group_id == _INVALID_MOTION_GROUP_ID
        and phase.reference.bank_id is None
    }
    if not unresolved:
        return document

    action_map = {}
    for id_path, param_path in _enemy_action_resource_pairs(document.source_path):
        action_ids = _load_wots_rsz_resource(handler, id_path, parent)
        action_params = _load_wots_rsz_resource(handler, param_path, parent)
        if action_ids is None or action_params is None:
            continue
        action_map.update(parse_wots_action_motion_map(action_ids, action_params))
        if unresolved.issubset(action_map):
            break
    if not action_map:
        return document

    def resolved_phase(phase: IssenMotionPhase) -> IssenMotionPhase:
        if (
            phase.motion_group_id != _INVALID_MOTION_GROUP_ID
            or phase.reference.bank_id is not None
        ):
            return phase
        binding = action_map.get(phase.action_guid)
        if binding is None:
            return phase
        action_name, groups = binding
        group_id = groups[0]
        return replace(
            phase,
            action_name=action_name,
            motion_group_id=group_id,
            reference=InteractionMotionRef(group_id, group_id >> 44, None),
        )

    return replace(
        document,
        patterns=tuple(
            replace(
                pattern,
                enemy_phases=tuple(
                    resolved_phase(phase) for phase in pattern.enemy_phases
                ),
            )
            for pattern in document.patterns
        ),
    )


def _mex_resource_path(motlist_path: str) -> str:
    normalized = str(motlist_path).replace("\\", "/")
    marker = normalized.casefold().rfind(".motlist")
    stemmed = normalized[:marker] if marker >= 0 else normalized
    parent, _, stem = stemmed.rpartition("/")
    return f"{parent}/{stem}_mex.user" if parent else f"{stem}_mex.user"


def load_wots_mex_motion_map(
    handler,
    candidate: MotionBankCandidate,
    parent=None,
) -> tuple[dict[int, int], str]:
    """Resolve and parse the MEX adjacent to a discovered MOTLIST candidate."""
    mex_path = _mex_resource_path(candidate.resource_path)
    hit = _resolve_interaction_resource(handler, mex_path, parent)
    if hit is None:
        return {}, f"MEX not found: {mex_path}"
    filepath, data = hit
    source_rsz = getattr(handler, "rsz_file", None)
    registry = getattr(source_rsz, "type_registry", None)
    if registry is None:
        return {}, "MEX could not be parsed because no WOTS type registry is loaded."
    try:
        mex = RszFile()
        mex.filepath = filepath
        mex.game_version = str(getattr(source_rsz, "game_version", "OnimushaWOTS"))
        mex.type_registry = registry
        mex.read(data, validate_type_registry=True)
        mapping = parse_wots_mex_motion_map(mex)
    except (OSError, TypeError, ValueError) as exc:
        return {}, f"Could not parse {mex_path}: {exc}"
    if not mapping:
        return {}, f"No MotionGroupSetID mappings were found in {mex_path}."
    return mapping, ""


_GRAPPLE_TYPES = {0: "PARRY", 1: "BLOCK"}
_CHANCE_LEVELS = {0: "NONE", 1: "VERY_SMALL", 2: "SMALL", 3: "LARGE"}
_CONST_TYPES = {0: "SYNCHRO", 4: "SYNCHRO_IGNORE_DIP"}
_FIXED_OBJECTS = {0: "DEFENDER", 1: "ATTACKER"}


def _named(value: int, names: dict[int, str]) -> str:
    return names.get(value, f"UNKNOWN_{value}")


def _condition_label(flags: int) -> str:
    if not flags:
        return "NONE"
    names = []
    for bit, name in (
        (2, "ATTACKER_FRONT"),
        (4, "ATTACKER_BACK"),
        (8, "REACTION_LEFT"),
        (16, "REACTION_RIGHT"),
    ):
        if flags & bit:
            names.append(name)
    unknown = flags & ~30
    if unknown:
        names.append(f"0x{unknown:X}")
    return " | ".join(names)


def interaction_property_rows(
    document: InteractionDocument,
    reaction: InteractionReaction,
) -> tuple[tuple[str, str], ...]:
    groups = ", ".join(f"0x{value:016X}" for value in reaction.trigger_groups)
    group_banks = ", ".join(
        str(value >> 44) for value in reaction.trigger_groups
    )
    action_type = (
        "NONE"
        if reaction.next_action_type == 0
        else f"ACT_{reaction.next_action_type:02d}"
    )
    rows = [
        ("Evidence", "Verified resource chain; runtime transition is simulated"),
        ("Attacker", document.attacker_role),
        ("Defender", document.defender_role),
        ("Trigger Group", str(reaction.group_index)),
        ("Reaction", str(reaction.reaction_index)),
        ("Trigger MotionGroupSetID", groups or "None"),
        ("Trigger MotionBank", group_banks or "None"),
        ("GrappleType", _named(reaction.grapple_type, _GRAPPLE_TYPES)),
        ("Source TriggerFrame", str(reaction.trigger_frame)),
        ("ConditionFlag", _condition_label(reaction.condition_flags)),
        ("ChanceLevel", _named(reaction.chance_level, _CHANCE_LEVELS)),
        ("ConstType", _named(reaction.const_type, _CONST_TYPES)),
        ("FixedObjType", _named(reaction.fixed_object_type, _FIXED_OBJECTS)),
        ("UseSystemParam", str(reaction.use_system_param).lower()),
        (
            "DisableNextFrameCtrl",
            str(reaction.disable_next_frame_control).lower(),
        ),
        ("GrappleMotionDataId", str(reaction.motion_data_id)),
        ("Attacker Motion", reaction.attacker_motion.label),
        ("Defender Start Motion", reaction.defender_start_motion.label),
        ("Defender Motion", reaction.defender_motion.label),
        ("NextActionType", action_type),
        ("RikidoBreakDataId", str(reaction.rikido_break_data_id)),
    ]
    transition = reaction.next_action
    if transition is not None:
        rows.extend((
            ("Next Action Class", transition.action_name),
            ("Next Action GUID", transition.action_guid or "Runtime-selected"),
            (
                "Next Action MotionGroupID",
                (
                    f"0x{transition.motion_group_id:016X}"
                    if transition.motion_group_id != _INVALID_MOTION_GROUP_ID
                    else "Not serialized"
                ),
            ),
            ("Next Action Motion", transition.reference.label),
        ))
        if transition.diagnostic:
            rows.append(("Next Action Diagnostic", transition.diagnostic))
    return tuple(rows)


class InteractionTimeline(QWidget):
    frame_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = 0.0
        self._attacker_start = 0.0
        self._attacker_end = 0.0
        self._defender_start = 0.0
        self._defender_end = 0.0
        self._source_trigger = 0
        self._attacker_label = "Attacker"
        self._defender_label = "Defender"
        self._attacker_segments: tuple[tuple[str, float, float], ...] = ()
        self._defender_segments: tuple[tuple[str, float, float], ...] = ()
        self.setMinimumHeight(142)

    def set_role_labels(self, attacker: str, defender: str) -> None:
        self._attacker_label = str(attacker or "Attacker")
        self._defender_label = str(defender or "Defender")
        self.update()

    @property
    def end_frame(self) -> float:
        return max(self._attacker_end, self._defender_end, 1.0)

    def configure(
        self,
        attacker_end: float,
        defender_end: float,
        source_trigger: int,
        *,
        attacker_start: float = 0.0,
        defender_start: float = 0.0,
        attacker_segments: tuple[tuple[str, float, float], ...] = (),
        defender_segments: tuple[tuple[str, float, float], ...] = (),
    ) -> None:
        self._attacker_start = max(0.0, float(attacker_start))
        self._attacker_end = max(0.0, float(attacker_end))
        self._defender_start = max(0.0, float(defender_start))
        self._defender_end = max(0.0, float(defender_end))
        self._source_trigger = int(source_trigger)
        self._attacker_segments = tuple(attacker_segments)
        self._defender_segments = tuple(defender_segments)
        self.setMinimumHeight(
            142 if self._attacker_segments or self._defender_segments else 92
        )
        self._frame = min(self._frame, self.end_frame)
        self.update()

    @staticmethod
    def _frame_label(value: float) -> str:
        rounded = round(float(value))
        return str(rounded) if abs(float(value) - rounded) < 0.001 else f"{value:.2f}"

    @staticmethod
    def _short_phase_label(label: str) -> str:
        text = str(label)
        if ". " in text and text.split(". ", 1)[0].isdigit():
            text = text.split(". ", 1)[1]
        for prefix in (
            "Counter Issen Enemy ",
            "Counter Issen ",
            "Block Issen Enemy ",
            "Block Issen ",
            "Enemy Fully Chain Issen ",
            "Oni Change Finish ",
            "Chain Issen ",
            "Enemy ",
        ):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return text

    def _draw_segments(
        self,
        painter: QPainter,
        segments: tuple[tuple[str, float, float], ...],
        fallback_start: float,
        fallback_end: float,
        y: int,
        left: int,
        width: int,
        end: float,
        color: QColor,
    ) -> None:
        draw_segments = segments or (("", fallback_start, fallback_end),)
        last_index = len(draw_segments) - 1
        for index, (label, start, finish) in enumerate(draw_segments):
            start = max(0.0, float(start))
            finish = max(start, float(finish))
            x = left + round(width * start / end)
            right = left + round(width * finish / end)
            segment_width = max(1, right - x)
            fill = color.lighter(100 + (index % 2) * 18)
            painter.fillRect(x, y, segment_width, 20, fill)
            active = start <= self._frame < finish or (
                index == last_index and math.isclose(self._frame, finish)
            )
            painter.setPen(QPen(QColor(250, 210, 80) if active else color.lighter(145), 2 if active else 1))
            painter.drawRect(x, y, segment_width, 20)
            if label and segment_width >= 22:
                caption = self._short_phase_label(label)
                text = painter.fontMetrics().elidedText(
                    caption,
                    Qt.TextElideMode.ElideRight,
                    max(0, segment_width - 6),
                )
                painter.setPen(QColor(245, 245, 245))
                painter.drawText(x + 3, y + 15, text)

    def _draw_segment_summary(
        self,
        painter: QPainter,
        role: str,
        segments: tuple[tuple[str, float, float], ...],
        y: int,
        left: int,
        width: int,
    ) -> None:
        if not segments:
            return
        parts = [
            f"{self._short_phase_label(label)} "
            f"[{self._frame_label(start)}-{self._frame_label(finish)}]"
            for label, start, finish in segments
        ]
        summary = f"{role}: " + "  |  ".join(parts)
        painter.setPen(self.palette().text().color())
        painter.drawText(
            left,
            y,
            painter.fontMetrics().elidedText(
                summary,
                Qt.TextElideMode.ElideRight,
                width,
            ),
        )

    def set_current_frame(self, frame: float) -> None:
        self._frame = min(max(0.0, float(frame)), self.end_frame)
        self.update()

    def _frame_at(self, x: float) -> float:
        left, right = 92.0, max(93.0, float(self.width() - 12))
        return min(max((x - left) / (right - left), 0.0), 1.0) * self.end_frame

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.frame_requested.emit(self._frame_at(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.frame_requested.emit(self._frame_at(event.position().x()))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        left, right = 100, max(101, self.width() - 12)
        width = right - left
        end = self.end_frame

        painter.setPen(palette.text().color())
        painter.drawText(8, 33, self._attacker_label)
        painter.drawText(8, 65, self._defender_label)
        self._draw_segments(
            painter,
            self._attacker_segments,
            self._attacker_start,
            self._attacker_end,
            17,
            left,
            width,
            end,
            QColor(54, 126, 190),
        )
        self._draw_segments(
            painter,
            self._defender_segments,
            self._defender_start,
            self._defender_end,
            49,
            left,
            width,
            end,
            QColor(83, 159, 104),
        )
        self._draw_segment_summary(
            painter,
            self._attacker_label,
            self._attacker_segments,
            88,
            left,
            width,
        )
        self._draw_segment_summary(
            painter,
            self._defender_label,
            self._defender_segments,
            108,
            left,
            width,
        )

        tick_step = max(1, int(math.ceil(end / 8.0)))
        painter.setPen(QColor(130, 130, 130))
        for frame in range(0, int(end) + 1, tick_step):
            x = left + round(width * frame / end)
            painter.drawLine(x, 10, x, 72)
            painter.drawText(x + 2, 136, str(frame))

        current_x = left + round(width * self._frame / end)
        painter.setPen(QPen(QColor(245, 245, 245), 2))
        painter.drawLine(current_x, 8, current_x, 72)
        painter.setPen(QColor(236, 166, 45))
        painter.drawText(
            left,
            12,
            f"Sync frame {self._frame:.2f} · source trigger {self._source_trigger}",
        )


class _InteractionSceneHost:
    """Merge independent motion renderers into one shared scene viewport."""

    def __init__(self, viewport: ScenePreviewWidget):
        self.viewport = viewport
        self._meshes: dict[str, list] = {}
        self._overlays: dict[str, list] = {}
        self._labels: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        self._bindings: dict[str, object] = {}
        self._palettes: dict[str, np.ndarray] = {}
        self._sources: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        self._material_profiles: dict[str, dict] = {}
        self._material_images: dict[str, dict] = {}
        self._material_failures: dict[str, dict] = {}
        self._material_parameters: dict[str, dict] = {}

    @staticmethod
    def _key(actor: str, key: str) -> str:
        return f"interaction:{actor}:{key}"

    @classmethod
    def _material_key(cls, actor: str, key: str) -> str:
        return cls._key(actor, key) if key else key

    @classmethod
    def _copy_mesh(cls, actor: str, mesh):
        batches = [
            replace(
                batch,
                material_name=cls._material_key(actor, batch.material_name),
            )
            for batch in mesh.batches
        ]
        return replace(
            mesh,
            key=cls._key(actor, mesh.key),
            geometry_key=(
                cls._key(actor, mesh.geometry_key)
                if mesh.geometry_key
                else ""
            ),
            material_name=cls._material_key(actor, mesh.material_name),
            batches=batches,
        )

    def viewport_factory(self, actor: str):
        def create(parent=None, **_kwargs):
            return _InteractionViewportProxy(self, actor, parent)

        return create

    def set_scene(self, actor: str, meshes: list, *, reset_camera: bool) -> None:
        self._drop_actor_skinning(actor)
        self._meshes[actor] = [self._copy_mesh(actor, mesh) for mesh in meshes]
        self._refresh_scene(reset_camera=reset_camera)

    def _refresh_scene(self, *, reset_camera: bool = False) -> None:
        meshes = [
            mesh
            for actor_meshes in self._meshes.values()
            for mesh in actor_meshes
        ]
        self.viewport.set_scene(meshes, reset_camera=reset_camera)
        for key, binding in self._bindings.items():
            self.viewport.set_mesh_skinning(key, binding)
            source = self._sources.get(key)
            if source is not None:
                self.viewport.update_mesh_skinning_source(key, *source)
            palette = self._palettes.get(key)
            if palette is not None:
                self.viewport.update_mesh_skinning(key, palette)
        self._refresh_overlays()
        self._refresh_labels()

    def _drop_actor_skinning(self, actor: str) -> None:
        prefix = self._key(actor, "")
        keys = {key for key in self._bindings if key.startswith(prefix)}
        for mapping in (self._bindings, self._palettes, self._sources):
            for key in tuple(mapping):
                if key.startswith(prefix):
                    mapping.pop(key, None)
        if keys:
            self.viewport.clear_mesh_skinning(keys)

    def set_wireframe_overlays(self, actor: str, meshes: list) -> None:
        self._overlays[actor] = [self._copy_mesh(actor, mesh) for mesh in meshes]
        self._refresh_overlays()

    def _refresh_overlays(self) -> None:
        self.viewport.set_wireframe_overlays([
            mesh
            for actor_meshes in self._overlays.values()
            for mesh in actor_meshes
        ])

    def set_labels(self, actor: str, names, positions) -> None:
        labels = tuple(f"{actor.title()} · {name}" for name in names)
        points = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        self._labels[actor] = labels, points
        self._refresh_labels()

    def clear_labels(self, actor: str) -> None:
        self._labels.pop(actor, None)
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        names = tuple(
            name
            for actor_names, _points in self._labels.values()
            for name in actor_names
        )
        point_chunks = [points for _names, points in self._labels.values() if len(points)]
        points = (
            np.concatenate(point_chunks, axis=0)
            if point_chunks
            else np.zeros((0, 3), dtype=np.float32)
        )
        if names:
            self.viewport.set_bone_name_labels(names, points)
        else:
            self.viewport.clear_bone_name_labels()

    def set_material_map(self, actor: str, kind: str, values: dict) -> None:
        mapping = getattr(self, f"_material_{kind}")
        mapping[actor] = {
            self._material_key(actor, str(key)): value
            for key, value in values.items()
        }
        merged = {
            key: value
            for actor_values in mapping.values()
            for key, value in actor_values.items()
        }
        getattr(self.viewport, f"set_material_{kind}")(merged)

    def update_material_images(self, actor: str, values: dict) -> None:
        scoped = {
            self._material_key(actor, str(key)): value
            for key, value in values.items()
        }
        self._material_images.setdefault(actor, {}).update(scoped)
        self.viewport.update_material_images(scoped)

    def clear_actor(self, actor: str) -> None:
        self._drop_actor_skinning(actor)
        self._meshes.pop(actor, None)
        self._overlays.pop(actor, None)
        self._labels.pop(actor, None)
        for kind in ("profiles", "images", "failures", "parameters"):
            getattr(self, f"_material_{kind}").pop(actor, None)
        self._refresh_scene(reset_camera=False)
        for kind in ("profiles", "images", "failures", "parameters"):
            mapping = getattr(self, f"_material_{kind}")
            merged = {
                key: value
                for actor_values in mapping.values()
                for key, value in actor_values.items()
            }
            getattr(self.viewport, f"set_material_{kind}")(merged)


class _InteractionViewportProxy(QWidget):
    """Namespace one normal motion renderer inside a composite viewport."""

    texture_quality_changed = Signal(str)
    render_failure = Signal(str)

    def __init__(self, host: _InteractionSceneHost, actor: str, parent=None):
        super().__init__(parent)
        self.host = host
        self.actor = actor
        self._cleaned = False
        host.viewport.texture_quality_changed.connect(self.texture_quality_changed.emit)
        host.viewport.render_failure.connect(self.render_failure.emit)

    @property
    def texture_quality(self) -> str:
        return self.host.viewport.texture_quality

    def set_frame_callback(self, _callback) -> None:
        pass

    def set_scene(self, meshes, *, reset_camera=True) -> None:
        self.host.set_scene(self.actor, meshes, reset_camera=reset_camera)

    def can_use_gpu_skinning(self, binding) -> bool:
        return self.host.viewport.can_use_gpu_skinning(binding)

    def set_mesh_skinning(self, key, binding) -> None:
        scoped = self.host._key(self.actor, key)
        self.host._bindings[scoped] = binding
        self.host.viewport.set_mesh_skinning(scoped, binding)

    def update_mesh_skinning(self, key, matrices) -> None:
        scoped = self.host._key(self.actor, key)
        palette = np.asarray(matrices, dtype=np.float32)
        self.host._palettes[scoped] = palette
        self.host.viewport.update_mesh_skinning(scoped, palette)

    def update_mesh_skinning_source(self, key, positions, normals) -> None:
        scoped = self.host._key(self.actor, key)
        source = (
            np.asarray(positions, dtype=np.float32),
            None if normals is None else np.asarray(normals, dtype=np.float32),
        )
        self.host._sources[scoped] = source
        self.host.viewport.update_mesh_skinning_source(scoped, *source)

    def clear_mesh_skinning(self, keys=None) -> None:
        if keys is None:
            self.host._drop_actor_skinning(self.actor)
            return
        scoped_keys = {self.host._key(self.actor, key) for key in keys}
        for mapping in (
            self.host._bindings,
            self.host._palettes,
            self.host._sources,
        ):
            for key in scoped_keys:
                mapping.pop(key, None)
        self.host.viewport.clear_mesh_skinning(scoped_keys)

    def update_mesh_geometry(
        self,
        key,
        vertices,
        normals=None,
        *,
        recompute_bounds=True,
    ) -> None:
        self.host.viewport.update_mesh_geometry(
            self.host._key(self.actor, key),
            vertices,
            normals,
            recompute_bounds=recompute_bounds,
        )

    def update_mesh_transforms(self, matrices, *, recompute_bounds=True) -> None:
        self.host.viewport.update_mesh_transforms(
            {
                self.host._key(self.actor, key): value
                for key, value in matrices.items()
            },
            recompute_bounds=recompute_bounds,
        )

    def set_wireframe_overlays(self, meshes) -> None:
        self.host.set_wireframe_overlays(self.actor, meshes)

    def update_wireframe_overlay_geometries(self, geometries) -> None:
        self.host.viewport.update_wireframe_overlay_geometries({
            self.host._key(self.actor, key): value
            for key, value in geometries.items()
        })

    def set_bone_name_labels(self, names, positions) -> None:
        self.host.set_labels(self.actor, names, positions)

    def clear_bone_name_labels(self) -> None:
        self.host.clear_labels(self.actor)

    def set_bone_name_labels_visible(self, visible) -> None:
        self.host.viewport.set_bone_name_labels_visible(bool(visible))

    def set_material_profiles(self, values) -> None:
        self.host.set_material_map(self.actor, "profiles", values)

    def set_material_images(self, values) -> None:
        self.host.set_material_map(self.actor, "images", values)

    def update_material_images(self, values) -> None:
        self.host.update_material_images(self.actor, values)

    def set_material_failures(self, values) -> None:
        self.host.set_material_map(self.actor, "failures", values)

    def set_material_parameters(self, values) -> None:
        self.host.set_material_map(self.actor, "parameters", values)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.host.clear_actor(self.actor)


class _ActorMotionPane(QWidget):
    motion_loaded = Signal()

    def __init__(self, owner_handler, role: str, viewport_factory, parent=None):
        super().__init__(parent)
        self.owner_handler = owner_handler
        self.role = role
        self._viewport_factory = viewport_factory
        self._preview: MotListPreviewWidget | None = None
        self._preview_handler: MotListHandler | None = None
        self._loaded_path = ""
        self._references: tuple[tuple[str, InteractionMotionRef], ...] = ()
        self._sequence: tuple[_MotionSegment, ...] = ()
        self._sequence_signature: tuple | None = None
        self._active_motion_id: int | None = None
        self._segment_correction = np.zeros(3, dtype=np.float32)
        self._world_transform = np.identity(4, dtype=np.float32)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        title = QLabel(role, self)
        title.setObjectName("motionInspectorLabel")
        root.addWidget(title)
        selectors = QHBoxLayout()
        self.motion_combo = QComboBox(self)
        self.motion_combo.currentIndexChanged.connect(self._reload)
        selectors.addWidget(self.motion_combo, 1)
        self.path_combo = QComboBox(self)
        self.path_combo.currentIndexChanged.connect(self._reload)
        selectors.addWidget(self.path_combo, 2)
        root.addLayout(selectors)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)

    @property
    def end_frame(self) -> float:
        return self._preview.preview_end_frame if self._preview is not None else 0.0

    @property
    def sequence_duration(self) -> float:
        return max(
            (segment.start + segment.duration for segment in self._sequence),
            default=0.0,
        )

    @property
    def sequence_segments(self) -> tuple[_MotionSegment, ...]:
        return self._sequence

    def set_content(
        self,
        references: tuple[tuple[str, InteractionMotionRef], ...],
        candidates: tuple[MotionBankCandidate, ...],
    ) -> None:
        self._references = tuple((label, ref) for label, ref in references if ref.available)
        with QSignalBlocker(self.motion_combo), QSignalBlocker(self.path_combo):
            self.motion_combo.clear()
            for label, ref in self._references:
                self.motion_combo.addItem(f"{label}: {ref.label}", ref)
            self.path_combo.clear()
            for candidate in candidates:
                self.path_combo.addItem(candidate.resource_path, candidate.resource_path)
            preferred = preferred_motion_bank_candidate(candidates, self.role)
            if preferred is not None:
                index = self.path_combo.findData(preferred.resource_path)
                self.path_combo.setCurrentIndex(max(0, index))
        self._reload()

    def seek(self, frame: float) -> None:
        if self._preview is not None:
            self._preview.seek_frame(frame)

    def _transform_snapshot(self, snapshot):
        return _transform_motion_snapshot(snapshot, self._world_transform)

    def _select_reference(self, reference: InteractionMotionRef) -> bool:
        preview = self._preview
        motion_id = int(reference.motion_id or 0)
        if preview is None:
            return False
        current = preview.current_entry
        if current is None or current.motion_id != motion_id:
            if not preview.select_motion_id(motion_id, bank_id=reference.bank_id):
                return False
        self._active_motion_id = motion_id
        for index, (_label, candidate) in enumerate(self._references):
            if candidate == reference:
                with QSignalBlocker(self.motion_combo):
                    self.motion_combo.setCurrentIndex(index)
                break
        rig_name = preview.rig_label.text()
        self.status.setText(
            f"{self._loaded_path}\n{reference.label}\n{rig_name}"
        )
        return True

    def _root_translation(self) -> np.ndarray:
        preview = self._preview
        if preview is None or not preview.controller.ready:
            return np.zeros(3, dtype=np.float32)
        snapshot = preview.controller.sample()
        rig = preview.controller.rig
        root_index = next(
            (
                index
                for index, joint in enumerate(rig.joints if rig is not None else ())
                if joint.parent_index is None
            ),
            0,
        )
        matrix = np.asarray(
            snapshot.pose.world_matrices[root_index],
            dtype=np.float32,
        ).reshape(4, 4)
        return matrix[3, :3].copy()

    def configure_sequence(
        self,
        labels: tuple[str, ...],
        start_frames: tuple[float, ...] | None = None,
        *,
        preserve_root_continuity: bool = True,
    ) -> None:
        starts = tuple(float(value) for value in start_frames or ())
        signature = (
            self._loaded_path,
            starts,
            bool(preserve_root_continuity),
            tuple(
                (label, ref.set_id, ref.bank_id, ref.motion_id)
                for label, ref in self._references
                if label in labels
            ),
        )
        if signature == self._sequence_signature:
            return
        self._sequence_signature = signature
        self._sequence = ()
        if self._preview is None:
            return

        by_label = {label: ref for label, ref in self._references}
        segments = []
        corrected_end = None
        contiguous_start = 0.0
        for index, label in enumerate(labels):
            reference = by_label.get(label)
            if reference is None or not self._select_reference(reference):
                continue
            duration = max(0.0, float(self._preview.preview_end_frame))
            self._preview.controller.set_frame(0.0)
            root_start = self._root_translation()
            self._preview.controller.set_frame(duration)
            root_end = self._root_translation()
            correction = (
                np.zeros(3, dtype=np.float32)
                if corrected_end is None or not preserve_root_continuity
                else corrected_end - root_start
            )
            corrected_end = (
                root_end + correction
                if preserve_root_continuity
                else None
            )
            segments.append(_MotionSegment(
                label,
                reference,
                starts[index] if index < len(starts) else contiguous_start,
                duration,
                root_start,
                root_end,
                correction,
            ))
            contiguous_start += duration
        self._sequence = tuple(segments)
        if self._sequence:
            self._select_reference(self._sequence[0].reference)

    def prepare_sequence_frame(self, frame: float, start_frame: float) -> np.ndarray:
        if self._preview is None or not self._sequence:
            return np.zeros(3, dtype=np.float32)
        local_clock = max(0.0, float(frame) - float(start_frame))
        selected = self._sequence[0]
        local_frame = 0.0
        for segment in self._sequence:
            segment_end = segment.start + segment.duration
            if local_clock < segment.start:
                break
            selected = segment
            local_frame = segment.duration
            if local_clock <= segment_end:
                selected = segment
                local_frame = min(
                    max(local_clock - segment.start, 0.0),
                    segment.duration,
                )
                break
        self._select_reference(selected.reference)
        self._segment_correction = selected.root_correction.copy()
        self._preview.controller.set_frame(local_frame)
        return self._root_translation() + self._segment_correction

    def set_anchor_translation(self, translation) -> None:
        self._world_transform = np.identity(4, dtype=np.float32)
        self._world_transform[3, :3] = (
            self._segment_correction
            + np.asarray(translation, dtype=np.float32).reshape(3)
        )

    def set_world_transform(self, transform) -> None:
        """Apply a complete row-vector world transform to this actor."""
        value = np.asarray(transform, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(value).all():
            raise ValueError("actor world transform must contain finite values")
        self._world_transform = value.copy()

    def render_prepared_frame(self) -> None:
        if self._preview is not None:
            self._preview._render()

    def _clear_preview(self) -> None:
        self._sequence = ()
        self._sequence_signature = None
        self._active_motion_id = None
        if self._preview is None:
            return
        self._preview.cleanup()
        self._preview.deleteLater()
        self._preview = None
        self._preview_handler = None
        self._loaded_path = ""

    def _reload(self, *_args) -> None:
        ref = self.motion_combo.currentData()
        resource_path = str(self.path_combo.currentData() or "")
        if not isinstance(ref, InteractionMotionRef):
            self._clear_preview()
            self.status.setText("No motion is configured for this role.")
            self.motion_loaded.emit()
            return
        if not resource_path:
            self._clear_preview()
            self.status.setText(
                f"No MOTBANK path was found for Bank {ref.bank_id}."
            )
            self.motion_loaded.emit()
            return

        if self._preview is not None and self._loaded_path == resource_path:
            found = self._preview.select_motion_id(int(ref.motion_id or 0))
            self.status.setText(
                f"{resource_path}\n{ref.label}"
                if found
                else f"Motion {ref.motion_id} is missing from {resource_path}."
            )
            self.motion_loaded.emit()
            return

        hit = _resolve_interaction_resource(
            self.owner_handler,
            resource_path,
            self,
        )
        if hit is None:
            self._clear_preview()
            self.status.setText(f"Could not resolve {resource_path}.")
            self.motion_loaded.emit()
            return
        filepath, data = hit
        self._clear_preview()
        try:
            handler = MotListHandler()
            handler.filepath = filepath
            handler.app = getattr(self.owner_handler, "app", None)
            handler.resource_context = getattr(
                self.owner_handler,
                "resource_context",
                None,
            )
            handler.read(data)
            preview = MotListPreviewWidget(
                handler,
                viewport_factory=self._viewport_factory,
                # Both actor previews share one OpenGL viewport. Resolve their
                # materials locally so Windows process spawning cannot re-enter
                # the interaction document while either target is being replaced.
                material_parse_in_subprocess=False,
                snapshot_transform=self._transform_snapshot,
            )
            preview.setParent(self)
            preview.hide()
            preview.playback.set_stop_on_hide(False)
            preview.play_pause_shortcut.setEnabled(False)
            preview.attack_hitboxes_toggle.setChecked(False)
            preview.bone_names_toggle.setChecked(False)
            normalized_role = self.role.casefold()
            preset_key = (
                "wots_ch001_00"
                if normalized_role.startswith("player")
                else "wots_em101_00"
                if normalized_role.startswith("enemy")
                else ""
            )
            preset_index = preview.model_preset_combo.findData(preset_key)
            if preset_index >= 0:
                preview.model_preset_combo.setCurrentIndex(preset_index)
            if not preview.select_motion_id(int(ref.motion_id or 0)):
                preview.cleanup()
                preview.deleteLater()
                raise ValueError(
                    f"Motion {ref.motion_id} is missing from {resource_path}"
                )
            if preset_index >= 0:
                preview._load_model_preset()
        except (OSError, ValueError) as exc:
            self._clear_preview()
            self.status.setText(str(exc))
            self.motion_loaded.emit()
            return

        self._preview_handler = handler
        self._preview = preview
        self._loaded_path = resource_path
        rig_name = preview.rig_label.text()
        self.status.setText(f"{resource_path}\n{ref.label}\n{rig_name}")
        self.motion_loaded.emit()

    def cleanup(self) -> None:
        self._clear_preview()


class WotsInteractionPreviewWidget(QWidget):
    """Pair two existing MOT preview surfaces from a Just Guard reaction."""

    def __init__(self, handler, document: InteractionDocument, parent=None):
        super().__init__(parent)
        self.handler = handler
        self.document = document
        self._current_reaction: InteractionReaction | None = None
        self._initial_reaction_item: QTreeWidgetItem | None = None
        self._activated = False
        self._frame = 0.0
        self._attacker_start = 0.0
        self._defender_start = 0.0
        self._next_action_start = 0.0
        self._next_action_role = ""
        self._next_action_label = ""
        self._next_action_transition: InteractionNextAction | None = None
        self._player_followup_start = 0.0
        self._player_followup_role = ""
        self._player_followup_label = ""
        self._player_followup_labels: tuple[str, ...] = ()
        self._player_followup_transition: InteractionNextAction | None = None
        self._multiple_parry_active = False
        self._multiple_enemy_role = (
            "attacker" if self.document.attacker_role == "Enemy" else "defender"
        )
        self._multiple_target_offset = np.array(
            (0.0, 0.0, 2.5),
            dtype=np.float32,
        )
        self._configuring_content = False
        self._mex_maps: dict[str, tuple[dict[int, int], str]] = {}
        self._playing = False
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        bank_ids = {
            ref.bank_id
            for reaction in document.reactions
            for ref in (
                reaction.attacker_motion,
                reaction.defender_motion,
                reaction.defender_start_motion,
                *(
                    (reaction.next_action.reference,)
                    if reaction.next_action is not None
                    else ()
                ),
            )
            if ref.bank_id is not None
        }
        self._bank_candidates = discover_motion_bank_candidates(
            document.source_path,
            {int(value) for value in bank_ids},
            getattr(handler, "resource_context", None),
        )
        self._build_ui()
        actors = self.attacker.parentWidget()
        next_actor_key = (
            "attacker" if self.document.attacker_role == "Enemy" else "defender"
        )
        self.next_action = _ActorMotionPane(
            self.handler,
            "Enemy Next Action",
            self._scene_host.viewport_factory(next_actor_key),
            parent=actors,
        )
        self.next_action.motion_loaded.connect(self._motion_loaded)
        self.next_action.hide()
        player_actor_key = (
            "attacker" if self.document.attacker_role == "Player" else "defender"
        )
        self.player_followup = _ActorMotionPane(
            self.handler,
            "Player Follow-up",
            self._scene_host.viewport_factory(player_actor_key),
            parent=actors,
        )
        self.player_followup.motion_loaded.connect(self._motion_loaded)
        self.player_followup.hide()
        self.multiple_enemy = _ActorMotionPane(
            self.handler,
            "Enemy Nearby",
            self._scene_host.viewport_factory("multiple_enemy"),
            parent=actors,
        )
        self.multiple_enemy.motion_loaded.connect(self._motion_loaded)
        actors.addWidget(self.multiple_enemy)
        self.multiple_enemy.hide()
        self._populate_reactions()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(5)
        banner = QLabel(
            self.tr(getattr(
                self,
                "preview_description",
                "Read-only WOTS interaction preview. Resource links are verified; "
                "the paired transition and constraint behavior are simulated.",
            )),
            self,
        )
        banner.setWordWrap(True)
        banner.setObjectName("motionStatusBar")
        root.addWidget(banner)

        body = QSplitter(Qt.Orientation.Horizontal, self)
        self.reaction_tree = QTreeWidget(body)
        self.reaction_tree.setHeaderLabels((self.tr("Reaction"), self.tr("Value")))
        self.reaction_tree.setAlternatingRowColors(True)
        self.reaction_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.reaction_tree.currentItemChanged.connect(self._on_reaction_changed)
        self.reaction_tree.setMinimumWidth(300)

        preview_area = QWidget(body)
        preview_layout = QVBoxLayout(preview_area)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        actors = QSplitter(Qt.Orientation.Horizontal, preview_area)
        settings = getattr(getattr(self.handler, "app", None), "settings", None)
        self.viewport = ScenePreviewWidget(
            preview_area,
            controls="motion",
            settings=settings if isinstance(settings, dict) else None,
            initial_distance=4.5,
        )
        self.viewport.setMinimumHeight(420)
        self._scene_host = _InteractionSceneHost(self.viewport)
        self.attacker = _ActorMotionPane(
            self.handler,
            self.document.attacker_role,
            self._scene_host.viewport_factory("attacker"),
            parent=actors,
        )
        self.defender = _ActorMotionPane(
            self.handler,
            self.document.defender_role,
            self._scene_host.viewport_factory("defender"),
            parent=actors,
        )
        self.attacker.motion_loaded.connect(self._motion_loaded)
        self.defender.motion_loaded.connect(self._motion_loaded)
        actors.addWidget(self.attacker)
        actors.addWidget(self.defender)
        actors.setSizes([700, 700])
        preview_layout.addWidget(actors)
        preview_layout.addWidget(self.viewport, 1)

        self.timeline = InteractionTimeline(preview_area)
        self.timeline.set_role_labels(
            self.document.attacker_role,
            self.document.defender_role,
        )
        self.timeline.frame_requested.connect(self.set_frame)
        preview_layout.addWidget(self.timeline)

        controls = QHBoxLayout()
        if isinstance(self.document, InteractionDocument):
            controls.addWidget(QLabel(self.tr("Player Follow-up"), preview_area))
            self.player_followup_combo = QComboBox(preview_area)
            self.player_followup_combo.addItem(
                self.tr("No Input · Hold End Pose"),
                None,
            )
            for transition in _player_justguard_followups():
                self.player_followup_combo.addItem(
                    transition.action_name,
                    transition,
                )
            self.player_followup_combo.setToolTip(self.tr(
                "Player follow-up is input-dependent and is not encoded by the "
                "enemy NextActionType. Choose a branch to preview it."
            ))
            self.player_followup_combo.currentIndexChanged.connect(
                self._on_player_followup_changed
            )
            controls.addWidget(self.player_followup_combo)
        self.play_button = QPushButton(self.tr("Play"), preview_area)
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, preview_area)
        self.frame_slider.valueChanged.connect(
            lambda value: self.set_frame(value / _SLIDER_SCALE)
        )
        controls.addWidget(self.frame_slider, 1)
        self.frame_spin = QDoubleSpinBox(preview_area)
        self.frame_spin.setDecimals(2)
        self.frame_spin.setSingleStep(0.25)
        self.frame_spin.valueChanged.connect(self.set_frame)
        controls.addWidget(self.frame_spin)
        controls.addWidget(QLabel(self.tr("60 fps"), preview_area))
        preview_layout.addLayout(controls)

        self.details = QTableWidget(body)
        self.details.setColumnCount(2)
        self.details.setHorizontalHeaderLabels((self.tr("Property"), self.tr("Value")))
        self.details.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.details.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.details.verticalHeader().hide()
        self.details.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.details.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )
        self.details.setMinimumWidth(330)

        body.addWidget(self.reaction_tree)
        body.addWidget(preview_area)
        body.addWidget(self.details)
        body.setSizes([330, 1100, 380])
        root.addWidget(body, 1)

        self.play_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.play_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.play_shortcut.activated.connect(self.toggle_playback)

    def _populate_reactions(self) -> None:
        groups = {}
        for index, reaction in enumerate(self.document.reactions):
            parent = groups.get(reaction.group_index)
            if parent is None:
                trigger = ", ".join(
                    f"0x{value:016X}" for value in reaction.trigger_groups
                ) or "No trigger group"
                parent = QTreeWidgetItem((f"Group {reaction.group_index}", trigger))
                self.reaction_tree.addTopLevelItem(parent)
                groups[reaction.group_index] = parent
            kind = _named(reaction.grapple_type, _GRAPPLE_TYPES)
            item = QTreeWidgetItem((
                f"{kind} · Reaction {reaction.reaction_index}",
                f"Frame {reaction.trigger_frame} · Data {reaction.motion_data_id}",
            ))
            item.setData(0, _ROLE, index)
            parent.addChild(item)
        self.reaction_tree.expandAll()
        first = (
            self.reaction_tree.topLevelItem(0).child(0)
            if self.reaction_tree.topLevelItemCount()
            and self.reaction_tree.topLevelItem(0).childCount()
            else None
        )
        if first is not None:
            self._initial_reaction_item = first

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._activated and self._initial_reaction_item is not None:
            self._activated = True
            QTimer.singleShot(0, self._activate_initial_reaction)

    def _activate_initial_reaction(self) -> None:
        if self._initial_reaction_item is not None and self.isVisible():
            self.reaction_tree.setCurrentItem(self._initial_reaction_item)

    def _candidate_map(self, candidate: MotionBankCandidate) -> tuple[dict[int, int], str]:
        key = candidate.resource_path.casefold()
        cached = self._mex_maps.get(key)
        if cached is None:
            cached = load_wots_mex_motion_map(self.handler, candidate, self)
            self._mex_maps[key] = cached
        return cached

    def _resolve_next_action(
        self,
        transition: InteractionNextAction | None,
    ) -> InteractionNextAction | None:
        if (
            transition is None
            or transition.reference.motion_id is not None
            or transition.reference.bank_id is None
        ):
            return transition
        candidates = self._bank_candidates.get(
            int(transition.reference.bank_id),
            (),
        )
        for candidate in candidates:
            mapping, _message = self._candidate_map(candidate)
            motion_id = mapping.get(transition.motion_group_id)
            if motion_id is None:
                continue
            return replace(
                transition,
                reference=replace(transition.reference, motion_id=int(motion_id)),
                resource_path=candidate.resource_path,
            )
        return replace(
            transition,
            diagnostic=(
                transition.diagnostic
                or f"MotionGroupID 0x{transition.motion_group_id:016X} was not found "
                "in an adjacent MEX resource."
            ),
        )

    def _configure_player_followup(self) -> None:
        combo = getattr(self, "player_followup_combo", None)
        pane = getattr(self, "player_followup", None)
        multiple_enemy = getattr(self, "multiple_enemy", None)
        if combo is None or pane is None:
            return
        transition = combo.currentData()
        if not isinstance(transition, InteractionNextAction):
            self._player_followup_transition = None
            self._player_followup_label = ""
            self._player_followup_labels = ()
            self._multiple_parry_active = False
            pane.configure_sequence(())
            if multiple_enemy is not None:
                multiple_enemy.set_content((), ())
                multiple_enemy.hide()
            return
        self._player_followup_transition = transition
        self._multiple_parry_active = transition.preview_kind == "multiple_parry"
        self._player_followup_label = f"Player Follow-up · {transition.action_name}"
        if self._multiple_parry_active and self._current_reaction is not None:
            reaction = self._current_reaction
            player_references = (
                (
                    ("Multiple Parry · ADJUST", reaction.defender_start_motion),
                    ("Multiple Parry · JUST_GUARD", reaction.defender_motion),
                )
                if self.document.defender_role == "Player"
                else (("Multiple Parry · JUST_GUARD", reaction.attacker_motion),)
            )
            player_references = tuple(
                item for item in player_references if item[1].available
            )
            self._player_followup_labels = tuple(
                label for label, _reference in player_references
            )
            player_bank_ids = {
                int(reference.bank_id)
                for _label, reference in player_references
                if reference.bank_id is not None
            }
            player_candidates = tuple(dict.fromkeys(
                candidate
                for bank_id in player_bank_ids
                for candidate in self._bank_candidates.get(bank_id, ())
            ))
            pane.set_content(player_references, player_candidates)
        else:
            self._player_followup_labels = (self._player_followup_label,)
            pane.set_content(
                ((self._player_followup_label, transition.reference),),
                (
                    MotionBankCandidate(
                        int(transition.reference.bank_id),
                        transition.resource_path,
                    ),
                ),
            )
        if multiple_enemy is None:
            return
        if not self._multiple_parry_active or self._current_reaction is None:
            multiple_enemy.set_content((), ())
            multiple_enemy.hide()
            return
        reaction = self._current_reaction
        enemy_reference = (
            reaction.attacker_motion
            if self.document.attacker_role == "Enemy"
            else reaction.defender_motion
        )
        candidates = self._bank_candidates.get(
            int(enemy_reference.bank_id or -1),
            (),
        )
        multiple_enemy.set_content(
            (("Nearby Enemy · Reaction", enemy_reference),),
            candidates,
        )
        multiple_enemy.show()

    def _on_player_followup_changed(self, _index: int) -> None:
        if not hasattr(self, "player_followup"):
            return
        self.stop_playback()
        self._configuring_content = True
        self._configure_player_followup()
        self._configuring_content = False
        self._motion_loaded()

    def _on_reaction_changed(self, current, _previous) -> None:
        index = current.data(0, _ROLE) if current is not None else None
        if not isinstance(index, int) or not (0 <= index < len(self.document.reactions)):
            return
        self.stop_playback()
        reaction = self.document.reactions[index]
        self._current_reaction = reaction
        transition = self._resolve_next_action(reaction.next_action)
        resolved_reaction = replace(reaction, next_action=transition)
        self._set_details(interaction_property_rows(self.document, resolved_reaction))
        self._next_action_transition = transition
        self._next_action_role = (
            "attacker" if self.document.attacker_role == "Enemy" else "defender"
        )
        self._player_followup_role = (
            "attacker" if self.document.attacker_role == "Player" else "defender"
        )
        self._next_action_label = (
            f"Next Action · {transition.action_name}"
            if transition is not None and transition.reference.available
            else ""
        )
        attacker_candidates = self._bank_candidates.get(
            int(reaction.attacker_motion.bank_id or -1),
            (),
        )
        defender_bank_ids = {
            int(ref.bank_id)
            for ref in (reaction.defender_motion, reaction.defender_start_motion)
            if ref.bank_id is not None
        }
        defender_candidates = tuple(dict.fromkeys(
            candidate
            for bank_id in defender_bank_ids
            for candidate in self._bank_candidates.get(bank_id, ())
        ))
        next_candidates = (
            (
                MotionBankCandidate(
                    int(transition.reference.bank_id),
                    transition.resource_path,
                ),
            )
            if transition is not None
            and transition.reference.available
            and transition.reference.bank_id is not None
            and transition.resource_path
            else ()
        )
        self._configuring_content = True
        if self._next_action_label and transition is not None:
            self.next_action.set_content(
                ((self._next_action_label, transition.reference),),
                next_candidates,
            )
        else:
            # This pane shares the enemy scene namespace with the normal
            # interaction pane.  Keep its cached preview alive so clearing it
            # cannot remove the visible enemy mesh while changing reactions.
            self.next_action.configure_sequence(())
        self._configure_player_followup()
        self.attacker.set_content(
            (("Main", reaction.attacker_motion),),
            attacker_candidates,
        )
        self.defender.set_content(
            (
                ("Main", reaction.defender_motion),
                ("Start", reaction.defender_start_motion),
            ),
            defender_candidates,
        )
        self._configuring_content = False
        self._frame = 0.0
        self._motion_loaded()

    def _set_details(self, rows: tuple[tuple[str, str], ...]) -> None:
        self.details.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.details.setItem(row, 0, QTableWidgetItem(name))
            self.details.setItem(row, 1, QTableWidgetItem(value))
        self.details.resizeRowsToContents()

    def _motion_loaded(self) -> None:
        if self._configuring_content:
            return
        reaction = self._current_reaction
        trigger = reaction.trigger_frame if reaction else 0
        self.attacker.configure_sequence(("Main",))
        # Defender Start is the approach/preparation clip that ran before the
        # synchronized interaction.  The paired preview begins when both Main
        # clips begin; prepending Start puts the two actors in different phases.
        self.defender.configure_sequence(("Main",))
        attacker_duration = self.attacker.sequence_duration
        defender_duration = self.defender.sequence_duration
        self.next_action.configure_sequence(
            (self._next_action_label,) if self._next_action_label else ()
        )
        self._next_action_start = max(attacker_duration, defender_duration)
        self.player_followup.configure_sequence(
            self._player_followup_labels
        )
        self._player_followup_start = self._next_action_start
        adjust_duration = (
            self.player_followup.sequence_segments[0].duration
            if self._multiple_parry_active
            and len(self.player_followup.sequence_segments) > 1
            else 0.0
        )
        self.multiple_enemy.configure_sequence(
            ("Nearby Enemy · Reaction",) if self._multiple_parry_active else (),
            (adjust_duration,) if self._multiple_parry_active else (),
        )
        next_duration = self.next_action.sequence_duration
        player_followup_duration = self.player_followup.sequence_duration
        multiple_enemy_duration = self.multiple_enemy.sequence_duration
        attacker_segments = [("Main", 0.0, attacker_duration)]
        defender_segments = [("Main", 0.0, defender_duration)]
        if next_duration > 0.0:
            segment = (
                self._next_action_label,
                self._next_action_start,
                self._next_action_start + next_duration,
            )
            if self._next_action_role == "attacker":
                attacker_segments.append(segment)
                attacker_duration = segment[2]
            else:
                defender_segments.append(segment)
                defender_duration = segment[2]
        if self._multiple_parry_active:
            for motion_segment in self.player_followup.sequence_segments:
                segment = (
                    motion_segment.label,
                    self._player_followup_start + motion_segment.start,
                    self._player_followup_start
                    + motion_segment.start
                    + motion_segment.duration,
                )
                if self._player_followup_role == "attacker":
                    attacker_segments.append(segment)
                    attacker_duration = max(attacker_duration, segment[2])
                else:
                    defender_segments.append(segment)
                    defender_duration = max(defender_duration, segment[2])
        elif player_followup_duration > 0.0:
            segment = (
                self._player_followup_label,
                self._player_followup_start,
                self._player_followup_start + player_followup_duration,
            )
            if self._player_followup_role == "attacker":
                attacker_segments.append(segment)
                attacker_duration = max(attacker_duration, segment[2])
            else:
                defender_segments.append(segment)
                defender_duration = max(defender_duration, segment[2])
        if multiple_enemy_duration > 0.0:
            segment = (
                "Multiple Parry · Nearby Enemy",
                self._player_followup_start + adjust_duration,
                self._player_followup_start + multiple_enemy_duration,
            )
            if self._multiple_enemy_role == "attacker":
                attacker_segments.append(segment)
                attacker_duration = max(attacker_duration, segment[2])
            else:
                defender_segments.append(segment)
                defender_duration = max(defender_duration, segment[2])
        self._attacker_start = 0.0
        self._defender_start = 0.0
        self.timeline.configure(
            self._attacker_start + attacker_duration,
            self._defender_start + defender_duration,
            trigger,
            attacker_start=self._attacker_start,
            defender_start=self._defender_start,
            attacker_segments=tuple(attacker_segments),
            defender_segments=tuple(defender_segments),
        )
        end = self.timeline.end_frame
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_spin):
            self.frame_slider.setRange(0, round(end * _SLIDER_SCALE))
            self.frame_spin.setRange(0.0, end)
        self.set_frame(min(self._frame, end))

    def set_frame(self, frame: float) -> None:
        if not math.isfinite(frame):
            return
        self._frame = min(max(0.0, float(frame)), self.timeline.end_frame)
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_spin):
            self.frame_slider.setValue(round(self._frame * _SLIDER_SCALE))
            self.frame_spin.setValue(self._frame)
        self.timeline.set_current_frame(self._frame)
        attacker_root = self.attacker.prepare_sequence_frame(
            self._frame,
            self._attacker_start,
        )
        defender_root = self.defender.prepare_sequence_frame(
            self._frame,
            self._defender_start,
        )
        reaction = self._current_reaction
        fixed_root = (
            defender_root
            if reaction is not None and reaction.fixed_object_type == 0
            else attacker_root
            if reaction is not None and reaction.fixed_object_type == 1
            else np.zeros(3, dtype=np.float32)
        )
        # FixedObjType anchors the interaction on the ground plane.  Sharing
        # its vertical root offset with the other actor makes one participant
        # rise or sink whenever the fixed actor has authored Y root motion.
        # X/Z remain a shared interaction-space anchor.  WOTS character
        # controllers keep each interaction actor on its own vertical base;
        # applying MOT root Y directly makes grapple participants float or
        # sink.  Lock Y independently for both actors instead of transferring
        # the fixed actor's vertical correction to its partner.
        planar_anchor = np.array(
            (-fixed_root[0], 0.0, -fixed_root[2]),
            dtype=np.float32,
        )
        attacker_anchor = planar_anchor.copy()
        defender_anchor = planar_anchor.copy()
        attacker_anchor[1] = -attacker_root[1]
        defender_anchor[1] = -defender_root[1]
        self.attacker.set_anchor_translation(attacker_anchor)
        self.defender.set_anchor_translation(defender_anchor)
        actor_panes = {
            "attacker": self.attacker,
            "defender": self.defender,
        }

        def activate_followup(pane, start_frame: float, role: str) -> None:
            if (
                pane is None
                or not pane.sequence_segments
                or self._frame < start_frame
            ):
                return
            followup_root = pane.prepare_sequence_frame(self._frame, start_frame)
            main_root = attacker_root if role == "attacker" else defender_root
            main_anchor = attacker_anchor if role == "attacker" else defender_anchor
            action_start = pane.sequence_segments[0].root_start
            world_start = main_root + main_anchor
            pane.set_anchor_translation(np.array((
                world_start[0] - action_start[0],
                -followup_root[1],
                world_start[2] - action_start[2],
            ), dtype=np.float32))
            actor_panes[role] = pane

        activate_followup(
            getattr(self, "next_action", None),
            self._next_action_start,
            self._next_action_role,
        )
        activate_followup(
            getattr(self, "player_followup", None),
            self._player_followup_start,
            self._player_followup_role,
        )
        if self._multiple_parry_active and self.multiple_enemy.sequence_segments:
            nearby_root = self.multiple_enemy.prepare_sequence_frame(
                self._frame,
                self._player_followup_start,
            )
            player_role = self._player_followup_role
            player_root = attacker_root if player_role == "attacker" else defender_root
            player_anchor = (
                attacker_anchor if player_role == "attacker" else defender_anchor
            )
            target = player_root + player_anchor + self._multiple_target_offset
            self.multiple_enemy.set_anchor_translation(np.array((
                target[0] - nearby_root[0],
                -nearby_root[1],
                target[2] - nearby_root[2],
            ), dtype=np.float32))
            self.multiple_enemy.render_prepared_frame()
        actor_panes["attacker"].render_prepared_frame()
        actor_panes["defender"].render_prepared_frame()

    def toggle_playback(self) -> None:
        if self._playing:
            self.stop_playback()
            return
        if self._frame >= self.timeline.end_frame:
            self.set_frame(0.0)
        self._playing = True
        self._elapsed.start()
        self._timer.start()
        self.play_button.setText(self.tr("Pause"))

    def stop_playback(self) -> None:
        self._playing = False
        self._timer.stop()
        self.play_button.setText(self.tr("Play"))

    def _tick(self) -> None:
        if not self._playing:
            return
        elapsed = min(max(0, self._elapsed.restart()), 250) / 1000.0
        next_frame = self._frame + elapsed * 60.0
        if next_frame >= self.timeline.end_frame:
            self.set_frame(self.timeline.end_frame)
            self.stop_playback()
        else:
            self.set_frame(next_frame)

    def cleanup(self) -> None:
        self.stop_playback()
        next_pane = getattr(self, "next_action", None)
        if next_pane is not None:
            next_pane.cleanup()
        player_followup = getattr(self, "player_followup", None)
        if player_followup is not None:
            player_followup.cleanup()
        multiple_enemy = getattr(self, "multiple_enemy", None)
        if multiple_enemy is not None:
            multiple_enemy.cleanup()
        self.attacker.cleanup()
        self.defender.cleanup()
        self.viewport.cleanup()

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)


class WotsIssenPreviewWidget(WotsInteractionPreviewWidget):
    """Synchronized player/enemy preview for WOTS grapple motion tables."""

    def __init__(self, handler, document: IssenDocument, parent=None):
        QWidget.__init__(self, parent)
        self.handler = handler
        self.document = document
        self._current_reaction: IssenPattern | None = None
        self._initial_reaction_item: QTreeWidgetItem | None = None
        self._activated = False
        self._frame = 0.0
        self._attacker_start = 0.0
        self._defender_start = 0.0
        self._playing = False
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._mex_maps: dict[str, tuple[dict[int, int], str]] = {}
        self._resolution_messages: list[str] = []
        self._player_labels: tuple[str, ...] = ()
        self._player_transition_labels: tuple[str, ...] = ()
        self._enemy_labels: tuple[str, ...] = ()
        self._enemy2_labels: tuple[str, ...] = ()
        self._combo_target_count = 1
        self._combo_player_phase_counts = (0, 0)
        self._combo_first_end = 0.0
        self._combo_second_start = 0.0
        self._combo_first_target = np.zeros(3, dtype=np.float32)
        self._combo_second_target = np.zeros(3, dtype=np.float32)
        self._combo_transition_transform = np.identity(4, dtype=np.float32)
        self._configuring_content = False
        bank_ids = {
            int(phase.reference.bank_id)
            for pattern in document.patterns
            for phase in (*pattern.player_phases, *pattern.enemy_phases)
            if phase.reference.bank_id is not None
        }
        self._bank_candidates = discover_motion_bank_candidates(
            document.source_path,
            bank_ids,
            getattr(handler, "resource_context", None),
        )
        lowered_source = document.source_path.replace("\\", "/").casefold()
        is_issen = "issen" in lowered_source
        self.preview_tab_title = "Issen Preview" if is_issen else "Grapple Preview"
        self.preview_description = (
            "Read-only WOTS grapple preview. Player and enemy phases are resolved "
            "from GrappleMotionTable, adjacent MEX files, and verified WOTS "
            "action GUID links; runtime state transitions are simulated."
        )
        self._build_ui()
        actors = self.attacker.parentWidget()
        self.enemy2 = _ActorMotionPane(
            self.handler,
            "Enemy 2",
            self._scene_host.viewport_factory("enemy2"),
            parent=actors,
        )
        self.enemy2.motion_loaded.connect(self._motion_loaded)
        # The transition lives in the global plw_Issen MOTLIST rather than the
        # enemy-specific paired-motion bank.  It renders through the same
        # actor namespace as the normal player pane, so only one player model
        # is present in the shared scene.
        self.player_transition = _ActorMotionPane(
            self.handler,
            "Player Transition",
            self._scene_host.viewport_factory("attacker"),
            parent=actors,
        )
        self.player_transition.motion_loaded.connect(self._motion_loaded)
        self.player_transition.hide()
        if isinstance(actors, QSplitter):
            actors.addWidget(self.enemy2)
            actors.setSizes([700, 700, 700])
        self.enemy2.hide()
        pattern_header = "Issen Pattern" if is_issen else "Grapple Pattern"
        self.reaction_tree.setHeaderLabels((self.tr(pattern_header), self.tr("Value")))
        self._populate_reactions()

    def _populate_reactions(self) -> None:
        combo_patterns = _break_issen_combo_patterns(self.document)
        combo_start_index = combo_patterns[0].index if len(combo_patterns) >= 2 else -1
        for index, pattern in enumerate(self.document.patterns):
            item = QTreeWidgetItem((
                pattern.title,
                f"{len(pattern.player_phases)} player / "
                f"{len(pattern.enemy_phases)} enemy phases",
            ))
            item.setData(0, _ROLE, index)
            item.setData(0, _TARGET_COUNT_ROLE, 1)
            self.reaction_tree.addTopLevelItem(item)
            if index == combo_start_index:
                combo = QTreeWidgetItem((
                    "Break Issen Combo · 2 Targets",
                    "COMMON_PAT_10 → COMMON_PAT_11",
                ))
                combo.setData(0, _ROLE, index)
                combo.setData(0, _TARGET_COUNT_ROLE, 2)
                item.addChild(combo)
                item.setExpanded(True)
        if self.reaction_tree.topLevelItemCount():
            self._initial_reaction_item = self.reaction_tree.topLevelItem(0)

    def _candidate_map(self, candidate: MotionBankCandidate) -> dict[int, int]:
        key = candidate.resource_path.casefold()
        cached = self._mex_maps.get(key)
        if cached is None:
            cached = load_wots_mex_motion_map(self.handler, candidate, self)
            self._mex_maps[key] = cached
        mapping, message = cached
        if message and message not in self._resolution_messages:
            self._resolution_messages.append(message)
        return mapping

    def _resolve_phase(self, phase: IssenMotionPhase, role: str) -> IssenMotionPhase:
        reference = phase.reference
        if reference.motion_id is not None or reference.bank_id is None:
            return phase
        candidates = self._bank_candidates.get(int(reference.bank_id), ())
        preferred = preferred_motion_bank_candidate(candidates, role)
        ordered = tuple(
            item
            for item in (preferred, *candidates)
            if item is not None
        )
        seen = set()
        for candidate in ordered:
            key = candidate.resource_path.casefold()
            if key in seen:
                continue
            seen.add(key)
            motion_id = self._candidate_map(candidate).get(phase.motion_group_id)
            if motion_id is None:
                continue
            return replace(
                phase,
                reference=replace(reference, motion_id=int(motion_id)),
            )
        message = (
            f"Unresolved {role} MotionGroupID "
            f"0x{phase.motion_group_id:016X} ({phase.action_name})."
        )
        if message not in self._resolution_messages:
            self._resolution_messages.append(message)
        return phase

    def _pattern_rows(
        self,
        pattern: IssenPattern,
        player: tuple[IssenMotionPhase, ...],
        enemy: tuple[IssenMotionPhase, ...],
    ) -> tuple[tuple[str, str], ...]:
        rows = [
            ("Evidence", "GrappleMotionTable + adjacent MEX; runtime transitions are simulated"),
            ("Pattern", pattern.title),
            ("GrappleType", str(pattern.grapple_type)),
            ("Pattern Value", f"{pattern.pattern_value} (0x{pattern.pattern_value & 0xFFFFFFFF:08X})"),
        ]
        for role, phases in (("Player", player), ("Enemy", enemy)):
            for index, phase in enumerate(phases, 1):
                rows.extend((
                    (f"{role} Phase {index}", phase.action_name),
                    (f"{role} Phase {index} ActionGuid", phase.action_guid or "None"),
                    (
                        f"{role} Phase {index} MotionGroupID",
                        (
                            "Action GUID link"
                            if phase.motion_group_id == _INVALID_MOTION_GROUP_ID
                            else f"0x{phase.motion_group_id:016X}"
                        ),
                    ),
                    (f"{role} Phase {index} Motion", phase.reference.label),
                ))
        rows.extend(("Diagnostic", value) for value in self._resolution_messages)
        return tuple(rows)

    def _candidate_pool(
        self,
        phases: tuple[IssenMotionPhase, ...],
    ) -> tuple[MotionBankCandidate, ...]:
        result = []
        for phase in phases:
            bank_id = phase.reference.bank_id
            if bank_id is None:
                continue
            for candidate in self._bank_candidates.get(int(bank_id), ()):
                if candidate not in result:
                    result.append(candidate)
        return tuple(result)

    def _on_reaction_changed(self, current, _previous) -> None:
        index = current.data(0, _ROLE) if current is not None else None
        if not isinstance(index, int) or not (0 <= index < len(self.document.patterns)):
            return
        self.stop_playback()
        self._combo_target_count = int(
            current.data(0, _TARGET_COUNT_ROLE) or 1
        )
        self._resolution_messages.clear()
        pattern = self.document.patterns[index]
        player = tuple(self._resolve_phase(phase, "Player") for phase in pattern.player_phases)
        enemy = tuple(self._resolve_phase(phase, "Enemy") for phase in pattern.enemy_phases)
        followup_pattern = None
        followup_player: tuple[IssenMotionPhase, ...] = ()
        followup_enemy: tuple[IssenMotionPhase, ...] = ()
        self._current_reaction = pattern
        if self._combo_target_count > 1:
            combo_patterns = _break_issen_combo_patterns(self.document)
            combo_index = next(
                (
                    item_index
                    for item_index, item in enumerate(combo_patterns)
                    if item.index == pattern.index
                ),
                -1,
            )
            if not 0 <= combo_index < len(combo_patterns) - 1:
                self._combo_target_count = 1
                self._on_reaction_changed(
                    self.reaction_tree.topLevelItem(index),
                    current,
                )
                return
            followup_pattern = combo_patterns[combo_index + 1]
            followup_player = tuple(
                self._resolve_phase(phase, "Player Target 2")
                for phase in followup_pattern.player_phases
            )
            followup_enemy = tuple(
                self._resolve_phase(phase, "Enemy Target 2")
                for phase in followup_pattern.enemy_phases
            )
            transition = _break_issen_next_turn_phase(1, "F")
            player_refs = tuple(
                (
                    f"Target 1 · {phase_index}. {phase.action_name}",
                    phase.reference,
                )
                for phase_index, phase in enumerate(player, 1)
            ) + tuple(
                (
                    f"Target 2 · {phase_index}. {phase.action_name}",
                    phase.reference,
                )
                for phase_index, phase in enumerate(followup_player, 1)
            )
            enemy_refs = tuple(
                (f"Target 1 · {phase_index}. {phase.action_name}", phase.reference)
                for phase_index, phase in enumerate(enemy, 1)
            )
            enemy2_refs = tuple(
                (f"Target 2 · {phase_index}. {phase.action_name}", phase.reference)
                for phase_index, phase in enumerate(followup_enemy, 1)
            )
            transition_refs = ((transition.action_name, transition.reference),)
            self._combo_player_phase_counts = (len(player), len(followup_player))
        else:
            player_refs = tuple(
                (f"{phase_index}. {phase.action_name}", phase.reference)
                for phase_index, phase in enumerate(player, 1)
            )
            enemy_refs = tuple(
                (f"{phase_index}. {phase.action_name}", phase.reference)
                for phase_index, phase in enumerate(enemy, 1)
            )
            enemy2_refs = ()
            transition_refs = ()
            self._combo_player_phase_counts = (0, 0)
        self._player_labels = tuple(label for label, _ref in player_refs)
        self._player_transition_labels = tuple(
            label for label, _ref in transition_refs
        )
        self._enemy_labels = tuple(label for label, _ref in enemy_refs)
        self._enemy2_labels = tuple(label for label, _ref in enemy2_refs)
        rows = list(self._pattern_rows(pattern, player, enemy))
        if self._combo_target_count > 1 and followup_pattern is not None:
            rows.extend((
                ("Preview Mode", "Break Issen Combo · 2 Targets"),
                ("Entry", "Parry / synchronized Grapple"),
                (
                    "Target 1 Pattern",
                    f"COMMON_PAT_10 · table index {pattern.index} · motions "
                    + "/".join(
                        str(phase.reference.motion_id)
                        for phase in player
                        if phase.reference.motion_id is not None
                    ),
                ),
                (
                    "Target 2 Pattern",
                    f"COMMON_PAT_11 · table index {followup_pattern.index} · motions "
                    + "/".join(
                        str(phase.reference.motion_id)
                        for phase in followup_player
                        if phase.reference.motion_id is not None
                    ),
                ),
                (
                    "Target Select Window",
                    f"up to {_BREAK_ISSEN_COMBO_DEFAULT_SELECT_FRAMES:.0f} frames (input limit; not playback duration)",
                ),
                (
                    "Target 2 Transition",
                    "NextTurn · ChainBreakIssen_1st_Move_F_Start · Bank 20031 / Motion 533",
                ),
            ))
        self._set_details(tuple(rows))
        self._configuring_content = True
        self.player_transition.set_content(
            transition_refs,
            (
                MotionBankCandidate(
                    _BREAK_ISSEN_MOTION_BANK_ID,
                    _BREAK_ISSEN_MOTLIST_PATH,
                ),
            ) if transition_refs else (),
        )
        self.attacker.set_content(
            player_refs,
            self._candidate_pool((*player, *followup_player)),
        )
        self.defender.set_content(enemy_refs, self._candidate_pool(enemy))
        self.enemy2.set_content(
            enemy2_refs,
            self._candidate_pool(followup_enemy),
        )
        self.enemy2.setVisible(self._combo_target_count > 1)
        self._configuring_content = False
        self._frame = 0.0
        self._motion_loaded()

    def _motion_loaded(self) -> None:
        if self._configuring_content:
            return
        if self._combo_target_count > 1:
            self._configure_break_issen_combo()
            return
        self.timeline.set_role_labels(
            self.document.attacker_role,
            self.document.defender_role,
        )
        self.attacker.configure_sequence(self._player_labels)
        self.defender.configure_sequence(self._enemy_labels)
        player_segments = self.attacker.sequence_segments
        enemy_segments = self.defender.sequence_segments
        player_starts = tuple(segment.start for segment in player_segments)
        enemy_starts: tuple[float, ...] = ()
        pattern = self._current_reaction
        if pattern is not None and enemy_segments:
            lowered_title = pattern.title.casefold()
            if (
                any(kind in lowered_title for kind in ("counter issen", "block issen"))
                and len(player_starts) > 1
            ):
                enemy_starts = player_starts[1: 1 + len(enemy_segments)]
            elif (
                "chain issen" in lowered_title
                and "fully" not in lowered_title
                and len(player_starts) >= 3
                and len(enemy_segments) == 2
            ):
                enemy_starts = (player_starts[0], player_starts[2])
            elif len(player_starts) > 1 and len(enemy_segments) == 1:
                # Blow/Fatal tables commonly encode an owner Adjust phase
                # followed by the synchronized Const phase, while the partner
                # only owns the Const motion.
                enemy_starts = (player_starts[1],)
            else:
                enemy_starts = player_starts[:len(enemy_segments)]
        if enemy_starts:
            self.defender.configure_sequence(self._enemy_labels, enemy_starts)
        attacker_duration = self.attacker.sequence_duration
        defender_duration = self.defender.sequence_duration
        attacker_segments = tuple(
            (segment.label, segment.start, segment.start + segment.duration)
            for segment in self.attacker.sequence_segments
        )
        defender_segments = tuple(
            (segment.label, segment.start, segment.start + segment.duration)
            for segment in self.defender.sequence_segments
        )
        self._attacker_start = 0.0
        self._defender_start = 0.0
        defender_display_start = min(enemy_starts, default=0.0)
        self.timeline.configure(
            attacker_duration,
            defender_duration,
            0,
            defender_start=defender_display_start,
            attacker_segments=attacker_segments,
            defender_segments=defender_segments,
        )
        end = self.timeline.end_frame
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_spin):
            self.frame_slider.setRange(0, round(end * _SLIDER_SCALE))
            self.frame_spin.setRange(0.0, end)
        self.set_frame(min(self._frame, end))

    def _configure_break_issen_combo(self) -> None:
        """Build a two-target sequence with the authored first NextTurn clip."""
        self.timeline.set_role_labels("Player", "Enemy 1 / Enemy 2")
        self.attacker.configure_sequence(
            self._player_labels,
            preserve_root_continuity=False,
        )
        self.player_transition.configure_sequence(
            self._player_transition_labels,
            preserve_root_continuity=False,
        )
        initial_segments = self.attacker.sequence_segments
        first_phase_count, second_phase_count = self._combo_player_phase_counts
        if (
            first_phase_count <= 0
            or second_phase_count <= 0
            or len(initial_segments) < first_phase_count + second_phase_count
        ):
            return

        first_durations = tuple(
            segment.duration for segment in initial_segments[:first_phase_count]
        )
        first_starts = []
        cursor = 0.0
        for duration in first_durations:
            first_starts.append(cursor)
            cursor += duration
        self._combo_first_end = cursor
        transition_segments = self.player_transition.sequence_segments
        transition_duration = sum(
            segment.duration for segment in transition_segments
        )
        self._combo_second_start = self._combo_first_end + transition_duration
        second_starts = []
        cursor = self._combo_second_start
        for segment in initial_segments[
            first_phase_count:first_phase_count + second_phase_count
        ]:
            second_starts.append(cursor)
            cursor += segment.duration
        player_starts = tuple(first_starts) + tuple(second_starts)
        self.attacker.configure_sequence(
            self._player_labels,
            player_starts,
            preserve_root_continuity=False,
        )

        player_segments = self.attacker.sequence_segments
        first_const_start = player_segments[first_phase_count - 1].start
        second_const_start = player_segments[-1].start
        self.defender.configure_sequence(
            self._enemy_labels,
            tuple(first_const_start for _label in self._enemy_labels),
        )
        self.enemy2.configure_sequence(
            self._enemy2_labels,
            tuple(second_const_start for _label in self._enemy2_labels),
        )
        enemy1_segments = self.defender.sequence_segments
        enemy2_segments = self.enemy2.sequence_segments
        self._combo_first_target = np.array((-1.3, 0.0, 0.0), dtype=np.float32)
        self._combo_second_target = np.array((1.3, 0.0, 0.0), dtype=np.float32)
        self._combo_transition_transform = np.identity(4, dtype=np.float32)
        if enemy1_segments and enemy2_segments and transition_segments:
            first_player_end = player_segments[first_phase_count - 1].root_end
            second_player_start = player_segments[first_phase_count].root_start
            first_enemy_end = enemy1_segments[-1].root_end
            second_enemy_start = enemy2_segments[0].root_start
            route_start = (
                self._combo_first_target + first_player_end - first_enemy_end
            )
            route_start[1] = 0.0

            transition = transition_segments[0]
            delta = transition.root_end - transition.root_start
            horizontal_length = math.hypot(float(delta[0]), float(delta[2]))
            if horizontal_length > 1e-5:
                # The preview lays targets out left-to-right. Rotate the
                # authored local root path onto that world-space route without
                # scaling it, preserving the clip's actual travel distance.
                source_angle = math.atan2(float(delta[0]), float(delta[2]))
                yaw = (math.pi * 0.5) - source_angle
                cosine = math.cos(yaw)
                sine = math.sin(yaw)
                transform = np.identity(4, dtype=np.float32)
                transform[0, 0] = cosine
                transform[0, 2] = -sine
                transform[2, 0] = sine
                transform[2, 2] = cosine
                transformed_start = transition.root_start @ transform[:3, :3]
                transform[3, :3] = route_start - transformed_start
                self._combo_transition_transform = transform
                route_end = (
                    transition.root_end @ transform[:3, :3]
                    + transform[3, :3]
                )
                self._combo_second_target = (
                    route_end - second_player_start + second_enemy_start
                )
                self._combo_second_target[1] = 0.0

        player_timeline = [
            (segment.label, segment.start, segment.start + segment.duration)
            for segment in player_segments[:first_phase_count]
        ]
        player_timeline.extend(
            (
                f"Next Turn · {segment.label}",
                self._combo_first_end + segment.start,
                self._combo_first_end + segment.start + segment.duration,
            )
            for segment in transition_segments
        )
        player_timeline.extend(
            (segment.label, segment.start, segment.start + segment.duration)
            for segment in player_segments[first_phase_count:]
        )
        enemy_timeline = tuple(
            (segment.label, segment.start, segment.start + segment.duration)
            for segment in (
                *self.defender.sequence_segments,
                *self.enemy2.sequence_segments,
            )
        )
        end = max(
            self.attacker.sequence_duration,
            self.defender.sequence_duration,
            self.enemy2.sequence_duration,
            1.0,
        )
        self.timeline.configure(
            end,
            end,
            0,
            attacker_segments=tuple(player_timeline),
            defender_segments=enemy_timeline,
        )
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_spin):
            self.frame_slider.setRange(0, round(end * _SLIDER_SCALE))
            self.frame_spin.setRange(0.0, end)
        self.set_frame(min(self._frame, end))

    def set_frame(self, frame: float) -> None:
        if self._combo_target_count <= 1:
            super().set_frame(frame)
            return
        if not math.isfinite(frame):
            return
        self._frame = min(max(0.0, float(frame)), self.timeline.end_frame)
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_spin):
            self.frame_slider.setValue(round(self._frame * _SLIDER_SCALE))
            self.frame_spin.setValue(self._frame)
        self.timeline.set_current_frame(self._frame)

        player_root = self.attacker.prepare_sequence_frame(self._frame, 0.0)
        enemy1_root = self.defender.prepare_sequence_frame(self._frame, 0.0)
        enemy2_root = self.enemy2.prepare_sequence_frame(self._frame, 0.0)
        first_target = self._combo_first_target
        second_target = self._combo_second_target
        if self._frame <= self._combo_first_end:
            player_anchor_root = enemy1_root
            player_target = first_target
        elif self._frame >= self._combo_second_start:
            player_anchor_root = enemy2_root
            player_target = second_target
        else:
            self.player_transition.prepare_sequence_frame(
                self._frame,
                self._combo_first_end,
            )
            self.player_transition.set_world_transform(
                self._combo_transition_transform
            )
            player_target = None
            player_anchor_root = None

        def grounded_anchor(
            target: np.ndarray,
            fixed_root: np.ndarray,
            actor_root: np.ndarray,
        ) -> np.ndarray:
            # Match the normal paired preview: the enemy is the planar anchor,
            # preserving the player/enemy authored X/Z root offset.  Vertical
            # roots remain grounded independently so neither actor floats.
            result = target.copy()
            result[0] -= fixed_root[0]
            result[2] -= fixed_root[2]
            result[1] = -actor_root[1]
            return result

        if player_target is not None and player_anchor_root is not None:
            self.attacker.set_anchor_translation(
                grounded_anchor(player_target, player_anchor_root, player_root)
            )
        self.defender.set_anchor_translation(
            grounded_anchor(first_target, enemy1_root, enemy1_root)
        )
        self.enemy2.set_anchor_translation(
            grounded_anchor(second_target, enemy2_root, enemy2_root)
        )
        if player_target is None:
            self.player_transition.render_prepared_frame()
        else:
            self.attacker.render_prepared_frame()
        self.defender.render_prepared_frame()
        self.enemy2.render_prepared_frame()

    def cleanup(self) -> None:
        self.player_transition.cleanup()
        self.enemy2.cleanup()
        super().cleanup()


def create_wots_interaction_preview(handler):
    rsz = getattr(handler, "rsz_file", None)
    if rsz is None or _root_type_name(rsz) != "app.user_data.JustGuardConditionMap":
        return None
    document = parse_wots_interaction_document(
        rsz,
        str(getattr(handler, "filepath", "") or ""),
    )
    if not document.reactions:
        return None
    document = resolve_wots_justguard_next_actions(handler, document)
    return WotsInteractionPreviewWidget(handler, document)


def create_wots_issen_preview(handler):
    rsz = getattr(handler, "rsz_file", None)
    source_path = str(getattr(handler, "filepath", "") or "")
    lowered = source_path.replace("\\", "/").casefold()
    if (
        rsz is None
        or _root_type_name(rsz) != "app.user_data.GrappleMotionTable"
        or not any(kind in lowered for kind in (
            "blockissen",
            "counterissen",
            "chainissen",
            "blowgrap",
            "fatalblow",
        ))
    ):
        return None
    document = parse_wots_issen_document(rsz, source_path)
    if not document.patterns:
        return None
    document = resolve_wots_grapple_action_phases(handler, document)
    return WotsIssenPreviewWidget(handler, document)
