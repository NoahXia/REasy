from __future__ import annotations

"""Read-only WOTS attack-parameter lookup used by the motion preview."""

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Iterable

from file_handlers.rsz.rsz_file import RszFile
from utils.enum_manager import EnumManager


_ATTACK_PARAM_TYPES = {
    "app.cAttackParamDataEnemy": "cdee5bcd",
    "app.cAttackParamDataPlayer": "b25b515f",
}


def _float_field(name: str) -> dict:
    return {
        "align": 4,
        "array": False,
        "name": name,
        "native": False,
        "original_type": "System.Single",
        "size": 4,
        "type": "F32",
    }


class _AttackParameterRegistry:
    """Registry overlay for fields missing from the current WOTS dump.

    The application-wide registry is intentionally not mutated: these layout
    corrections are only required when reading the two WOTS attack tables.
    """

    def __init__(self, base):
        self.base = base
        self.json_path = getattr(base, "json_path", "")
        self._patched_by_id: dict[int, dict] = {}
        self._patched_by_name: dict[str, tuple[dict, int]] = {}
        for name, crc in _ATTACK_PARAM_TYPES.items():
            info, type_id = base.find_type_by_name(name)
            if info is None or type_id is None:
                continue
            patched = deepcopy(info)
            fields = list(patched.get("fields", ()))
            self._insert_after(fields, "_FireDamage", _float_field("_MiasmaDamage"))
            self._insert_after(
                fields,
                "_MiasmaDamage",
                _float_field("_MiasmaOutsideDamage"),
            )
            self._insert_after(fields, "_MiasmaOutsideDamage", _float_field("_OilDamage"))
            self._insert_after(
                fields,
                "_OniEnergyAttack",
                _float_field("_OniChangeEnergyAttack"),
            )
            if name.endswith("Player"):
                self._insert_after(
                    fields,
                    "_AddSoulBoostGauge",
                    _float_field("_AddSkill5Gauge"),
                )
            patched["fields"] = fields
            patched["crc"] = crc
            self._patched_by_id[int(type_id)] = patched
            self._patched_by_name[name] = (patched, int(type_id))

    @staticmethod
    def _insert_after(fields: list[dict], anchor: str, field: dict) -> None:
        if any(item.get("name") == field["name"] for item in fields):
            return
        for index, item in enumerate(fields):
            if item.get("name") == anchor:
                fields.insert(index + 1, field)
                return
        raise ValueError(f"WOTS attack-parameter registry is missing {anchor}")

    def get_type_info(self, type_id: int):
        return self._patched_by_id.get(int(type_id)) or self.base.get_type_info(
            type_id
        )

    def find_type_by_name(self, type_name: str):
        return self._patched_by_name.get(type_name) or self.base.find_type_by_name(
            type_name
        )

    def pre_cache_types(self, type_ids) -> None:
        pre_cache = getattr(self.base, "pre_cache_types", None)
        if callable(pre_cache):
            pre_cache(type_ids)


@dataclass(frozen=True, slots=True)
class AttackParameterRecord:
    attack_param_id: int
    instance_id: int
    fields: dict


@dataclass(frozen=True, slots=True)
class AttackParameterSource:
    resource_path: str
    rsz: RszFile
    records: dict[int, tuple[AttackParameterRecord, ...]]

    def record(self, attack_param_id: int) -> AttackParameterRecord | None:
        matches = self.records.get(int(attack_param_id), ())
        return matches[0] if matches else None

    def record_count(self, attack_param_id: int) -> int:
        return len(self.records.get(int(attack_param_id), ()))


def attack_parameter_resource_candidates(anchor_path: str) -> tuple[str, ...]:
    normalized = str(anchor_path or "").replace("\\", "/")
    lowered = f"/{normalized.casefold()}"
    if "/motion/player/" in lowered:
        return (
            "natives/stm/GameDesign/Action/Player/Data/ParamPack/"
            "PlayerAttackParamList.user.3",
        )
    match = re.search(r"(?:^|/)motion/enemy/(em\d+)/(\d{2})(?:/|$)", lowered)
    if match is None:
        return ()
    actor = match.group(1)
    variant = match.group(2)
    display_actor = actor[:1].upper() + actor[1:]
    return (
        "natives/stm/GameDesign/Action/Enemy/"
        f"{display_actor}/{variant}/Data/Param/"
        f"{display_actor}_{variant}_AttackParam.user.3",
    )


def load_attack_parameter_source(
    anchor_path: str,
    resource_loader,
    *,
    type_registry=None,
) -> tuple[AttackParameterSource | None, tuple[str, ...]]:
    candidates = attack_parameter_resource_candidates(anchor_path)
    if not candidates:
        return None, (
            "attack parameter resource could not be inferred from the MOTLIST path",
        )
    if type_registry is None:
        return None, ("WOTS RSZ type registry is unavailable",)
    diagnostics: list[str] = []
    for candidate in candidates:
        hit = resource_loader(candidate)
        if hit is None:
            diagnostics.append(f"attack parameter resource was not found: {candidate}")
            continue
        resolved_path, data = hit
        try:
            rsz = RszFile()
            rsz.filepath = resolved_path
            rsz.game_version = "OnimushaWOTS"
            rsz.type_registry = _AttackParameterRegistry(type_registry)
            rsz.read(data, validate_type_registry=True)
            records = _index_attack_parameters(rsz)
            if not records:
                raise ValueError("no AttackParamListItem records were found")
            return AttackParameterSource(resolved_path, rsz, records), tuple(diagnostics)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            diagnostics.append(
                f"could not parse attack parameter resource {resolved_path!r}: {exc}"
            )
    return None, tuple(diagnostics)


def _index_attack_parameters(rsz: RszFile) -> dict[int, tuple[AttackParameterRecord, ...]]:
    grouped: dict[int, list[AttackParameterRecord]] = {}
    for instance_id, info in enumerate(rsz.instance_infos):
        type_info = rsz.type_registry.get_type_info(info.type_id)
        if not type_info or type_info.get("name") != (
            "app.user_data.AttackParamListUserData.AttackParamListItem"
        ):
            continue
        item = rsz.parsed_elements.get(instance_id, {})
        attack_id = _scalar(item.get("_ID"))
        target_id = _scalar(item.get("_AttackParam"))
        if not isinstance(attack_id, int) or not isinstance(target_id, int):
            continue
        fields = rsz.parsed_elements.get(target_id)
        if not isinstance(fields, dict):
            continue
        grouped.setdefault(attack_id, []).append(
            AttackParameterRecord(attack_id, target_id, fields)
        )
    return {key: tuple(value) for key, value in grouped.items()}


def attack_parameter_detail_sections(
    source: AttackParameterSource | None,
    attack_param_id: int,
    diagnostics: Iterable[str] = (),
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Return compact, human-readable inspector sections."""
    if source is None:
        message = "; ".join(diagnostics) or "attack parameter resource not resolved"
        return (("ATTACK PARAMETERS", (("Status", message),)),)
    record = source.record(attack_param_id)
    if record is None:
        return (
            (
                "ATTACK PARAMETERS",
                (
                    ("Resource", source.resource_path),
                    ("AttackParamID", str(attack_param_id)),
                    (
                        "Status",
                        "ID was not found in the resolved attack parameter table",
                    ),
                ),
            ),
        )

    fields = record.fields
    count = source.record_count(attack_param_id)
    identity_rows = [
        ("Resource", source.resource_path),
        ("AttackParamID", str(attack_param_id)),
        ("Parameter Instance", str(record.instance_id)),
    ]
    if count > 1:
        identity_rows.append(("Matching Records", f"{count} (first record shown)"))

    damage_rows = _present_fields(
        source,
        fields,
        (
            ("Attack Damage", "_Attack", None),
            ("Rikido Damage", "_RikidoAttack", None),
            ("Guard Damage", "_GuardAttack", None),
            ("Armor Damage", "_ArmorAttack", None),
            ("Oni Energy Damage", "_OniEnergyAttack", None),
            ("Oni Change Energy Damage", "_OniChangeEnergyAttack", None),
            ("Fire Damage", "_FireDamage", None),
            ("Miasma Damage", "_MiasmaDamage", None),
            ("Miasma Outside Damage", "_MiasmaOutsideDamage", None),
            ("Oil Damage", "_OilDamage", None),
            ("Attack Form", "_FormType", "enum"),
            ("Damage Category", "_DamageCategory", "enum"),
            ("Element", "_Element", "enum_ref"),
            ("Destruction Damage", "_DestructionDamage", "values"),
        ),
    )
    reaction_rows = _present_fields(
        source,
        fields,
        (
            ("Damage Reaction", "_DamageReactionType", "enum_ref"),
            ("Damage Reaction Level", "_DamageReactionLv", "enum_ref"),
            ("Guard Reaction Level", "_GuardReactionLv", "enum_ref"),
            ("Just Guard Data ID", "_JusdGuardDataID", None),
            ("Dodge Type", "_DodgeType", "enum_ref"),
            ("Dodge Move Type", "_DodgeMoveType", "enum_ref"),
            ("Dodge Translation Rate", "_DodgeTranslateRate", None),
            ("Dodge Move Translation Rate", "_DodgeMoveTranslateRate", None),
            ("Blocked Coefficient", "_BlockedCoeffType", "enum_ref"),
            ("Parried Coefficient", "_ParriedCoeffType", "enum_ref"),
            ("Just-Dodged Coefficient", "_JustDodgedCoeffType", "enum_ref"),
        ),
    )

    flag_value = _scalar(fields.get("_FlagBit"))
    rule_rows: list[tuple[str, str]] = []
    if isinstance(flag_value, int):
        flags = _flag_names("app.cAttackParamDataBase.FLAG_BIT", flag_value)
        rule_rows.extend((
            ("Flags", flags),
            ("Guard", "Disabled" if flag_value & 0x4 else "Allowed"),
            ("Parry", "Disabled" if flag_value & 0x8 else "Allowed"),
            ("Block", "Disabled" if flag_value & 0x10 else "Allowed"),
            ("Counter Issen", "Disabled" if flag_value & 0x20 else "Allowed"),
            ("Just Dodge", "Disabled" if flag_value & 0x4000 else "Allowed"),
            ("Dodge", "Disabled" if flag_value & 0x200000 else "Allowed"),
            (
                "Just Guard",
                "Blow reaction enabled"
                if flag_value & 0x4000000000
                else "Standard rules",
            ),
        ))
    multi_count = _scalar(fields.get("_MultiHitMaxCount"))
    if isinstance(flag_value, int) or isinstance(multi_count, int):
        rule_rows.append((
            "Multi-Hit Enabled",
            "Yes"
            if (isinstance(flag_value, int) and flag_value & 1)
            or (isinstance(multi_count, int) and multi_count > 0)
            else "No",
        ))
    rule_rows.extend(_present_fields(
        source,
        fields,
        (
            ("Multi-Hit Interval", "_MultiHitTimer", None),
            ("Multi-Hit Max Count", "_MultiHitMaxCount", None),
            ("Player Flags", "_PlayerFlagBit", "player_flags"),
        ),
    ))

    knockback_rows = _present_fields(
        source,
        fields,
        (
            ("Damage Receiver Knockback", "_IsKnockbackDamageReceiver", None),
            ("Direction Type", "_KnockbackDirType", "enum"),
            ("Reverse Type", "_KnockbackRevType", "enum"),
            ("Direction", "_KnockbackDirection", None),
            ("Limit Enabled", "_KnockbackLimitEnable", None),
            ("Limit Reference", "_KnockbackLimitType", "enum"),
            ("Limit Range", "_KnockbackLimitRange", None),
            ("Overwrite Count", "_KnockbackOverwriteArray", "count"),
        ),
    )
    attacker_rate = _referenced_fields(source.rsz, fields.get("_KnockbackAttackerRate"))
    if attacker_rate:
        knockback_rows += _present_fields(
            source,
            attacker_rate,
            (
                ("Use Z Speed Rate", "IsUseRateZ", None),
                ("Z Speed Rate", "ZSpeedRate", None),
                ("Z Deceleration Rate", "ZDecelerateRate", None),
                ("Use Y Speed Rate", "IsUseRateY", None),
                ("Y Speed Rate", "YSpeedRate", None),
                ("Y Gravity Rate", "YGravityRate", None),
                ("Use Z Motion Rate", "IsUseMotRateZ", None),
                ("Z Motion Rate", "ZMotionRate", None),
                ("Use Y Motion Rate", "IsUseMotRateY", None),
                ("Y Motion Rate", "YMotionRate", None),
            ),
        )
    hit_stop_rows = _present_fields(
        source,
        fields,
        (
            ("HitStop ID", "_HitStopID", None),
            ("HitStop Delay", "_HitStopDelayFrame", "frames"),
            ("Guard HitStop Delay", "_HitStopDelayFrameGuard", "frames"),
        ),
    )

    sections = [
        ("ATTACK PARAMETERS", tuple(identity_rows)),
        ("DAMAGE / STAGGER", damage_rows),
        ("HIT / DEFENSE RULES", tuple(rule_rows + list(reaction_rows))),
        ("KNOCKBACK", knockback_rows),
    ]
    if hit_stop_rows:
        sections.append(("HIT STOP", hit_stop_rows))
    return tuple((title, rows) for title, rows in sections if rows)


def _present_fields(source, fields, specifications):
    rows = []
    for label, field_name, mode in specifications:
        data = fields.get(field_name)
        if data is None:
            continue
        if mode == "enum":
            value = _enum_value(data)
        elif mode == "enum_ref":
            value = _enum_reference_value(source.rsz, data)
        elif mode == "player_flags":
            raw = _scalar(data)
            value = (
                _flag_names("app.cAttackParamDataPlayer.PLAYER_FLAG_BIT", raw)
                if isinstance(raw, int)
                else _format_scalar(raw)
            )
        elif mode == "values":
            value = ", ".join(
                _format_scalar(_scalar(item))
                for item in getattr(data, "values", ())
            ) or "None"
        elif mode == "count":
            value = str(len(getattr(data, "values", ())))
        elif mode == "frames":
            value = f"{_format_scalar(_scalar(data))} frames"
        else:
            value = _format_scalar(_scalar(data))
        rows.append((label, value))
    return tuple(rows)


def _scalar(data):
    if data is None:
        return None
    if hasattr(data, "value"):
        return data.value
    if isinstance(data, (bool, int, float, str)):
        return data
    return None


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return "—" if value is None else str(value)


def _enum_value(data) -> str:
    raw = _scalar(data)
    original_type = str(getattr(data, "orig_type", "") or "")
    return _enum_text(original_type, raw)


def _enum_reference_value(rsz: RszFile, data) -> str:
    instance_id = _scalar(data)
    if not isinstance(instance_id, int) or not 0 < instance_id < len(rsz.instance_infos):
        return _format_scalar(instance_id)
    fields = rsz.parsed_elements.get(instance_id, {})
    raw = _scalar(fields.get("_Value")) if isinstance(fields, dict) else None
    info = rsz.type_registry.get_type_info(rsz.instance_infos[instance_id].type_id)
    type_name = str(info.get("name", "")) if info else ""
    enum_type = type_name.removesuffix("_Serializable") + "_Fixed"
    return _enum_text(enum_type, raw)


def _referenced_fields(rsz: RszFile, data) -> dict:
    instance_id = _scalar(data)
    if not isinstance(instance_id, int) or not 0 < instance_id < len(rsz.instance_infos):
        return {}
    fields = rsz.parsed_elements.get(instance_id, {})
    return fields if isinstance(fields, dict) else {}


def _enum_text(enum_type: str, raw) -> str:
    if not isinstance(raw, int):
        return _format_scalar(raw)
    manager = EnumManager.instance()
    manager.game_version = "OnimushaWOTS"
    members = manager.get_enum_values(enum_type)
    match = next((item for item in members if item.get("value") == raw), None)
    return f"{match['name']} ({raw})" if match else str(raw)


def _flag_names(enum_type: str, value: int) -> str:
    manager = EnumManager.instance()
    manager.game_version = "OnimushaWOTS"
    members = manager.get_enum_values(enum_type)
    names = [
        str(item["name"])
        for item in members
        if int(item.get("value", 0)) != 0
        and value & int(item["value"]) == int(item["value"])
    ]
    return " | ".join(names) if names else "NONE (0)"
