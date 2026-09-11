from __future__ import annotations

"""Verified WOTS 1.0.1.0 RSZ layout corrections."""

from collections import ChainMap
from copy import deepcopy
from pathlib import Path


WOTS_ATTACK_PARAM_CRCS = {
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


class WotsTypeRegistry:
    """Non-mutating overlay for known WOTS RSZ dump mismatches."""

    def __init__(self, base):
        self.base = base
        self.json_path = getattr(base, "json_path", "")
        self._patched_by_id: dict[int, dict] = {}
        self._patched_by_name: dict[str, tuple[dict, int]] = {}
        registry_overrides: dict[str, dict] = {}
        for name, crc in WOTS_ATTACK_PARAM_CRCS.items():
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
            self._insert_after(
                fields,
                "_MiasmaOutsideDamage",
                _float_field("_OilDamage"),
            )
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
            type_id = int(type_id)
            self._patched_by_id[type_id] = patched
            self._patched_by_name[name] = (patched, type_id)
            registry_overrides[format(type_id, "x")] = patched

        self.registry = ChainMap(registry_overrides, getattr(base, "registry", {}))

    @staticmethod
    def _insert_after(fields: list[dict], anchor: str, field: dict) -> None:
        if any(item.get("name") == field["name"] for item in fields):
            return
        for index, item in enumerate(fields):
            if item.get("name") == anchor:
                fields.insert(index + 1, field)
                return
        raise ValueError(f"WOTS RSZ registry is missing {anchor}")

    def get_type_info(self, type_id: int):
        return self._patched_by_id.get(int(type_id)) or self.base.get_type_info(type_id)

    def find_type_by_name(self, type_name: str):
        return self._patched_by_name.get(type_name) or self.base.find_type_by_name(type_name)

    def pre_cache_types(self, type_ids) -> None:
        pre_cache = getattr(self.base, "pre_cache_types", None)
        if callable(pre_cache):
            pre_cache(type_ids)

    def __getattr__(self, name):
        return getattr(self.base, name)


def needs_wots_registry_overlay(registry, game: str = "") -> bool:
    normalized_game = "".join(character for character in str(game) if character.isalnum()).casefold()
    registry_name = Path(str(getattr(registry, "json_path", "") or "")).name.casefold()
    return normalized_game in {"onimushawots", "oniwots", "wots"} or registry_name == "rszoniwots.json"


def apply_wots_registry_overlay(registry, game: str = ""):
    if registry is None or isinstance(registry, WotsTypeRegistry):
        return registry
    return WotsTypeRegistry(registry) if needs_wots_registry_overlay(registry, game) else registry
