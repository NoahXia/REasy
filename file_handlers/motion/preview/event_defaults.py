"""Read-only event fields omitted from a clip, with evidence-backed defaults."""
from __future__ import annotations

import re
import json

from .wots_event_defaults_data import CONSTRUCTOR_DEFAULTS, RESET_DIFFERENCES


def property_name(name: str) -> str:
    name = name.lstrip("_")
    match = re.fullmatch(r"<(.+)>k__BackingField", name)
    return match.group(1) if match else name


# WOTS EXE 0x1408A6BF0: scalar initialization and nested MotRateXZ creation.
# No primitive zero is inferred: a field may be assigned by other initialization.
WOTS_DEFAULTS = {
    "app.motion_track.PlayerTrackingTarget": {
        "SearchDistance": "2.0",
        "MoveLimmit": "0.5",
        "OffsetDistance": "1.0",
        "MotionTransRate.x": "1.0",
        "MotionTransRate.z": "1.0",
        "MotionTransRateXZ": "1.0",
    },
}


def missing_event_fields(track_type, properties, registry, *, game=""):
    """Return (path, display value) without adding synthetic clip properties."""
    if registry is None or not track_type:
        return ()
    type_name = track_type if "." in track_type else "app.motion_track." + track_type
    defaults = WOTS_DEFAULTS.get(type_name, {}) if game == "wots" else {}
    observed = CONSTRUCTOR_DEFAULTS.get(type_name, {}) if game == "wots" else {}
    reset_differences = RESET_DIFFERENCES.get(type_name, {}) if game == "wots" else {}
    rows = []

    def observed_text(path, field):
        sample = observed.get(path)
        if sample is None or sample["declared_type"] != field.get("original_type"):
            return None
        text = json.dumps(sample["value"], ensure_ascii=False)
        text += " — observed constructor default (WotS, 2026-09-28); not stored in clip"
        if path in reset_differences:
            text += "; reset-after samples: " + json.dumps(reset_differences[path])
            text += " (differs; not the same initialization stage)"
        return text

    def fields_for(name, visited=()):
        if name in visited:
            return {}
        info, _ = registry.find_type_by_name(name)
        if not info:
            return {}
        parent = info.get("parent", "")
        fields = fields_for(parent, (*visited, name)) if parent.startswith("app.") else {}
        fields.update({property_name(f["name"]): f for f in info.get("fields", [])})
        return fields

    def visit(name, explicit, prefix="", ancestors=()):
        if name in ancestors or len(ancestors) >= 8:
            return
        supplied = {property_name(p.name): p for p in explicit}
        for field_name, field in fields_for(name).items():
            path = prefix + field_name
            prop = supplied.get(field_name)
            nested_type = field.get("original_type", "")
            nested = fields_for(nested_type) if not field.get("array") else {}
            # Explicit object references must not be interpreted as inline defaults.
            sampled = observed_text(path, field)
            if prop is None and sampled is not None:
                rows.append((path, sampled))
            elif nested and (prop is None or prop.children):
                visit(nested_type, prop.children if prop else (), path + ".", (*ancestors, name))
            elif prop is None:
                if path in defaults:
                    value = defaults[path] + " — native constructor default (WotS); not stored in clip"
                else:
                    value = "Default unresolved — not stored in clip (" + field.get("original_type", field.get("type", "")) + ")"
                rows.append((path, value))
    visit(type_name, properties)
    return tuple(rows)
