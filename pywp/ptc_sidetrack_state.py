from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import math

SIDETRACK_AUTO = "Авто"
SIDETRACK_MANUAL = "Ручной"
SIDETRACK_MODE_OPTIONS = (SIDETRACK_AUTO, SIDETRACK_MANUAL)
SIDETRACK_KIND_OPTIONS = ("MD", "Z")
SIDETRACK_PARENT_KEY = "wt_sidetrack_window_parent_name"
SIDETRACK_EDITOR_OVERRIDES_KEY = "wt_sidetrack_window_editor_overrides"


def sidetrack_mode_key(parent_name: str) -> str:
    return f"wt_sidetrack_window_mode::{parent_name}"


def sidetrack_kind_key(parent_name: str) -> str:
    return f"wt_sidetrack_window_kind::{parent_name}"


def sidetrack_value_key(parent_name: str) -> str:
    return f"wt_sidetrack_window_value::{parent_name}"


def queue_editor_sidetrack_window_override(
    state: MutableMapping[str, object],
    *,
    well_name: str,
    kind: str,
    value_m: float,
) -> None:
    name = str(well_name).strip()
    normalized_kind = str(kind).strip().upper()
    if not name or normalized_kind not in SIDETRACK_KIND_OPTIONS:
        return
    value = float(value_m)
    if not math.isfinite(value):
        return
    raw_overrides = state.get(SIDETRACK_EDITOR_OVERRIDES_KEY)
    overrides = dict(raw_overrides) if isinstance(raw_overrides, Mapping) else {}
    overrides[name] = {"kind": normalized_kind, "value_m": value}
    state[SIDETRACK_EDITOR_OVERRIDES_KEY] = overrides


def apply_editor_sidetrack_window_defaults(
    state: MutableMapping[str, object],
    *,
    parent_names: list[str],
) -> None:
    raw_overrides = state.get(SIDETRACK_EDITOR_OVERRIDES_KEY)
    if not isinstance(raw_overrides, Mapping):
        return

    allowed = {str(name) for name in parent_names}
    remaining: dict[str, object] = {}
    selected_parent = ""
    for raw_name, raw_payload in raw_overrides.items():
        name = str(raw_name).strip()
        if name not in allowed or not isinstance(raw_payload, Mapping):
            remaining[name] = raw_payload
            continue
        kind = str(raw_payload.get("kind", "MD")).strip().upper()
        if kind not in SIDETRACK_KIND_OPTIONS:
            kind = "MD"
        try:
            value = float(raw_payload.get("value_m"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        state[sidetrack_mode_key(name)] = SIDETRACK_MANUAL
        state[sidetrack_kind_key(name)] = kind
        state[sidetrack_value_key(name)] = value
        selected_parent = name

    if selected_parent:
        state[SIDETRACK_PARENT_KEY] = selected_parent
    if remaining:
        state[SIDETRACK_EDITOR_OVERRIDES_KEY] = remaining
    else:
        state.pop(SIDETRACK_EDITOR_OVERRIDES_KEY, None)
