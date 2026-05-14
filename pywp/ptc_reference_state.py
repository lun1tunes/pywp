from __future__ import annotations

from collections.abc import Callable, Iterable

import streamlit as st

from pywp.ptc_anticollision_params import (
    ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY,
    ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY,
)
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    ImportedTrajectoryWell,
)

__all__ = [
    "clear_reference_dev_folder_state",
    "clear_reference_import_state",
    "init_reference_state_defaults",
    "reference_dev_folder_count_key",
    "reference_dev_folder_path_key",
    "reference_dev_folder_paths",
    "reference_kind_title",
    "reference_kind_wells",
    "reference_source_mode_key",
    "reference_source_text_key",
    "reference_wells_from_state",
    "reference_wells_state_key",
    "reference_welltrack_path_key",
    "set_reference_wells_for_kind",
]


def reference_wells_state_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_wells"


def reference_source_mode_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_source_mode"


def reference_source_text_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_source_text"


def reference_welltrack_path_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_welltrack_path"


def reference_dev_folder_count_key(kind: str) -> str:
    return f"wt_reference_{str(kind)}_dev_folder_count"


def reference_dev_folder_path_key(kind: str, index: int) -> str:
    return f"wt_reference_{str(kind)}_dev_folder_path_{int(index)}"


def reference_dev_folder_paths(kind: str) -> tuple[str, ...]:
    folder_count = _folder_count(kind)
    return tuple(
        str(
            st.session_state.get(reference_dev_folder_path_key(kind, index), "")
        ).strip()
        for index in range(folder_count)
    )


def clear_reference_dev_folder_state(kind: str) -> None:
    for index in range(_folder_count(kind)):
        st.session_state[reference_dev_folder_path_key(kind, index)] = ""
    st.session_state[reference_dev_folder_count_key(kind)] = 1


def clear_reference_import_state(
    kind: str,
    *,
    on_clear: Callable[[], None] | None = None,
) -> None:
    set_reference_wells_for_kind(kind=kind, wells=())
    if str(kind) == REFERENCE_WELL_ACTUAL:
        st.session_state[ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY] = []
        st.session_state[ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY] = []
    st.session_state[reference_source_text_key(kind)] = ""
    st.session_state[reference_welltrack_path_key(kind)] = ""
    clear_reference_dev_folder_state(kind)
    if on_clear is not None:
        on_clear()


def set_reference_wells_for_kind(
    *,
    kind: str,
    wells: Iterable[ImportedTrajectoryWell],
) -> None:
    normalized_kind = str(kind)
    st.session_state[reference_wells_state_key(normalized_kind)] = tuple(wells)
    actual_wells = tuple(
        st.session_state.get(reference_wells_state_key(REFERENCE_WELL_ACTUAL))
        or ()
    )
    approved_wells = tuple(
        st.session_state.get(reference_wells_state_key(REFERENCE_WELL_APPROVED))
        or ()
    )
    st.session_state["wt_reference_wells"] = actual_wells + approved_wells


def reference_wells_from_state() -> tuple[ImportedTrajectoryWell, ...]:
    actual_wells = tuple(
        st.session_state.get(reference_wells_state_key(REFERENCE_WELL_ACTUAL))
        or ()
    )
    approved_wells = tuple(
        st.session_state.get(reference_wells_state_key(REFERENCE_WELL_APPROVED))
        or ()
    )
    legacy_combined = tuple(st.session_state.get("wt_reference_wells") or ())
    if legacy_combined:
        legacy_actual = tuple(
            item
            for item in legacy_combined
            if str(getattr(item, "kind", "")) == REFERENCE_WELL_ACTUAL
        )
        legacy_approved = tuple(
            item
            for item in legacy_combined
            if str(getattr(item, "kind", "")) == REFERENCE_WELL_APPROVED
        )
        if not actual_wells and legacy_actual:
            actual_wells = legacy_actual
            st.session_state[reference_wells_state_key(REFERENCE_WELL_ACTUAL)] = (
                actual_wells
            )
        if not approved_wells and legacy_approved:
            approved_wells = legacy_approved
            st.session_state[reference_wells_state_key(REFERENCE_WELL_APPROVED)] = (
                approved_wells
            )
    combined = actual_wells + approved_wells
    if combined:
        st.session_state["wt_reference_wells"] = combined
        return combined

    return legacy_combined


def reference_kind_title(kind: str) -> str:
    if str(kind) == REFERENCE_WELL_ACTUAL:
        return "Фактические скважины"
    return "Проектные утвержденные скважины"


def reference_kind_wells(kind: str) -> tuple[ImportedTrajectoryWell, ...]:
    wells = tuple(st.session_state.get(reference_wells_state_key(kind)) or ())
    if wells or not st.session_state.get("wt_reference_wells"):
        return wells
    reference_wells_from_state()
    return tuple(st.session_state.get(reference_wells_state_key(kind)) or ())


def init_reference_state_defaults() -> None:
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        st.session_state.setdefault(reference_wells_state_key(kind), ())
        st.session_state.setdefault(reference_source_mode_key(kind), "Загрузить .dev")
        st.session_state.setdefault(reference_source_text_key(kind), "")
        st.session_state.setdefault(reference_welltrack_path_key(kind), "")
        st.session_state.setdefault(reference_dev_folder_count_key(kind), 1)
        st.session_state.setdefault(reference_dev_folder_path_key(kind, 0), "")
    st.session_state.setdefault("wt_reference_wells", ())


def _folder_count(kind: str) -> int:
    count_key = reference_dev_folder_count_key(kind)
    try:
        folder_count = int(st.session_state.get(count_key, 1))
    except (TypeError, ValueError):
        folder_count = 1
    folder_count = max(1, folder_count)
    st.session_state[count_key] = folder_count
    return folder_count
