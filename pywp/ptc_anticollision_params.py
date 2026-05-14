from __future__ import annotations

from collections.abc import Iterable, MutableMapping

import streamlit as st

from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    ImportedTrajectoryWell,
)
from pywp.uncertainty import (
    PlanningUncertaintyModel,
    UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC,
    UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC,
    planning_uncertainty_model_for_preset,
)

__all__ = [
    "ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY",
    "ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY",
    "actual_reference_names",
    "reference_uncertainty_models_from_state",
    "reference_uncertainty_models_for_unknown_actual_names",
    "render_anticollision_params_block",
]

ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY = "wt_actual_reference_mwd_unknown_names"
ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY = (
    "wt_actual_reference_mwd_unknown_widget_names"
)
_ACTUAL_REFERENCE_MWD_POOR_DISPLAY_NAMES_KEY = (
    "wt_actual_reference_mwd_poor_display_names"
)


def actual_reference_names(
    reference_wells: Iterable[ImportedTrajectoryWell],
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for well in reference_wells:
        if str(getattr(well, "kind", "")) != REFERENCE_WELL_ACTUAL:
            continue
        name = str(getattr(well, "name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def reference_uncertainty_models_for_unknown_actual_names(
    reference_wells: Iterable[ImportedTrajectoryWell],
    *,
    unknown_names: object,
) -> dict[str, PlanningUncertaintyModel]:
    actual_names = actual_reference_names(reference_wells)
    unknown_set = set(
        _sanitize_unknown_names(unknown_names, actual_names=actual_names)
    )
    poor_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_POOR_MAGNETIC
    )
    unknown_model = planning_uncertainty_model_for_preset(
        UNCERTAINTY_PRESET_MWD_UNKNOWN_MAGNETIC
    )
    return {
        name: (unknown_model if name in unknown_set else poor_model)
        for name in actual_names
    }


def reference_uncertainty_models_from_state(
    reference_wells: Iterable[ImportedTrajectoryWell],
    *,
    state: MutableMapping[str, object] | None = None,
) -> dict[str, PlanningUncertaintyModel]:
    state_mapping = st.session_state if state is None else state
    actual_names = actual_reference_names(reference_wells)
    raw_unknown_names = state_mapping.get(
        ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY,
        state_mapping.get(ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY, ()),
    )
    unknown_names = _sanitize_unknown_names(
        raw_unknown_names,
        actual_names=actual_names,
    )
    return reference_uncertainty_models_for_unknown_actual_names(
        reference_wells,
        unknown_names=unknown_names,
    )


def render_anticollision_params_block(
    *,
    reference_wells: Iterable[ImportedTrajectoryWell],
    state: MutableMapping[str, object] | None = None,
) -> None:
    state_mapping = st.session_state if state is None else state
    actual_names = actual_reference_names(reference_wells)
    st.markdown("#### Параметры anti-collision")
    if not actual_names:
        st.caption("Фактический фонд не загружен.")
        state_mapping[ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY] = []
        state_mapping[_ACTUAL_REFERENCE_MWD_POOR_DISPLAY_NAMES_KEY] = []
        return

    unknown_names = _sanitize_unknown_names(
        state_mapping.get(
            ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY,
            state_mapping.get(ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY, ()),
        ),
        actual_names=actual_names,
    )
    poor_names = tuple(name for name in actual_names if name not in set(unknown_names))
    state_mapping[ACTUAL_REFERENCE_MWD_UNKNOWN_NAMES_KEY] = list(unknown_names)
    state_mapping[_ACTUAL_REFERENCE_MWD_POOR_DISPLAY_NAMES_KEY] = list(poor_names)
    if ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY not in state_mapping:
        state_mapping[ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY] = list(unknown_names)

    c1, c2 = st.columns([1.0, 1.0], gap="small", vertical_alignment="bottom")
    with c1:
        st.multiselect(
            "MWD POOR Magnetic",
            options=list(actual_names),
            default=list(poor_names),
            disabled=True,
            help=(
                "Дефолтная ISCWSA модель для фактического фонда. "
                "Скважины автоматически исключаются отсюда при выборе MWD Unknown."
            ),
        )
    with c2:
        st.multiselect(
            "MWD Unknown Magnetic",
            options=list(actual_names),
            key=ACTUAL_REFERENCE_MWD_UNKNOWN_WIDGET_KEY,
            help=(
                "Более консервативная ISCWSA модель. Выбранные скважины "
                "считаются Unknown и исключаются из списка MWD POOR."
            ),
        )
    st.caption(f"Факт фонд: POOR — {len(poor_names)}, Unknown — {len(unknown_names)}.")


def _sanitize_unknown_names(
    raw_names: object,
    *,
    actual_names: tuple[str, ...],
) -> tuple[str, ...]:
    allowed = set(actual_names)
    result: list[str] = []
    if isinstance(raw_names, str):
        raw_iterable: Iterable[object] = (raw_names,)
    elif isinstance(raw_names, bytes) or not isinstance(raw_names, Iterable):
        return ()
    else:
        raw_iterable = raw_names
    for raw_name in raw_iterable:
        name = str(raw_name).strip()
        if name in allowed and name not in result:
            result.append(name)
    return tuple(result)
