from __future__ import annotations

import math
from collections.abc import Mapping


MD_POSTCHECK_TOLERANCE_M = 1e-6
_PILOT_TRAJECTORY_TYPES = frozenset({"PILOT"})
_SIDETRACK_TRAJECTORY_TYPES = frozenset(
    {
        "PILOT_SIDETRACK",
        "FACT_SIDETRACK",
    }
)
_PILOT_FROM_MAIN_BORE_MODE = "pilot_from_main_bore"


def _finite_summary_float(
    summary: Mapping[str, object],
    key: str,
) -> float | None:
    try:
        value = float(summary.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def md_total_display_label_ru(summary: Mapping[str, object]) -> str:
    trajectory_type = str(summary.get("trajectory_type", "")).strip().upper()
    if trajectory_type in _PILOT_TRAJECTORY_TYPES:
        return "MD пилота от устья до забоя"
    if trajectory_type in _SIDETRACK_TRAJECTORY_TYPES:
        if (
            trajectory_type == "PILOT_SIDETRACK"
            and str(summary.get("pilot_planning_mode", "")).strip()
            == _PILOT_FROM_MAIN_BORE_MODE
        ):
            return "MD ГС от устья до забоя"
        return "MD бокового ствола от устья до забоя"
    return "Итоговая MD"


def parent_bore_md_display_label_ru(summary: Mapping[str, object]) -> str:
    trajectory_type = str(summary.get("trajectory_type", "")).strip().upper()
    if trajectory_type == "FACT_SIDETRACK":
        return "MD исходного ствола от устья до забоя"
    return "MD пилота от устья до забоя"


def sidetrack_branch_md_display_label_ru(summary: Mapping[str, object]) -> str:
    """Return the label for the calculated branch after the window."""

    trajectory_type = str(summary.get("trajectory_type", "")).strip().upper()
    if (
        trajectory_type == "PILOT_SIDETRACK"
        and str(summary.get("pilot_planning_mode", "")).strip()
        == _PILOT_FROM_MAIN_BORE_MODE
    ):
        return "ГС от окна до забоя"
    return "Боковой ствол от окна до забоя"


def md_postcheck_values(
    summary: Mapping[str, object],
) -> tuple[float | None, float | None, float]:
    """Return the well MD, configured limit and excess used by postcheck.

    ``md_total_m`` is always the measured depth of one bore from the wellhead
    to its bottom. Drilled-footage metrics for pilot/sidetrack combinations
    are intentionally excluded from this limit check.
    """

    trajectory_type = str(summary.get("trajectory_type", "")).strip().upper()
    checked_md_m = _finite_summary_float(summary, "md_total_m")
    if checked_md_m is not None and checked_md_m < 0.0:
        checked_md_m = None
    if checked_md_m is None and trajectory_type in _SIDETRACK_TRAJECTORY_TYPES:
        checked_md_m = _finite_summary_float(summary, "sidetrack_total_md_m")
    elif checked_md_m is None and trajectory_type in _PILOT_TRAJECTORY_TYPES:
        checked_md_m = _finite_summary_float(summary, "pilot_total_md_m")
    if checked_md_m is not None and checked_md_m < 0.0:
        checked_md_m = None
    md_limit_m = _finite_summary_float(summary, "max_total_md_postcheck_m")

    if checked_md_m is not None and checked_md_m >= 0.0:
        if md_limit_m is not None and md_limit_m > 0.0:
            return checked_md_m, md_limit_m, max(0.0, checked_md_m - md_limit_m)
        return checked_md_m, md_limit_m, 0.0

    if trajectory_type in _SIDETRACK_TRAJECTORY_TYPES:
        # Legacy sidetrack excess may have been computed from pilot + lateral.
        # It cannot establish the final MD when the bore MD is unavailable.
        return checked_md_m, md_limit_m, 0.0

    reported_excess_m = _finite_summary_float(summary, "md_postcheck_excess_m")
    excess_m = max(0.0, reported_excess_m or 0.0)
    if excess_m <= MD_POSTCHECK_TOLERANCE_M:
        return checked_md_m, md_limit_m, 0.0
    if checked_md_m is None and md_limit_m is not None:
        checked_md_m = md_limit_m + excess_m
    if md_limit_m is None and checked_md_m is not None:
        md_limit_m = max(0.0, checked_md_m - excess_m)
    return checked_md_m, md_limit_m, excess_m


def md_postcheck_issue_message_ru(summary: Mapping[str, object]) -> str:
    checked_md_m, md_limit_m, excess_m = md_postcheck_values(summary)
    if excess_m <= MD_POSTCHECK_TOLERANCE_M:
        return ""

    label = md_total_display_label_ru(summary)
    subject = "итоговой MD" if label == "Итоговая MD" else label
    if checked_md_m is None or md_limit_m is None:
        return f"Превышен лимит {subject} (постпроверка): +{excess_m:.2f} м."
    return (
        f"Превышен лимит {subject} (постпроверка): "
        f"{checked_md_m:.2f} м > {md_limit_m:.2f} м "
        f"(+{excess_m:.2f} м)."
    )
