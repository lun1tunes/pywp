from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from pywp.anticollision import AntiCollisionAnalysis
from pywp.constants import SMALL
from pywp.models import Point3D
from pywp.pilot_wells import is_pilot_name
from pywp.plot_axes import equalized_axis_ranges
from pywp.three_config import DEFAULT_THREE_CAMERA
from pywp import ptc_three_overrides
from pywp import ptc_three_payload
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    REFERENCE_WELL_KIND_COLORS,
    REFERENCE_WELL_KIND_LABELS,
    ImportedTrajectoryWell,
    reference_well_display_label,
)
from pywp.uncertainty import (
    WellUncertaintyOverlay,
    build_uncertainty_tube_mesh,
)
from pywp.welltrack_batch import SuccessfulWellPlan

__all__ = [
    "single_well_three_payload",
    "single_well_target_only_three_payload",
    "all_wells_three_payload",
    "anticollision_three_payload",
]

THREE_BACKGROUND = "#FFFFFF"
TRAJECTORY_COLOR_PRIMARY = "#C1121F"
PLAN_CSB_COLOR = "#0B6E4F"
ACTUAL_PROFILE_COLOR = "#111111"
TARGET_COLOR_PRIMARY = "#C1121F"
UNCERTAINTY_SURFACE_COLOR = "#5A5A5A"
REFERENCE_LABEL_ACTUAL_COLOR = "#111111"
REFERENCE_LABEL_APPROVED_COLOR = "#C62828"
REFERENCE_PAD_LABEL_COLOR = "#334155"
REFERENCE_LABEL_HORIZONTAL_INC_THRESHOLD_DEG = 80.0
REFERENCE_LABEL_HORIZONTAL_MIN_INTERVAL_M = 100.0
REFERENCE_PAD_GROUP_DISTANCE_M = 300.0
WT_3D_RENDER_FAST = "Быстро"
WT_3D_FAST_REFERENCE_TARGET_POINTS = 72
WT_3D_FAST_CALC_TARGET_POINTS = 180
WT_THREE_MAX_HOVER_POINTS_PER_TRACE = (
    ptc_three_payload.WT_THREE_MAX_HOVER_POINTS_PER_TRACE
)
WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE = (
    ptc_three_payload.WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
)


@dataclass(frozen=True)
class _ReferencePadLabel:
    label: str
    x: float
    y: float
    z: float


def single_well_three_payload(
    stations: pd.DataFrame,
    *,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = (),
    well_name: str | None = None,
    md_t1_m: float | None = None,
    kop_md_m: float | None = None,
    trajectory_line_dash: str = "solid",
    plan_csb_df: pd.DataFrame | None = None,
    actual_df: pd.DataFrame | None = None,
    uncertainty_overlay: WellUncertaintyOverlay | None = None,
    pilot_name: str | None = None,
    pilot_stations: pd.DataFrame | None = None,
    pilot_study_points: tuple[Point3D, ...] = (),
) -> dict[str, object]:
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    payload = _base_payload(title="3D траектория")

    _append_station_line(
        payload,
        stations=stations,
        name="Траектория",
        color=TRAJECTORY_COLOR_PRIMARY,
        width_role="line",
        dash=trajectory_line_dash,
        hover_name=well_name or "Траектория",
        x_arrays=x_arrays,
        y_arrays=y_arrays,
        z_arrays=z_arrays,
    )
    if uncertainty_overlay is not None and uncertainty_overlay.samples:
        _append_uncertainty_overlay(
            payload,
            overlay=uncertainty_overlay,
            color=UNCERTAINTY_SURFACE_COLOR,
            terminal_boundary_color=_lighten_hex(TRAJECTORY_COLOR_PRIMARY),
            legend_label="Конус неопределенности (2σ)",
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
        )
    if plan_csb_df is not None and len(plan_csb_df) > 0:
        _append_station_line(
            payload,
            stations=plan_csb_df,
            name="План ЦСБ",
            color=PLAN_CSB_COLOR,
            width_role="line",
            hover_name="План ЦСБ",
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
        )
    if actual_df is not None and len(actual_df) > 0:
        _append_station_line(
            payload,
            stations=actual_df,
            name="Фактический профиль",
            color=ACTUAL_PROFILE_COLOR,
            width_role="line",
            hover_name="Фактический профиль",
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
        )
    if pilot_stations is not None and len(pilot_stations) > 0:
        _append_station_line(
            payload,
            stations=pilot_stations,
            name=str(pilot_name or "Пилот"),
            color=TRAJECTORY_COLOR_PRIMARY,
            width_role="line",
            hover_name=str(pilot_name or "Пилот"),
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
        )

    target_points, target_labels = _target_marker_points_and_labels(
        surface=surface,
        t1=t1,
        t3=t3,
        target_pairs=target_pairs,
    )
    _append_arrays_for_points(x_arrays, y_arrays, z_arrays, target_points)
    payload["points"].append(
        {
            "name": "Цели",
            "points": target_points,
            "color": TARGET_COLOR_PRIMARY,
            "opacity": 1.0,
            "size": 6.0,
            "symbol": "circle",
            "hover": [
                _target_hover_item(label, "Цели")
                for label in target_labels
            ],
            "role": "marker",
        }
    )
    for target_label, target_point in zip(target_labels, target_points, strict=False):
        if str(target_label).strip().upper() == "S":
            continue
        payload["labels"].append(
            _point_payload_label(
                str(target_label),
                target_point,
                TARGET_COLOR_PRIMARY,
                role="target_label",
            )
        )
    if well_name:
        payload["labels"].append(_well_name_label(str(well_name), t3, TARGET_COLOR_PRIMARY))
    _append_pilot_study_markers(
        payload,
        pilot_name=pilot_name,
        study_points=pilot_study_points,
        color=TRAJECTORY_COLOR_PRIMARY,
        x_arrays=x_arrays,
        y_arrays=y_arrays,
        z_arrays=z_arrays,
    )

    if md_t1_m is not None and not stations.empty and "MD_m" in stations.columns:
        calc_points = _calculated_t1_t3_points(stations, md_t1_m=float(md_t1_m))
        if calc_points:
            _append_arrays_for_points(x_arrays, y_arrays, z_arrays, calc_points)
            payload["points"].append(
                {
                    "name": "Расчетные в t1/t3",
                    "points": calc_points,
                    "color": "#073B4C",
                    "opacity": 1.0,
                    "size": 5.0,
                    "symbol": "diamond",
                    "hover": [
                        {"name": "Расчетные в t1/t3", "point": "t1"},
                        {"name": "Расчетные в t1/t3", "point": "t3"},
                    ],
                    "role": "marker",
                }
            )

    if kop_md_m is not None:
        kop_point = _station_point_at_md(stations, float(kop_md_m))
        if kop_point is not None:
            _append_arrays_for_points(x_arrays, y_arrays, z_arrays, [kop_point])
            payload["points"].append(
                {
                    "name": "KOP",
                    "points": [kop_point],
                    "color": "#073B4C",
                    "opacity": 1.0,
                    "size": 5.5,
                    "symbol": "circle",
                    "hover": [
                        {
                            "name": "KOP",
                            "point": "KOP",
                            "md": float(kop_md_m),
                        }
                    ],
                    "role": "marker",
                }
            )
            payload["labels"].append(
                _point_payload_label(
                    "KOP",
                    kop_point,
                    "#073B4C",
                    role="control_point_label",
                )
            )

    _append_single_well_zero_axes(
        payload,
        surface=surface,
        x_arrays=x_arrays,
        y_arrays=y_arrays,
        z_arrays=z_arrays,
    )
    payload["bounds"] = _bounds_from_arrays(x_arrays=x_arrays, y_arrays=y_arrays, z_arrays=z_arrays)
    return ptc_three_payload.optimize_three_payload(payload)


def single_well_target_only_three_payload(
    *,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    well_name: str = "single_well",
    color: str = TARGET_COLOR_PRIMARY,
) -> dict[str, object]:
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    payload = _base_payload(title="3D цели (без траектории)")
    marker_name = f"{well_name}: цели (без траектории)"
    target_points, target_labels = _target_marker_points_and_labels(
        surface=surface,
        t1=t1,
        t3=t3,
    )
    _append_target_markers(
        payload,
        name=marker_name,
        surface=surface,
        t1=t1,
        t3=t3,
        color=color,
        size=8.0,
        symbol="cross",
        x_arrays=x_arrays,
        y_arrays=y_arrays,
        z_arrays=z_arrays,
    )
    for label, point in zip(target_labels, target_points, strict=False):
        payload["labels"].append(
            _point_payload_label(
                str(label),
                point,
                str(color),
                role="target_label",
            )
        )
    payload["labels"].append(_well_name_label(str(well_name), t3, str(color)))
    _append_unique_legend_item(
        payload,
        label=marker_name,
        color=str(color),
        opacity=1.0,
        symbol="point",
    )
    _append_single_well_zero_axes(
        payload,
        surface=surface,
        x_arrays=x_arrays,
        y_arrays=y_arrays,
        z_arrays=z_arrays,
    )
    payload["bounds"] = _bounds_from_arrays(
        x_arrays=x_arrays,
        y_arrays=y_arrays,
        z_arrays=z_arrays,
    )
    return ptc_three_payload.optimize_three_payload(payload)


def all_wells_three_payload(
    successes: list[SuccessfulWellPlan],
    *,
    target_only_wells: list[object] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    name_to_color: Mapping[str, str] | None = None,
    pilot_study_points_by_name: Mapping[str, tuple[Point3D, ...]] | None = None,
    focus_well_names: tuple[str, ...] = (),
    render_mode: str = WT_3D_RENDER_FAST,
    fallback_color: Callable[[int], str] | None = None,
) -> dict[str, object]:
    payload = _base_payload(title="Все рассчитанные скважины (3D)")
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
    z_focus_arrays: list[np.ndarray] = []
    focus_set = _clean_name_set(focus_well_names)
    color_map = dict(name_to_color or {})
    pilot_points_map = dict(pilot_study_points_by_name or {})

    for index, success in enumerate(successes):
        well_name = str(success.name)
        color = color_map.get(well_name, _fallback_color(index, fallback_color))
        is_pilot = is_pilot_name(well_name)
        pilot_study_points = (
            _pilot_study_points_for_well(
                well_name,
                pilot_points_map,
                fallback_points=(success.t1, success.t3),
            )
            if is_pilot
            else ()
        )
        stations = _maybe_decimated_stations(
            success.stations,
            render_mode=render_mode,
            target_points=WT_3D_FAST_CALC_TARGET_POINTS,
        )
        _append_station_line(
            payload,
            stations=stations,
            name=well_name,
            color=color,
            width_role="line",
            dash="dash" if bool(success.md_postcheck_exceeded) else "solid",
            hover_name=well_name,
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
            focus_arrays=(x_focus_arrays, y_focus_arrays, z_focus_arrays)
            if _include_name_in_focus(well_name, focus_set)
            else None,
        )
        _append_target_markers(
            payload,
            name=f"{well_name}: цели",
            surface=success.surface,
            t1=success.t1,
            t3=success.t3,
            target_pairs=tuple(getattr(success, "target_pairs", ()) or ()),
            target_points=pilot_study_points,
            target_labels=_pilot_study_label_texts(
                well_name,
                len(pilot_study_points),
            )
            if is_pilot
            else (),
            color=color,
            size=5.0,
            symbol="circle",
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
            focus_arrays=(x_focus_arrays, y_focus_arrays, z_focus_arrays)
            if _include_name_in_focus(well_name, focus_set)
            else None,
        )
        payload["labels"].append(_well_name_label(well_name, success.t3, color))
        if pilot_study_points:
            _append_pilot_study_labels(
                payload,
                pilot_name=well_name,
                study_points=pilot_study_points,
                color=color,
            )
        _append_unique_legend_item(payload, label=well_name, color=color, opacity=1.0)

    _append_reference_wells(
        payload,
        reference_wells=reference_wells,
        render_mode=render_mode,
        x_arrays=x_arrays,
        y_arrays=y_arrays,
        z_arrays=z_arrays,
    )

    for target_only in target_only_wells or ():
        well_name = str(getattr(target_only, "name"))
        color = color_map.get(well_name, "#6B7280")
        target_points = tuple(getattr(target_only, "target_points", ()) or ())
        target_labels = tuple(getattr(target_only, "target_labels", ()) or ())
        target_focus_arrays = (
            (x_focus_arrays, y_focus_arrays, z_focus_arrays)
            if _include_target_only_in_focus(
                well_name=well_name,
                focus_set=focus_set,
                target_points=target_points,
                target_labels=target_labels,
            )
            else None
        )
        _append_target_markers(
            payload,
            name=f"{well_name}: цели (без траектории)",
            surface=getattr(target_only, "surface"),
            t1=getattr(target_only, "t1"),
            t3=getattr(target_only, "t3"),
            target_pairs=tuple(getattr(target_only, "target_pairs", ()) or ()),
            target_points=target_points,
            target_labels=target_labels,
            color=color,
            size=10.0,
            symbol="cross",
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
            focus_arrays=target_focus_arrays,
            hover_extra={
                "status": str(getattr(target_only, "status", "") or ""),
                "problem": str(getattr(target_only, "problem", "") or ""),
            },
        )
        payload["labels"].append(_well_name_label(well_name, getattr(target_only, "t3"), color))
        _append_unique_legend_item(
            payload,
            label=f"{well_name}: цели (без траектории)",
            color=color,
            opacity=1.0,
            symbol="point",
        )

    payload["bounds"] = _bounds_from_arrays(
        x_arrays=x_focus_arrays or x_arrays,
        y_arrays=y_focus_arrays or y_arrays,
        z_arrays=z_focus_arrays or z_arrays,
    )
    return ptc_three_payload.optimize_three_payload(payload)


def _include_target_only_in_focus(
    *,
    well_name: str,
    focus_set: set[str],
    target_points: tuple[Point3D, ...],
    target_labels: tuple[str, ...],
) -> bool:
    if _include_name_in_focus(well_name, focus_set):
        return True
    if not focus_set:
        return True
    # ZBS has no source wellhead in target records and is intentionally not a
    # pad member. Keep its two editable targets in camera bounds for focused
    # views so a failed sidetrack can still be fixed from 3D.
    labels = tuple(label.lower() for label in target_labels)
    return len(target_points) == 2 and labels == ("t1", "t3")


def anticollision_three_payload(
    analysis: AntiCollisionAnalysis,
    *,
    previous_successes_by_name: Mapping[str, SuccessfulWellPlan] | None = None,
    pilot_study_points_by_name: Mapping[str, tuple[Point3D, ...]] | None = None,
    focus_well_names: tuple[str, ...] = (),
    render_mode: str = WT_3D_RENDER_FAST,
) -> dict[str, object]:
    payload = _base_payload(title="Anti-collision: 3D конусы неопределенности")
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    z_arrays: list[np.ndarray] = []
    x_focus_arrays: list[np.ndarray] = []
    y_focus_arrays: list[np.ndarray] = []
    z_focus_arrays: list[np.ndarray] = []
    focus_set = _clean_name_set(focus_well_names)
    pilot_points_map = dict(pilot_study_points_by_name or {})
    reference_wells = tuple(well for well in analysis.wells if bool(well.is_reference_only))
    focus_reference_names = (
        _anticollision_reference_cone_focus_names(analysis)
        if str(render_mode).strip() == WT_3D_RENDER_FAST
        else set()
    )
    well_lookup = {str(well.name): well for well in analysis.wells}
    aggregated_reference_wells: list[object] = []

    for well in analysis.wells:
        well_name = str(well.name)
        is_reference_only = bool(well.is_reference_only)
        is_focus_reference = well_name in focus_reference_names
        if (
            str(render_mode).strip() == WT_3D_RENDER_FAST
            and is_reference_only
            and not is_focus_reference
        ):
            aggregated_reference_wells.append(well)
            continue
        well_label = _analysis_well_label(well)
        include_in_focus = (not focus_set and not is_reference_only) or (
            bool(focus_set) and well_name in focus_set
        )
        focus_arrays = (
            (x_focus_arrays, y_focus_arrays, z_focus_arrays)
            if include_in_focus
            else None
        )
        tube_mesh = (
            build_uncertainty_tube_mesh(well.overlay)
            if (
                str(render_mode).strip() != WT_3D_RENDER_FAST
                or not is_reference_only
                or is_focus_reference
            )
            else None
        )
        if tube_mesh is not None:
            _append_mesh(
                payload,
                vertices_xyz=tube_mesh.vertices_xyz,
                faces_ijk=np.column_stack([tube_mesh.i, tube_mesh.j, tube_mesh.k]),
                name=f"{well_label} cone",
                color=str(well.color),
                opacity=0.10,
                role="cone",
                x_arrays=x_arrays,
                y_arrays=y_arrays,
                z_arrays=z_arrays,
                focus_arrays=focus_arrays,
            )
        if well.overlay.samples and (
            str(render_mode).strip() != WT_3D_RENDER_FAST or not is_reference_only
        ):
            terminal_ring = np.asarray(well.overlay.samples[-1].ring_xyz, dtype=float)
            _append_line_segments(
                payload,
                segments=[_points_from_xyz_array(terminal_ring)],
                name=f"{well_label}: граница конуса",
                color=_lighten_hex(str(well.color), 0.55),
                opacity=0.72,
                dash="solid",
                role="cone_tip",
            )
            _append_arrays(x_arrays, y_arrays, z_arrays, terminal_ring)
            if focus_arrays is not None:
                _append_arrays(*focus_arrays, terminal_ring)

        stations = _maybe_decimated_stations(
            well.stations,
            render_mode=render_mode,
            target_points=(
                WT_3D_FAST_REFERENCE_TARGET_POINTS
                if is_reference_only
                else WT_3D_FAST_CALC_TARGET_POINTS
            ),
        )
        _append_station_line(
            payload,
            stations=stations,
            name=well_label,
            color=str(well.color),
            width_role="line",
            hover_name=_reference_hover_name(well) if is_reference_only else well_label,
            hover_role="reference_hover" if is_reference_only else "trajectory_hover",
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
            focus_arrays=focus_arrays,
        )
        if not is_reference_only:
            _append_unique_legend_item(
                payload,
                label=well_label,
                color=str(well.color),
                opacity=1.0,
            )
        previous_success = (previous_successes_by_name or {}).get(well_name)
        if previous_success is not None and not previous_success.stations.empty:
            _append_station_line(
                payload,
                stations=previous_success.stations,
                name=f"{well_name}: до anti-collision",
                color=str(well.color),
                opacity=0.78,
                width_role="line",
                dash="dot",
                hover_name=f"{well_name}: до anti-collision",
                x_arrays=x_arrays,
                y_arrays=y_arrays,
                z_arrays=z_arrays,
                focus_arrays=focus_arrays,
            )
        if (well.t1 is not None) and (well.t3 is not None) and not is_reference_only:
            is_pilot = is_pilot_name(well_name)
            pilot_points = (
                _pilot_study_points_for_well(
                    well_name,
                    pilot_points_map,
                    fallback_points=(well.t1, well.t3),
                )
                if is_pilot
                else ()
            )
            _append_target_markers(
                payload,
                name=f"{well_label}: цели",
                surface=well.surface,
                t1=well.t1,
                t3=well.t3,
                target_pairs=tuple(getattr(well, "target_pairs", ()) or ()),
                target_points=pilot_points,
                target_labels=_pilot_study_label_texts(
                    well_name,
                    len(pilot_points),
                )
                if is_pilot
                else (),
                color=str(well.color),
                size=5.0,
                symbol="circle",
                x_arrays=x_arrays,
                y_arrays=y_arrays,
                z_arrays=z_arrays,
                focus_arrays=focus_arrays,
            )
            if pilot_points:
                _append_pilot_study_labels(
                    payload,
                    pilot_name=well_name,
                    study_points=pilot_points,
                    color=str(well.color),
                )
            payload["labels"].append(_well_name_label(well_name, well.t3, str(well.color)))

    if aggregated_reference_wells:
        _append_combined_reference_wells(
            payload,
            reference_wells=aggregated_reference_wells,
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
        )
    _append_reference_legend(payload, reference_wells)
    analysis_reference_wells = _analysis_reference_wells(analysis)
    if str(render_mode).strip() != WT_3D_RENDER_FAST:
        _append_reference_name_labels(payload, analysis_reference_wells)
    _append_reference_pad_labels(payload, analysis_reference_wells)

    for corridor in analysis.corridors:
        overlap_rings = _valid_overlap_rings(corridor.overlap_rings_xyz)
        if not overlap_rings:
            continue
        stacked = np.vstack(overlap_rings)
        _append_arrays(x_arrays, y_arrays, z_arrays, stacked)
        if (not focus_set) or bool({str(corridor.well_a), str(corridor.well_b)} & focus_set):
            _append_arrays(x_focus_arrays, y_focus_arrays, z_focus_arrays, stacked)

    segment_legend_added = False
    for segment in analysis.well_segments:
        well = well_lookup.get(str(segment.well_name))
        if well is None:
            continue
        interval_points = _trajectory_interval_points(
            well.stations,
            md_start_m=float(segment.md_start_m),
            md_end_m=float(segment.md_end_m),
        )
        if interval_points is None:
            continue
        x_segment = interval_points["X_m"].to_numpy(dtype=float)
        y_segment = interval_points["Y_m"].to_numpy(dtype=float)
        z_segment = interval_points["Z_m"].to_numpy(dtype=float)
        md_segment = interval_points["MD_m"].to_numpy(dtype=float)
        dls_segment = _interp_optional_column(
            well.stations,
            md_values=md_segment,
            column="DLS_deg_per_30m",
            default=0.0,
        )
        inc_segment = _interp_optional_column(
            well.stations,
            md_values=md_segment,
            column="INC_deg",
            default=0.0,
        )
        segment_points = _points_from_xyz_arrays(x_segment, y_segment, z_segment)
        _append_line_segments(
            payload,
            segments=[segment_points],
            name="Конфликтный участок ствола",
            color="#C62828",
            opacity=1.0,
            dash="solid",
            role="conflict_segment",
        )
        payload["points"].append(
            {
                "name": "Конфликтный участок ствола",
                "points": segment_points,
                "color": "#C62828",
                "opacity": 0.001,
                "size": 8.5,
                "symbol": "circle",
                "hover": [
                    {
                        "name": str(segment.well_name),
                        "md": float(md_value),
                        "dls": float(dls_value),
                        "inc": float(inc_value),
                        "segment": "Конфликтный участок",
                    }
                    for md_value, dls_value, inc_value in zip(
                        md_segment,
                        dls_segment,
                        inc_segment,
                        strict=False,
                    )
                ],
                "hover_only": True,
                "role": "conflict_hover",
            }
        )
        if not segment_legend_added:
            _append_unique_legend_item(
                payload,
                label="Конфликтный участок ствола",
                color="#C62828",
                opacity=1.0,
            )
            segment_legend_added = True
        _append_arrays(x_arrays, y_arrays, z_arrays, np.column_stack([x_segment, y_segment, z_segment]))
        if (not focus_set) or str(segment.well_name) in focus_set:
            _append_arrays(
                x_focus_arrays,
                y_focus_arrays,
                z_focus_arrays,
                np.column_stack([x_segment, y_segment, z_segment]),
            )

    payload["bounds"] = _bounds_from_arrays(
        x_arrays=x_focus_arrays or x_arrays,
        y_arrays=y_focus_arrays or y_arrays,
        z_arrays=z_focus_arrays or z_arrays,
    )
    overlap_volumes = ptc_three_overrides.overlap_volume_payloads(analysis)
    optimized = ptc_three_payload.optimize_three_payload(payload)
    if overlap_volumes:
        optimized["meshes"] = [*list(optimized.get("meshes") or []), *overlap_volumes]
        _append_unique_legend_item(
            optimized,
            label="Зоны пересечений",
            color="#C62828",
            opacity=0.34,
        )
    return optimized


def _base_payload(*, title: str) -> dict[str, object]:
    return {
        "background": THREE_BACKGROUND,
        "title": title,
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [1000.0, 1000.0, 1000.0]},
        "camera": DEFAULT_THREE_CAMERA,
        "lines": [],
        "meshes": [],
        "points": [],
        "labels": [],
        "legend": [],
    }


def _append_station_line(
    payload: dict[str, object],
    *,
    stations: pd.DataFrame,
    name: str,
    color: str,
    width_role: str,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
    focus_arrays: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None = None,
    dash: str = "solid",
    opacity: float = 1.0,
    hover_name: str | None = None,
    hover_role: str = "trajectory_hover",
    show_hover: bool = True,
) -> None:
    if stations.empty or not {"X_m", "Y_m", "Z_m"}.issubset(stations.columns):
        return
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    z_values = stations["Z_m"].to_numpy(dtype=float)
    segments = _split_finite_segments(x_values=x_values, y_values=y_values, z_values=z_values)
    if segments:
        _append_line_segments(
            payload,
            segments=segments,
            name=name,
            color=color,
            opacity=opacity,
            dash=dash,
            role=width_role,
        )
    _append_arrays_from_xyz(x_arrays, y_arrays, z_arrays, x_values, y_values, z_values)
    if focus_arrays is not None:
        _append_arrays_from_xyz(*focus_arrays, x_values, y_values, z_values)
    if show_hover:
        _append_station_hover_points(
            payload,
            stations=stations,
            name=hover_name or name,
            color=color,
            hover_role=hover_role,
        )


def _append_line_segments(
    payload: dict[str, object],
    *,
    segments: list[list[list[float]]],
    name: str,
    color: str,
    opacity: float,
    dash: str,
    role: str,
) -> None:
    if not segments:
        return
    payload["lines"].append(
        {
            "name": str(name),
            "segments": segments,
            "color": str(color),
            "opacity": float(opacity),
            "dash": str(dash or "solid"),
            "role": str(role or "line"),
        }
    )


def _append_mesh(
    payload: dict[str, object],
    *,
    vertices_xyz: np.ndarray,
    faces_ijk: np.ndarray,
    name: str,
    color: str,
    opacity: float,
    role: str,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
    focus_arrays: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None = None,
) -> None:
    vertices = np.asarray(vertices_xyz, dtype=float)
    faces = np.asarray(faces_ijk, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return
    payload["meshes"].append(
        {
            "name": str(name),
            "vertices": _points_from_xyz_array(vertices),
            "faces": [[int(a), int(b), int(c)] for a, b, c in faces.tolist()],
            "color": str(color),
            "opacity": float(opacity),
            "role": str(role or "mesh"),
        }
    )
    _append_arrays(x_arrays, y_arrays, z_arrays, vertices)
    if focus_arrays is not None:
        _append_arrays(*focus_arrays, vertices)


def _append_uncertainty_overlay(
    payload: dict[str, object],
    *,
    overlay: WellUncertaintyOverlay,
    color: str,
    terminal_boundary_color: str,
    legend_label: str,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
) -> None:
    tube_mesh = build_uncertainty_tube_mesh(overlay)
    if tube_mesh is not None:
        _append_mesh(
            payload,
            vertices_xyz=tube_mesh.vertices_xyz,
            faces_ijk=np.column_stack([tube_mesh.i, tube_mesh.j, tube_mesh.k]),
            name=legend_label,
            color=color,
            opacity=0.14,
            role="cone",
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
        )
        _append_unique_legend_item(payload, label=legend_label, color=color, opacity=0.14)
    if overlay.samples:
        terminal_ring = np.asarray(overlay.samples[-1].ring_xyz, dtype=float)
        _append_line_segments(
            payload,
            segments=[_points_from_xyz_array(terminal_ring)],
            name="Граница конуса неопределенности",
            color=terminal_boundary_color,
            opacity=0.72,
            dash="solid",
            role="cone_tip",
        )
        _append_arrays(x_arrays, y_arrays, z_arrays, terminal_ring)


def _append_target_markers(
    payload: dict[str, object],
    *,
    name: str,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    color: str,
    size: float,
    symbol: str,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
    focus_arrays: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None = None,
    hover_extra: Mapping[str, str] | None = None,
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = (),
    target_points: tuple[Point3D, ...] = (),
    target_labels: tuple[str, ...] = (),
) -> None:
    if target_points:
        points = [_point3d_payload(point) for point in target_points]
        labels = _target_labels_for_points(
            point_count=len(points),
            target_labels=target_labels,
            target_pairs=target_pairs,
        )
    else:
        points, labels = _target_marker_points_and_labels(
            surface=surface,
            t1=t1,
            t3=t3,
            target_pairs=target_pairs,
        )
    _append_arrays_for_points(x_arrays, y_arrays, z_arrays, points)
    if focus_arrays is not None:
        _append_arrays_for_points(*focus_arrays, points)
    payload["points"].append(
        {
            "name": str(name),
            "points": points,
            "color": str(color),
            "opacity": 1.0,
            "size": float(size),
            "symbol": str(symbol),
            "hover": [
                _target_hover_item(label, name, hover_extra=hover_extra)
                for label in labels
            ],
            "role": "marker",
        }
    )


def _target_marker_points_and_labels(
    *,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = (),
) -> tuple[list[list[float]], list[str]]:
    pairs = tuple(target_pairs or ())
    if len(pairs) <= 1:
        return (
            [_point3d_payload(surface), _point3d_payload(t1), _point3d_payload(t3)],
            ["S", "t1", "t3"],
        )
    points = [_point3d_payload(surface)]
    labels = ["S"]
    for index, (pair_t1, pair_t3) in enumerate(pairs, start=1):
        points.append(_point3d_payload(pair_t1))
        labels.append(f"{index}_t1")
        points.append(_point3d_payload(pair_t3))
        labels.append(f"{index}_t3")
    return points, labels


def _target_labels_for_points(
    *,
    point_count: int,
    target_labels: tuple[str, ...],
    target_pairs: tuple[tuple[Point3D, Point3D], ...] = (),
) -> list[str]:
    if len(target_labels) == point_count:
        return [str(label) for label in target_labels]
    if len(target_pairs) > 1 and point_count == 1 + 2 * len(target_pairs):
        labels = ["S"]
        for index in range(1, len(target_pairs) + 1):
            labels.extend([f"{index}_t1", f"{index}_t3"])
        return labels
    if point_count == 2:
        return ["t1", "t3"]
    if point_count == 3:
        return ["S", "t1", "t3"]
    return ["S", *(f"P{index}" for index in range(1, point_count))]


def _append_pilot_study_markers(
    payload: dict[str, object],
    *,
    pilot_name: str | None,
    study_points: tuple[Point3D, ...],
    color: str,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
) -> None:
    if not pilot_name or not study_points:
        return
    labels = _pilot_study_label_texts(str(pilot_name), len(study_points))
    points = [_point3d_payload(point) for point in study_points]
    _append_arrays_for_points(x_arrays, y_arrays, z_arrays, points)
    payload["points"].append(
        {
            "name": f"{str(pilot_name)}: точки пилота",
            "points": points,
            "color": str(color),
            "opacity": 1.0,
            "size": 6.0,
            "symbol": "diamond",
            "hover": [
                {"name": f"{str(pilot_name)}: точки пилота", "point": label}
                for label in labels
            ],
            "role": "marker",
        }
    )
    _append_pilot_study_labels(
        payload,
        pilot_name=str(pilot_name),
        study_points=study_points,
        color=color,
    )


def _append_pilot_study_labels(
    payload: dict[str, object],
    *,
    pilot_name: str,
    study_points: tuple[Point3D, ...],
    color: str,
) -> None:
    if not pilot_name or not study_points:
        return
    labels = _pilot_study_label_texts(str(pilot_name), len(study_points))
    payload["labels"].extend(
        _pilot_point_label(label, point, color)
        for label, point in zip(labels, study_points, strict=False)
    )


def _pilot_study_label_texts(pilot_name: str, point_count: int) -> tuple[str, ...]:
    return tuple(
        f"{str(pilot_name)}: {index}"
        for index in range(1, int(max(point_count, 0)) + 1)
    )


def _pilot_study_points_for_well(
    well_name: str,
    pilot_study_points_by_name: Mapping[str, tuple[Point3D, ...]],
    *,
    fallback_points: Iterable[Point3D | None],
) -> tuple[Point3D, ...]:
    mapped_points = tuple(pilot_study_points_by_name.get(str(well_name), ()) or ())
    if mapped_points:
        return mapped_points
    return _unique_points(fallback_points)


def _unique_points(points: Iterable[Point3D | None]) -> tuple[Point3D, ...]:
    result: list[Point3D] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        if point is None:
            continue
        key = (
            round(float(point.x), 9),
            round(float(point.y), 9),
            round(float(point.z), 9),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(point)
    return tuple(result)


def _append_single_well_zero_axes(
    payload: dict[str, object],
    *,
    surface: Point3D,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
) -> None:
    bounds = _bounds_from_arrays(x_arrays=x_arrays, y_arrays=y_arrays, z_arrays=z_arrays)
    x_min, y_min, z_min = bounds["min"]
    x_max, y_max, z_max = bounds["max"]
    _ = z_min, z_max
    z_zero_ref = float(surface.z)
    zero_axis_color = "#111111"
    if y_min <= 0.0 <= y_max:
        _append_line_segments(
            payload,
            segments=[[[0.0, float(y_min), z_zero_ref], [0.0, float(y_max), z_zero_ref]]],
            name="Y=0 axis",
            color=zero_axis_color,
            opacity=0.7,
            dash="solid",
            role="axis",
        )
    if x_min <= 0.0 <= x_max:
        _append_line_segments(
            payload,
            segments=[[[float(x_min), 0.0, z_zero_ref], [float(x_max), 0.0, z_zero_ref]]],
            name="X=0 axis",
            color=zero_axis_color,
            opacity=0.7,
            dash="solid",
            role="axis",
        )


def _append_reference_wells(
    payload: dict[str, object],
    *,
    reference_wells: Iterable[object],
    render_mode: str,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
) -> None:
    reference_tuple = tuple(reference_wells)
    if str(render_mode).strip() == WT_3D_RENDER_FAST:
        _append_combined_reference_wells(
            payload,
            reference_wells=reference_tuple,
            x_arrays=x_arrays,
            y_arrays=y_arrays,
            z_arrays=z_arrays,
        )
    else:
        for reference_well in reference_tuple:
            stations = getattr(reference_well, "stations")
            if stations.empty:
                continue
            kind = _reference_kind_value(reference_well)
            color = REFERENCE_WELL_KIND_COLORS.get(kind, "#A0A0A0")
            _append_station_line(
                payload,
                stations=stations,
                name=reference_well_display_label(reference_well),
                color=color,
                width_role="line",
                hover_name=_reference_hover_name(reference_well),
                hover_role="reference_hover",
                x_arrays=x_arrays,
                y_arrays=y_arrays,
                z_arrays=z_arrays,
            )
    _append_reference_legend(payload, reference_tuple)
    if str(render_mode).strip() != WT_3D_RENDER_FAST:
        _append_reference_name_labels(payload, reference_tuple)
    _append_reference_pad_labels(payload, reference_tuple)


def _append_combined_reference_wells(
    payload: dict[str, object],
    *,
    reference_wells: Iterable[object],
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
) -> None:
    reference_tuple = tuple(reference_wells)
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        label = REFERENCE_WELL_KIND_LABELS.get(str(kind), str(kind))
        matching = [
            well
            for well in reference_tuple
            if _reference_kind_value(well) == str(kind)
            and not getattr(well, "stations").empty
        ]
        if not matching:
            continue
        segments: list[list[list[float]]] = []
        for well in matching:
            stations = _decimated_station_frame(
                getattr(well, "stations"),
                target_points=WT_3D_FAST_REFERENCE_TARGET_POINTS,
            )
            x_values = stations["X_m"].to_numpy(dtype=float)
            y_values = stations["Y_m"].to_numpy(dtype=float)
            z_values = stations["Z_m"].to_numpy(dtype=float)
            segments.extend(
                _split_finite_segments(
                    x_values=x_values,
                    y_values=y_values,
                    z_values=z_values,
                )
            )
            _append_arrays_from_xyz(
                x_arrays,
                y_arrays,
                z_arrays,
                x_values,
                y_values,
                z_values,
            )
            _append_station_hover_points(
                payload,
                stations=stations,
                name=_reference_hover_name(well),
                color=REFERENCE_WELL_KIND_COLORS.get(str(kind), "#A0A0A0"),
                hover_role="reference_hover",
                payload_name=f"{label}: hover",
            )
        _append_line_segments(
            payload,
            segments=segments,
            name=f"{label} (сводно)",
            color=REFERENCE_WELL_KIND_COLORS.get(str(kind), "#A0A0A0"),
            opacity=1.0,
            dash="solid",
            role="line",
        )


def _append_station_hover_points(
    payload: dict[str, object],
    *,
    stations: pd.DataFrame,
    name: str,
    color: str,
    hover_role: str,
    payload_name: str | None = None,
) -> None:
    if stations.empty or not {"X_m", "Y_m", "Z_m"}.issubset(stations.columns):
        return
    hover_points = _points_from_xyz_arrays(
        stations["X_m"].to_numpy(dtype=float),
        stations["Y_m"].to_numpy(dtype=float),
        stations["Z_m"].to_numpy(dtype=float),
    )
    hover_items = _hover_items_from_stations(
        stations,
        fallback_name=str(name),
    )
    if not hover_points or not hover_items:
        return
    max_points = (
        WT_THREE_MAX_HOVER_POINTS_PER_REFERENCE_TRACE
        if hover_role == "reference_hover"
        else WT_THREE_MAX_HOVER_POINTS_PER_TRACE
    )
    hover_points, hover_items = ptc_three_payload.decimate_hover_payload(
        points=hover_points,
        hover_items=hover_items,
        max_points=max_points,
    )
    payload["points"].append(
        {
            "name": str(payload_name or name),
            "points": hover_points,
            "color": color,
            "opacity": 0.001,
            "size": 8.5,
            "symbol": "circle",
            "hover": hover_items,
            "hover_only": True,
            "role": hover_role,
        }
    )


def _append_reference_legend(payload: dict[str, object], reference_wells: Iterable[object]) -> None:
    for kind in _reference_kinds_present(reference_wells):
        _append_unique_legend_item(
            payload,
            label=_reference_legend_label(kind),
            color=_reference_legend_color(kind),
            opacity=1.0,
        )


def _append_reference_name_labels(
    payload: dict[str, object],
    reference_wells: Iterable[object],
) -> None:
    for kind in (REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED):
        for well in reference_wells:
            if _reference_kind_value(well) != str(kind):
                continue
            anchor = _reference_label_anchor_point(well)
            if anchor is None:
                continue
            payload["labels"].append(
                {
                    "text": str(getattr(well, "name")),
                    "position": [float(anchor[0]), float(anchor[1]), float(anchor[2])],
                    "color": _reference_label_color(str(kind)),
                    "role": "reference_label",
                }
            )


def _append_reference_pad_labels(
    payload: dict[str, object],
    reference_wells: Iterable[object],
) -> None:
    for item in _reference_pad_labels(reference_wells):
        payload["labels"].append(
            {
                "text": str(item.label),
                "position": [float(item.x), float(item.y), float(item.z)],
                "color": REFERENCE_PAD_LABEL_COLOR,
                "role": "reference_pad_label",
            }
        )


def _append_unique_legend_item(
    payload: dict[str, object],
    *,
    label: str,
    color: str,
    opacity: float,
    symbol: str = "line",
) -> None:
    label_text = str(label).strip()
    if not label_text:
        return
    legend = list(payload.get("legend") or [])
    if any(str(item.get("label", "")).strip() == label_text for item in legend):
        return
    legend.append(
        {
            "label": label_text,
            "color": str(color),
            "opacity": float(opacity),
            "symbol": str(symbol or "line"),
        }
    )
    payload["legend"] = legend


def _hover_items_from_stations(
    stations: pd.DataFrame,
    *,
    fallback_name: str,
) -> list[dict[str, object]]:
    if stations.empty:
        return []
    md_values = _numeric_column_or_nan(stations, "MD_m")
    dls_values = _numeric_column_or_nan(stations, "DLS_deg_per_30m")
    inc_values = _numeric_column_or_nan(stations, "INC_deg")
    segment_values = (
        stations["segment"].astype(str).to_numpy(dtype=object)
        if "segment" in stations.columns
        else np.full(len(stations), "", dtype=object)
    )
    items: list[dict[str, object]] = []
    for md_value, dls_value, inc_value, segment_value in zip(
        md_values,
        dls_values,
        inc_values,
        segment_values,
        strict=False,
    ):
        item: dict[str, object] = {"name": str(fallback_name)}
        if np.isfinite(float(md_value)):
            item["md"] = float(md_value)
        if np.isfinite(float(dls_value)):
            item["dls"] = float(dls_value)
        if np.isfinite(float(inc_value)):
            item["inc"] = float(inc_value)
        if str(segment_value).strip():
            item["segment"] = str(segment_value).strip()
        items.append(item)
    return items


def _target_hover_item(
    point_name: str,
    label: str,
    *,
    hover_extra: Mapping[str, str] | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {"name": str(label), "point": str(point_name)}
    if hover_extra:
        status = str(hover_extra.get("status", "") or "").strip()
        problem = str(hover_extra.get("problem", "") or "").strip()
        if status:
            item["segment"] = status
        if problem:
            item["problem"] = problem
    return item


def _calculated_t1_t3_points(stations: pd.DataFrame, *, md_t1_m: float) -> list[list[float]]:
    if stations.empty or not {"MD_m", "X_m", "Y_m", "Z_m"}.issubset(stations.columns):
        return []
    t1_idx = int((stations["MD_m"] - float(md_t1_m)).abs().idxmin())
    rows = [stations.loc[t1_idx], stations.iloc[-1]]
    return [
        [float(row["X_m"]), float(row["Y_m"]), float(row["Z_m"])]
        for row in rows
    ]


def _station_point_at_md(stations: pd.DataFrame, md_m: float) -> list[float] | None:
    if stations.empty or not {"MD_m", "X_m", "Y_m", "Z_m"}.issubset(stations.columns):
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    coords = stations[["X_m", "Y_m", "Z_m"]].to_numpy(dtype=float)
    valid_mask = np.isfinite(md_values) & np.all(np.isfinite(coords), axis=1)
    if not np.any(valid_mask):
        return None
    md_values = md_values[valid_mask]
    coords = coords[valid_mask]
    unique_md, unique_indices = np.unique(md_values, return_index=True)
    coords = coords[unique_indices]
    if len(unique_md) == 0:
        return None
    if len(unique_md) == 1:
        row = coords[0]
        return [float(row[0]), float(row[1]), float(row[2])]
    target_md = float(np.clip(float(md_m), float(unique_md[0]), float(unique_md[-1])))
    return [
        float(np.interp(target_md, unique_md, coords[:, axis]))
        for axis in range(3)
    ]


def _trajectory_interval_points(
    stations: pd.DataFrame,
    *,
    md_start_m: float,
    md_end_m: float,
) -> pd.DataFrame | None:
    if len(stations) == 0:
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    if len(md_values) == 0:
        return None
    start_md = float(np.clip(md_start_m, float(md_values[0]), float(md_values[-1])))
    end_md = float(np.clip(md_end_m, float(md_values[0]), float(md_values[-1])))
    if end_md <= start_md + SMALL:
        return None

    interior_mask = (md_values > start_md + SMALL) & (md_values < end_md - SMALL)
    segment_md = np.concatenate(
        [
            np.array([start_md], dtype=float),
            md_values[interior_mask],
            np.array([end_md], dtype=float),
        ]
    )
    if len(segment_md) < 2:
        return None
    return pd.DataFrame(
        {
            "MD_m": segment_md,
            "X_m": np.interp(segment_md, md_values, stations["X_m"].to_numpy(dtype=float)),
            "Y_m": np.interp(segment_md, md_values, stations["Y_m"].to_numpy(dtype=float)),
            "Z_m": np.interp(segment_md, md_values, stations["Z_m"].to_numpy(dtype=float)),
        }
    )


def _interp_optional_column(
    stations: pd.DataFrame,
    *,
    md_values: np.ndarray,
    column: str,
    default: float,
) -> np.ndarray:
    if column not in stations.columns or "MD_m" not in stations.columns:
        return np.full(len(md_values), float(default), dtype=float)
    return np.interp(
        md_values,
        stations["MD_m"].to_numpy(dtype=float),
        stations[column].fillna(float(default)).to_numpy(dtype=float),
    )


def _maybe_decimated_stations(
    stations: pd.DataFrame,
    *,
    render_mode: str,
    target_points: int,
) -> pd.DataFrame:
    if str(render_mode).strip() != WT_3D_RENDER_FAST:
        return stations
    return _decimated_station_frame(stations, target_points=target_points)


def _decimated_station_frame(stations: pd.DataFrame, *, target_points: int) -> pd.DataFrame:
    if target_points <= 2 or len(stations.index) <= target_points:
        return stations
    indices = np.linspace(0, len(stations.index) - 1, num=int(target_points), dtype=int)
    unique_indices = np.unique(indices)
    return stations.iloc[unique_indices].reset_index(drop=True)


def _bounds_from_arrays(
    *,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
) -> dict[str, list[float]]:
    x_values = np.concatenate(x_arrays) if x_arrays else np.array([0.0], dtype=float)
    y_values = np.concatenate(y_arrays) if y_arrays else np.array([0.0], dtype=float)
    z_values = np.concatenate(z_arrays) if z_arrays else np.array([0.0], dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
    if not finite_mask.any():
        x_values = np.array([0.0], dtype=float)
        y_values = np.array([0.0], dtype=float)
        z_values = np.array([0.0], dtype=float)
    else:
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]
        z_values = z_values[finite_mask]
    x_range, y_range, z_range = equalized_axis_ranges(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )
    return {
        "min": [float(x_range[0]), float(y_range[0]), float(z_range[0])],
        "max": [float(x_range[1]), float(y_range[1]), float(z_range[1])],
    }


def _split_finite_segments(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> list[list[list[float]]]:
    segments: list[list[list[float]]] = []
    current: list[list[float]] = []
    for x_value, y_value, z_value in zip(x_values, y_values, z_values, strict=False):
        if not (
            np.isfinite(float(x_value))
            and np.isfinite(float(y_value))
            and np.isfinite(float(z_value))
        ):
            if len(current) >= 2:
                segments.append(list(current))
            current = []
            continue
        current.append([float(x_value), float(y_value), float(z_value)])
    if len(current) >= 2:
        segments.append(list(current))
    return segments


def _valid_overlap_rings(raw_rings: Iterable[object]) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    for raw_ring in raw_rings:
        try:
            ring = np.asarray(raw_ring, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            ring.ndim == 2
            and ring.shape[1] == 3
            and len(ring) >= 3
            and np.all(np.isfinite(ring))
        ):
            rings.append(ring)
    return rings


def _append_arrays(
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
    xyz: np.ndarray,
) -> None:
    if xyz.size == 0:
        return
    x_arrays.append(np.asarray(xyz[:, 0], dtype=float))
    y_arrays.append(np.asarray(xyz[:, 1], dtype=float))
    z_arrays.append(np.asarray(xyz[:, 2], dtype=float))


def _append_arrays_from_xyz(
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> None:
    x_arrays.append(np.asarray(x_values, dtype=float))
    y_arrays.append(np.asarray(y_values, dtype=float))
    z_arrays.append(np.asarray(z_values, dtype=float))


def _append_arrays_for_points(
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    z_arrays: list[np.ndarray],
    points: list[list[float]],
) -> None:
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        return
    _append_arrays(x_arrays, y_arrays, z_arrays, xyz)


def _points_from_xyz_arrays(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> list[list[float]]:
    return [
        [float(x_value), float(y_value), float(z_value)]
        for x_value, y_value, z_value in zip(x_values, y_values, z_values, strict=False)
        if (
            np.isfinite(float(x_value))
            and np.isfinite(float(y_value))
            and np.isfinite(float(z_value))
        )
    ]


def _points_from_xyz_array(xyz: np.ndarray) -> list[list[float]]:
    return [
        [float(row[0]), float(row[1]), float(row[2])]
        for row in np.asarray(xyz, dtype=float)
        if np.all(np.isfinite(row))
    ]


def _numeric_column_or_nan(stations: pd.DataFrame, column: str) -> np.ndarray:
    if column not in stations.columns:
        return np.full(len(stations), np.nan, dtype=float)
    return stations[column].to_numpy(dtype=float)


def _point3d_payload(point: Point3D) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def _well_name_label(well_name: str, point: Point3D, color: str) -> dict[str, object]:
    return {
        "text": str(well_name),
        "position": [float(point.x), float(point.y), float(point.z)],
        "color": str(color),
        "role": "well_label",
    }


def _pilot_point_label(text: str, point: Point3D, color: str) -> dict[str, object]:
    return {
        "text": str(text),
        "position": [float(point.x), float(point.y), float(point.z)],
        "color": str(color),
        "role": "pilot_point_label",
    }


def _point_payload_label(
    text: str,
    point: list[float],
    color: str,
    *,
    role: str,
) -> dict[str, object]:
    return {
        "text": str(text),
        "position": [float(point[0]), float(point[1]), float(point[2])],
        "color": str(color),
        "role": str(role),
    }


def _clean_name_set(names: Iterable[object]) -> set[str]:
    return {str(name).strip() for name in names if str(name).strip()}


def _include_name_in_focus(name: str, focus_set: set[str]) -> bool:
    return not focus_set or str(name) in focus_set


def _fallback_color(index: int, fallback_color: Callable[[int], str] | None) -> str:
    if fallback_color is not None:
        return str(fallback_color(index))
    palette = (
        "#15D562",
        "#1562D5",
        "#F59E0B",
        "#AF15D5",
        "#00A0A0",
        "#D54A9A",
        "#62D515",
        "#8A5CF6",
    )
    return palette[int(index) % len(palette)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(90, 90, 90, {float(alpha):.3f})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {float(alpha):.3f})"


def _lighten_hex(hex_color: str, blend_with_white: float = 0.38) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return "#A0A0A0"
    blend = float(np.clip(blend_with_white, 0.0, 1.0))
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    red = int(round(red + (255 - red) * blend))
    green = int(round(green + (255 - green) * blend))
    blue = int(round(blue + (255 - blue) * blend))
    return f"#{red:02X}{green:02X}{blue:02X}"


def _analysis_well_label(well: object) -> str:
    if not bool(getattr(well, "is_reference_only", False)):
        return str(getattr(well, "name"))
    kind = str(getattr(well, "well_kind", ""))
    return (
        f"{getattr(well, 'name')} "
        f"({REFERENCE_WELL_KIND_LABELS.get(kind, kind)})"
    )


def _analysis_reference_wells(analysis: AntiCollisionAnalysis) -> list[ImportedTrajectoryWell]:
    return [
        ImportedTrajectoryWell(
            name=str(well.name),
            kind=str(well.well_kind),
            stations=well.stations,
            surface=Point3D(
                x=float(well.surface.x),
                y=float(well.surface.y),
                z=float(well.surface.z),
            ),
            azimuth_deg=0.0,
        )
        for well in analysis.wells
        if bool(well.is_reference_only)
    ]


def _anticollision_reference_cone_focus_names(
    analysis: AntiCollisionAnalysis,
) -> set[str]:
    focus_names = _anticollision_focus_reference_names(analysis)
    calculated_bounds = [
        bounds
        for well in analysis.wells
        if not bool(well.is_reference_only)
        for bounds in [_xy_bounds_from_stations(well.stations)]
        if bounds is not None
    ]
    if not calculated_bounds:
        return focus_names
    for well in analysis.wells:
        if not bool(well.is_reference_only):
            continue
        reference_bounds = _xy_bounds_from_stations(well.stations)
        if reference_bounds is None:
            continue
        if any(_xy_bounds_gap_m(reference_bounds, item) <= 500.0 for item in calculated_bounds):
            focus_names.add(str(well.name))
    return focus_names


def _anticollision_focus_reference_names(analysis: AntiCollisionAnalysis) -> set[str]:
    reference_names = {
        str(well.name) for well in analysis.wells if bool(well.is_reference_only)
    }
    focus_names: set[str] = set()
    for corridor in analysis.corridors:
        if str(corridor.well_a) in reference_names:
            focus_names.add(str(corridor.well_a))
        if str(corridor.well_b) in reference_names:
            focus_names.add(str(corridor.well_b))
    return focus_names


def _xy_bounds_from_stations(stations: pd.DataFrame) -> tuple[float, float, float, float] | None:
    if stations.empty or not {"X_m", "Y_m"}.issubset(stations.columns):
        return None
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not finite_mask.any():
        return None
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    return (
        float(np.min(x_values)),
        float(np.max(x_values)),
        float(np.min(y_values)),
        float(np.max(y_values)),
    )


def _xy_bounds_gap_m(
    bounds_a: tuple[float, float, float, float],
    bounds_b: tuple[float, float, float, float],
) -> float:
    min_x_a, max_x_a, min_y_a, max_y_a = bounds_a
    min_x_b, max_x_b, min_y_b, max_y_b = bounds_b
    dx = max(min_x_a - max_x_b, min_x_b - max_x_a, 0.0)
    dy = max(min_y_a - max_y_b, min_y_b - max_y_a, 0.0)
    return float(np.hypot(dx, dy))


def _reference_label_anchor_point(reference_well: object) -> tuple[float, float, float] | None:
    stations = getattr(reference_well, "stations")
    required = {"MD_m", "INC_deg", "X_m", "Y_m", "Z_m"}
    if stations.empty or not required.issubset(stations.columns):
        return None
    md_values = stations["MD_m"].to_numpy(dtype=float)
    inc_values = stations["INC_deg"].to_numpy(dtype=float)
    x_values = stations["X_m"].to_numpy(dtype=float)
    y_values = stations["Y_m"].to_numpy(dtype=float)
    z_values = stations["Z_m"].to_numpy(dtype=float)
    if len(md_values) == 0:
        return None
    high_angle_mask = inc_values >= float(REFERENCE_LABEL_HORIZONTAL_INC_THRESHOLD_DEG)
    if bool(high_angle_mask[-1]):
        start_index = len(high_angle_mask) - 1
        while start_index > 0 and bool(high_angle_mask[start_index - 1]):
            start_index -= 1
        if float(md_values[-1] - md_values[start_index]) >= float(
            REFERENCE_LABEL_HORIZONTAL_MIN_INTERVAL_M
        ):
            return (
                float(x_values[start_index]),
                float(y_values[start_index]),
                float(z_values[start_index]),
            )
    return (float(x_values[-1]), float(y_values[-1]), float(z_values[-1]))


def _reference_kind_value(reference_well: object) -> str:
    return str(
        getattr(reference_well, "kind", getattr(reference_well, "well_kind", ""))
    )


def _reference_kinds_present(reference_wells: Iterable[object]) -> tuple[str, ...]:
    kinds: list[str] = []
    seen: set[str] = set()
    for reference_well in reference_wells:
        kind = _reference_kind_value(reference_well)
        if not kind or kind in seen:
            continue
        seen.add(kind)
        kinds.append(kind)
    return tuple(kinds)


def _reference_legend_label(kind: str) -> str:
    if str(kind) == REFERENCE_WELL_APPROVED:
        return "Проектные утвержденные скважины"
    return "Фактические скважины"


def _reference_legend_color(kind: str) -> str:
    return str(REFERENCE_WELL_KIND_COLORS.get(str(kind), "#A0A0A0"))


def _reference_label_color(kind: str) -> str:
    if str(kind) == REFERENCE_WELL_APPROVED:
        return REFERENCE_LABEL_APPROVED_COLOR
    return REFERENCE_LABEL_ACTUAL_COLOR


def _reference_hover_name(reference_well: object) -> str:
    name = str(getattr(reference_well, "name", "")).strip()
    if name:
        return name
    return reference_well_display_label(reference_well)


def _reference_pad_numeric_id(well_name: str) -> str | None:
    digits: list[str] = []
    for char in str(well_name).strip():
        if char.isdigit():
            digits.append(char)
            continue
        break
    if not digits:
        return None
    digit_text = "".join(digits)
    if len(digit_text) > 2:
        digit_text = digit_text[:-2]
    normalized = digit_text.lstrip("0")
    return normalized or "0"


def _reference_pad_labels(reference_wells: Iterable[object]) -> list[_ReferencePadLabel]:
    ordered_wells = [
        well
        for well in reference_wells
        if np.isfinite(float(getattr(well, "surface").x))
        and np.isfinite(float(getattr(well, "surface").y))
    ]
    if not ordered_wells:
        return []

    cell_size = float(REFERENCE_PAD_GROUP_DISTANCE_M)
    parents = list(range(len(ordered_wells)))
    ranks = [0] * len(ordered_wells)

    def _find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root == right_root:
            return
        if ranks[left_root] < ranks[right_root]:
            parents[left_root] = right_root
            return
        if ranks[left_root] > ranks[right_root]:
            parents[right_root] = left_root
            return
        parents[right_root] = left_root
        ranks[left_root] += 1

    cells: dict[tuple[int, int], list[int]] = {}
    for index, well in enumerate(ordered_wells):
        surface = getattr(well, "surface")
        cell = (
            int(np.floor(float(surface.x) / cell_size)),
            int(np.floor(float(surface.y) / cell_size)),
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for candidate_index in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    candidate_surface = getattr(ordered_wells[candidate_index], "surface")
                    if float(
                        np.hypot(
                            float(surface.x) - float(candidate_surface.x),
                            float(surface.y) - float(candidate_surface.y),
                        )
                    ) <= float(REFERENCE_PAD_GROUP_DISTANCE_M):
                        _union(index, candidate_index)
        cells.setdefault(cell, []).append(index)

    component_indices: dict[int, list[int]] = {}
    for index in range(len(ordered_wells)):
        component_indices.setdefault(_find(index), []).append(index)

    pad_labels: list[_ReferencePadLabel] = []
    fallback_counter = 1
    for component in component_indices.values():
        ordered_component = sorted(component)
        anchor_well = ordered_wells[ordered_component[0]]
        numeric_ids = [
            pad_id
            for index in ordered_component
            for pad_id in [_reference_pad_numeric_id(str(getattr(ordered_wells[index], "name")))]
            if pad_id is not None
        ]
        if numeric_ids:
            counts = {pad_id: numeric_ids.count(pad_id) for pad_id in set(numeric_ids)}
            pad_id = sorted(
                counts.keys(),
                key=lambda value: (-counts[value], numeric_ids.index(value), value),
            )[0]
        else:
            if len(ordered_component) <= 1:
                continue
            pad_id = str(fallback_counter)
            fallback_counter += 1
        surface = getattr(anchor_well, "surface")
        pad_labels.append(
            _ReferencePadLabel(
                label=f"Куст {pad_id}",
                x=float(surface.x),
                y=float(surface.y),
                z=float(surface.z),
            )
        )
    return pad_labels
