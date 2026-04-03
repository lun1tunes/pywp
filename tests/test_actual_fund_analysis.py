from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

import pywp.actual_fund_analysis as actual_fund_analysis_module
from pywp.actual_fund_analysis import (
    ActualFundWellMetrics,
    CALIBRATION_STATUS_READY,
    _reconstruct_actual_survey,
    actual_well_family_name,
    actual_well_is_horizontal,
    actual_well_pad_group,
    build_actual_fund_kop_depth_function,
    build_actual_fund_well_analysis,
    build_actual_fund_well_metrics,
    calibrate_uncertainty_from_actual_fund,
    summarize_actual_fund_by_depth,
)
from pywp.mcm import compute_positions_min_curv
from pywp.models import Point3D
from pywp.reference_trajectories import parse_reference_trajectory_table
from pywp.uncertainty import DEFAULT_UNCERTAINTY_PRESET, DEFAULT_PLANNING_UNCERTAINTY_MODEL


def _actual_wells():
    return tuple(
        parse_reference_trajectory_table(
            [
                {"Wellname": "7401", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "7401", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 1000.0, "MD": 1000.0},
                {"Wellname": "7401", "Type": "actual", "X": 1000.0, "Y": 0.0, "Z": 1000.0, "MD": 2000.0},
                {"Wellname": "7401", "Type": "actual", "X": 2000.0, "Y": 0.0, "Z": 1000.0, "MD": 3000.0},
                {"Wellname": "7401_PL", "Type": "actual", "X": 0.0, "Y": 20.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "7401_PL", "Type": "actual", "X": 0.0, "Y": 20.0, "Z": 1000.0, "MD": 1000.0},
                {"Wellname": "7401_PL", "Type": "actual", "X": 1000.0, "Y": 20.0, "Z": 1000.0, "MD": 2000.0},
                {"Wellname": "7401_PL", "Type": "actual", "X": 2000.0, "Y": 20.0, "Z": 1000.0, "MD": 3000.0},
                {"Wellname": "7402", "Type": "actual", "X": 0.0, "Y": 120.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "7402", "Type": "actual", "X": 0.0, "Y": 120.0, "Z": 1000.0, "MD": 1000.0},
                {"Wellname": "7402", "Type": "actual", "X": 1000.0, "Y": 120.0, "Z": 1000.0, "MD": 2000.0},
                {"Wellname": "7402", "Type": "actual", "X": 2000.0, "Y": 120.0, "Z": 1000.0, "MD": 3000.0},
            ]
        )
    )


def _synthetic_hold_actual_well(name: str = "9001") -> tuple:
    survey = pd.DataFrame(
        {
            "MD_m": [0.0, 600.0, 950.0, 1200.0, 1500.0, 1800.0, 2050.0, 2300.0, 2650.0, 3000.0],
            "INC_deg": [0.0, 0.0, 25.0, 55.0, 55.0, 55.0, 70.0, 88.0, 90.0, 90.0],
            "AZI_deg": [90.0] * 10,
        }
    )
    positioned = compute_positions_min_curv(survey, start=Point3D(0.0, 0.0, 0.0))
    rows = [
        {
            "Wellname": name,
            "Type": "actual",
            "X": float(row["X_m"]),
            "Y": float(row["Y_m"]),
            "Z": float(row["Z_m"]),
            "MD": float(row["MD_m"]),
        }
        for _, row in positioned.iterrows()
    ]
    return tuple(parse_reference_trajectory_table(rows))


def _synthetic_anomalous_horizontal_well(name: str = "9002") -> tuple:
    survey = pd.DataFrame(
        {
            "MD_m": [0.0, 500.0, 850.0, 1100.0, 1400.0, 1750.0, 2100.0, 2450.0],
            "INC_deg": [0.0, 0.0, 35.0, 65.0, 82.0, 88.0, 90.0, 90.0],
            "AZI_deg": [90.0] * 8,
        }
    )
    positioned = compute_positions_min_curv(survey, start=Point3D(0.0, 0.0, 0.0))
    rows = [
        {
            "Wellname": name,
            "Type": "actual",
            "X": float(row["X_m"]),
            "Y": float(row["Y_m"]),
            "Z": float(row["Z_m"]),
            "MD": float(row["MD_m"]),
        }
        for _, row in positioned.iterrows()
    ]
    return tuple(parse_reference_trajectory_table(rows))


def _synthetic_noisy_vertical_hold_actual_well(name: str = "9003") -> tuple:
    survey = pd.DataFrame(
        {
            "MD_m": [0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0, 2500.0, 3200.0],
            "INC_deg": [0.0, 0.8, 1.8, 4.0, 16.0, 35.0, 55.0, 55.0, 82.0, 90.0],
            "AZI_deg": [90.0] * 10,
        }
    )
    positioned = compute_positions_min_curv(survey, start=Point3D(0.0, 0.0, 0.0))
    rows = [
        {
            "Wellname": name,
            "Type": "actual",
            "X": float(row["X_m"]),
            "Y": float(row["Y_m"]),
            "Z": float(row["Z_m"]),
            "MD": float(row["MD_m"]),
        }
        for _, row in positioned.iterrows()
    ]
    return tuple(parse_reference_trajectory_table(rows))


def _synthetic_close_pair_actual_wells() -> tuple:
    return tuple(
        parse_reference_trajectory_table(
            [
                {"Wellname": "8001", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "8001", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 1000.0, "MD": 1000.0},
                {"Wellname": "8001", "Type": "actual", "X": 1000.0, "Y": 0.0, "Z": 1000.0, "MD": 2000.0},
                {"Wellname": "8001", "Type": "actual", "X": 2000.0, "Y": 0.0, "Z": 1000.0, "MD": 3000.0},
                {"Wellname": "8002", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "8002", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 1000.0, "MD": 1000.0},
                {"Wellname": "8002", "Type": "actual", "X": 1000.0, "Y": 0.0, "Z": 1000.0, "MD": 2000.0},
                {"Wellname": "8002", "Type": "actual", "X": 2000.0, "Y": 0.0, "Z": 1000.0, "MD": 3000.0},
                {"Wellname": "8003", "Type": "actual", "X": 0.0, "Y": 220.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "8003", "Type": "actual", "X": 0.0, "Y": 220.0, "Z": 1000.0, "MD": 1000.0},
                {"Wellname": "8003", "Type": "actual", "X": 1000.0, "Y": 220.0, "Z": 1000.0, "MD": 2000.0},
                {"Wellname": "8003", "Type": "actual", "X": 2000.0, "Y": 220.0, "Z": 1000.0, "MD": 3000.0},
            ]
        )
    )


def test_actual_well_family_and_pad_group_extract_numeric_prefixes() -> None:
    assert actual_well_family_name("7401_PL") == "7401"
    assert actual_well_family_name("7402_2") == "7402"
    assert actual_well_pad_group("6103") == "61"
    assert actual_well_pad_group("8210") == "82"


def test_actual_fund_metrics_detect_horizontal_wells_and_estimate_kop() -> None:
    wells = _actual_wells()
    metrics = build_actual_fund_well_metrics(wells)

    assert len(metrics) == 3
    assert all(actual_well_is_horizontal(well.stations) for well in wells)
    assert all(item.is_horizontal for item in metrics)
    assert all(item.horizontal_entry_md_m is not None for item in metrics)
    assert all(1900.0 <= float(item.horizontal_entry_md_m) <= 2050.0 for item in metrics)
    assert all(item.horizontal_length_m >= 900.0 for item in metrics)
    assert all(item.kop_md_m is not None for item in metrics)
    assert all(item.max_inc_deg >= 80.0 for item in metrics)


def test_actual_fund_metrics_reconstruct_hold_and_robust_dls_from_irregular_xyzmd() -> None:
    wells = _synthetic_hold_actual_well()

    metrics = build_actual_fund_well_metrics(wells)

    assert len(metrics) == 1
    item = metrics[0]
    assert item.is_horizontal is True
    assert item.kop_md_m is not None
    assert 500.0 <= float(item.kop_md_m) <= 950.0
    assert item.horizontal_entry_md_m is not None
    assert 2280.0 <= float(item.horizontal_entry_md_m) <= 2320.0
    assert float(item.horizontal_length_m) >= 600.0
    assert item.hold_inc_deg is not None
    assert 52.0 <= float(item.hold_inc_deg) <= 58.0
    assert float(item.hold_length_m) >= 350.0
    assert item.max_dls_deg_per_30m is not None
    assert float(item.max_dls_deg_per_30m) < 5.0
    assert item.max_build_dls_before_hold_deg_per_30m is not None
    assert float(item.max_build_dls_before_hold_deg_per_30m) < 5.0


def test_actual_fund_well_analysis_builds_segmented_view_for_selected_well() -> None:
    well = _synthetic_hold_actual_well()[0]

    analysis = build_actual_fund_well_analysis(well)

    assert analysis.metrics.horizontal_entry_md_m is not None
    assert 2280.0 <= float(analysis.metrics.horizontal_entry_md_m) <= 2320.0
    assert {"Lateral_m", "AnalysisZoneKey", "AnalysisZoneLabel"} <= set(analysis.survey.columns)
    assert [zone.zone_key for zone in analysis.zone_summaries] == [
        "vertical",
        "build1",
        "hold",
        "build2",
        "horizontal",
    ]
    horizontal_zone = analysis.zone_summaries[-1]
    assert horizontal_zone.zone_label == "Горизонтальный"
    assert float(horizontal_zone.md_from_m) == float(analysis.metrics.horizontal_entry_md_m)
    assert float(horizontal_zone.md_to_m) == float(analysis.metrics.md_total_m)


def test_actual_fund_kop_ignores_small_noisy_inc_before_real_build1() -> None:
    wells = _synthetic_noisy_vertical_hold_actual_well()

    metrics = build_actual_fund_well_metrics(wells)

    assert len(metrics) == 1
    item = metrics[0]
    assert item.kop_md_m is not None
    assert 800.0 <= float(item.kop_md_m) <= 1300.0
    assert item.hold_inc_deg is not None
    assert 50.0 <= float(item.hold_inc_deg) <= 58.0


def test_actual_fund_analysis_excludes_horizontal_anomaly_without_stable_hold() -> None:
    wells = _synthetic_anomalous_horizontal_well()

    metrics = build_actual_fund_well_metrics(wells)

    assert len(metrics) == 1
    item = metrics[0]
    assert item.is_horizontal is True
    assert item.is_analysis_eligible is False
    assert item.analysis_exclusion_reason is not None
    assert "HOLD" in str(item.analysis_exclusion_reason)


def test_reconstruct_actual_survey_respects_custom_resample_step() -> None:
    well = _synthetic_hold_actual_well()[0]

    rebuilt = _reconstruct_actual_survey(well.stations, resample_step_m=25.0)

    md_values = rebuilt["MD_m"].to_numpy(dtype=float)
    assert md_values[0] == 0.0
    assert md_values[-1] == float(well.stations["MD_m"].iloc[-1])
    assert np.allclose(np.diff(md_values[:-1]), 25.0)


def test_calibration_excludes_same_family_pairs_and_builds_custom_model() -> None:
    wells = _actual_wells()
    result = calibrate_uncertainty_from_actual_fund(
        actual_wells=wells,
        base_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        base_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert result.status == CALIBRATION_STATUS_READY
    assert result.custom_model is not None
    assert result.scale_factor is not None
    assert result.scale_factor < 1.0
    assert result.excluded_pilot_well_count == 1
    assert result.skipped_same_family_pair_count == 0
    assert result.analyzed_pair_count == 1
    assert result.overlapping_pair_count_before >= result.overlapping_pair_count_after


def test_calibration_can_ignore_single_extreme_close_pair_and_still_build_model() -> None:
    wells = _synthetic_close_pair_actual_wells()

    result = calibrate_uncertainty_from_actual_fund(
        actual_wells=wells,
        base_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        base_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert result.status == CALIBRATION_STATUS_READY
    assert result.custom_model is not None
    assert result.ignored_close_pair_count == 1
    assert result.ignored_close_pairs == ("8001 ↔ 8002",)
    assert result.overlapping_pair_count_after == 0


def test_calibration_reuses_prebuilt_analyses_without_recomputing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wells = _actual_wells()
    analyses = actual_fund_analysis_module.build_actual_fund_well_analyses(wells)
    analysis_call_count = 0

    def _fake_build_actual_fund_well_analyses(
        actual_wells: Iterable[object],
    ) -> tuple[actual_fund_analysis_module.ActualFundWellAnalysis, ...]:
        nonlocal analysis_call_count
        analysis_call_count += 1
        assert len(tuple(actual_wells)) == len(wells)
        return analyses

    def _unexpected_metrics_call(*_: object, **__: object) -> tuple[object, ...]:
        raise AssertionError("calibration should reuse metrics from prebuilt analyses")

    monkeypatch.setattr(
        actual_fund_analysis_module,
        "build_actual_fund_well_analyses",
        _fake_build_actual_fund_well_analyses,
    )
    monkeypatch.setattr(
        actual_fund_analysis_module,
        "build_actual_fund_well_metrics",
        _unexpected_metrics_call,
    )

    result = calibrate_uncertainty_from_actual_fund(
        actual_wells=wells,
        analyses=analyses,
        base_model=DEFAULT_PLANNING_UNCERTAINTY_MODEL,
        base_preset=DEFAULT_UNCERTAINTY_PRESET,
    )

    assert analysis_call_count == 0
    assert result.status == CALIBRATION_STATUS_READY


def test_depth_cluster_summary_and_kop_function_build_from_generated_dataset() -> None:
    from pathlib import Path

    from pywp.reference_trajectories import parse_reference_trajectory_welltrack_text

    wells = parse_reference_trajectory_welltrack_text(
        Path("tests/test_data/WELLTRACKS_FACT_DEPTH_CLUSTERS.INC").read_text(encoding="utf-8"),
        kind="actual",
    )
    metrics = build_actual_fund_well_metrics(wells)
    clusters = summarize_actual_fund_by_depth(metrics)
    kop_function = build_actual_fund_kop_depth_function(metrics)

    assert len(clusters) == 3
    assert kop_function is not None
    assert kop_function.mode == "piecewise_linear"
    assert tuple(round(value, 1) for value in kop_function.anchor_depths_tvd_m) == tuple(
        round(float(cluster.anchor_horizontal_entry_tvd_m), 1) for cluster in clusters
    )


def test_depth_cluster_anchor_uses_min_plus_std_after_outlier_filter() -> None:
    metrics = (
        ActualFundWellMetrics(
            name="A",
            family_name="A",
            pad_group="1",
            is_horizontal=True,
            md_total_m=0.0,
            tvd_end_m=0.0,
            lateral_departure_m=0.0,
            kop_md_m=800.0,
            kop_tvd_m=700.0,
            horizontal_entry_md_m=2000.0,
            horizontal_entry_tvd_m=1800.0,
            horizontal_length_m=1000.0,
            hold_inc_deg=45.0,
            hold_azi_deg=90.0,
            hold_length_m=300.0,
            max_inc_deg=90.0,
            max_dls_deg_per_30m=2.0,
            max_build_dls_before_hold_deg_per_30m=2.0,
            is_analysis_eligible=True,
        ),
        ActualFundWellMetrics(
            name="B",
            family_name="B",
            pad_group="1",
            is_horizontal=True,
            md_total_m=0.0,
            tvd_end_m=0.0,
            lateral_departure_m=0.0,
            kop_md_m=820.0,
            kop_tvd_m=710.0,
            horizontal_entry_md_m=2000.0,
            horizontal_entry_tvd_m=1820.0,
            horizontal_length_m=1000.0,
            hold_inc_deg=45.0,
            hold_azi_deg=90.0,
            hold_length_m=300.0,
            max_inc_deg=90.0,
            max_dls_deg_per_30m=2.0,
            max_build_dls_before_hold_deg_per_30m=2.0,
            is_analysis_eligible=True,
        ),
        ActualFundWellMetrics(
            name="C",
            family_name="C",
            pad_group="1",
            is_horizontal=True,
            md_total_m=0.0,
            tvd_end_m=0.0,
            lateral_departure_m=0.0,
            kop_md_m=840.0,
            kop_tvd_m=720.0,
            horizontal_entry_md_m=2000.0,
            horizontal_entry_tvd_m=1840.0,
            horizontal_length_m=1000.0,
            hold_inc_deg=45.0,
            hold_azi_deg=90.0,
            hold_length_m=300.0,
            max_inc_deg=90.0,
            max_dls_deg_per_30m=2.0,
            max_build_dls_before_hold_deg_per_30m=2.0,
            is_analysis_eligible=True,
        ),
        ActualFundWellMetrics(
            name="OUTLIER",
            family_name="OUTLIER",
            pad_group="1",
            is_horizontal=True,
            md_total_m=0.0,
            tvd_end_m=0.0,
            lateral_departure_m=0.0,
            kop_md_m=1400.0,
            kop_tvd_m=1300.0,
            horizontal_entry_md_m=2000.0,
            horizontal_entry_tvd_m=1950.0,
            horizontal_length_m=1000.0,
            hold_inc_deg=45.0,
            hold_azi_deg=90.0,
            hold_length_m=300.0,
            max_inc_deg=90.0,
            max_dls_deg_per_30m=2.0,
            max_build_dls_before_hold_deg_per_30m=2.0,
            is_analysis_eligible=True,
        ),
    )

    clusters = summarize_actual_fund_by_depth(metrics, relative_tolerance=0.5)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.anchor_horizontal_entry_tvd_m < 2000.0
    assert cluster.anchor_kop_md_m < 900.0
