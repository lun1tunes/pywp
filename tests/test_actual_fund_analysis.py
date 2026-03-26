from __future__ import annotations

import numpy as np
import pandas as pd

from pywp.actual_fund_analysis import (
    CALIBRATION_STATUS_READY,
    _reconstruct_actual_survey,
    actual_well_family_name,
    actual_well_is_horizontal,
    actual_well_pad_group,
    build_actual_fund_well_metrics,
    calibrate_uncertainty_from_actual_fund,
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
    assert all(item.horizontal_length_m >= 1000.0 for item in metrics)
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
    assert float(item.horizontal_entry_md_m) >= 2200.0
    assert float(item.horizontal_length_m) >= 600.0
    assert item.hold_inc_deg is not None
    assert 52.0 <= float(item.hold_inc_deg) <= 58.0
    assert float(item.hold_length_m) >= 350.0
    assert item.max_dls_deg_per_30m is not None
    assert float(item.max_dls_deg_per_30m) < 5.0
    assert item.max_build_dls_before_hold_deg_per_30m is not None
    assert float(item.max_build_dls_before_hold_deg_per_30m) < 5.0


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
    assert result.skipped_same_family_pair_count == 1
    assert result.analyzed_pair_count == 2
    assert result.overlapping_pair_count_before >= result.overlapping_pair_count_after
