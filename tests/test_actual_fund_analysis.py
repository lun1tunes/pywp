from __future__ import annotations

from pywp.actual_fund_analysis import (
    CALIBRATION_STATUS_READY,
    actual_well_family_name,
    actual_well_is_horizontal,
    actual_well_pad_group,
    build_actual_fund_well_metrics,
    calibrate_uncertainty_from_actual_fund,
)
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
