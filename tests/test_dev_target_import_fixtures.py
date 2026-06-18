from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pywp.mcm import add_dls, compute_positions_min_curv
from pywp.models import Point3D, TrajectoryConfig, TURN_SOLVER_LEAST_SQUARES
from pywp.planner_types import ProfileParameters
from pywp.planner_validation import (
    _build_trajectory,
    _evaluate_profile_endpoints,
    _offset_endpoint_evaluation,
)
from pywp.ptc_batch_results import build_batch_survey_dev_file
from pywp.reference_trajectories import (
    REFERENCE_WELL_APPROVED,
    ImportedTrajectoryWell,
    parse_reference_trajectory_dev_file,
)
from pywp.welltrack_batch import SuccessfulWellPlan

FIXTURE_DIR = Path("tests/test_data/dev_target_import")
SURFACE = Point3D(598863.0, 7411139.0, 0.0)


@dataclass(frozen=True)
class DevTargetImportCase:
    file_name: str
    params: ProfileParameters
    config: TrajectoryConfig
    meaningful_dls_values: tuple[float, ...]


def _config(**overrides: float | str | bool) -> TrajectoryConfig:
    base: dict[str, float | str | bool] = {
        "md_step_m": 30.0,
        "md_step_control_m": 10.0,
        "lateral_tolerance_m": 30.0,
        "vertical_tolerance_m": 2.0,
        "entry_inc_target_deg": 86.0,
        "entry_inc_tolerance_deg": 2.0,
        "dls_build_max_deg_per_30m": 6.0,
        "dls_horizontal_max_deg_per_30m": 1.5,
        "max_total_md_postcheck_m": 20000.0,
        "turn_solver_mode": TURN_SOLVER_LEAST_SQUARES,
        "optimization_mode": "none",
        "offer_j_profile": False,
        "kop_min_vertical_m": 400.0,
    }
    base.update(overrides)
    return TrajectoryConfig(**base)


def _cases() -> tuple[DevTargetImportCase, ...]:
    return (
        DevTargetImportCase(
            file_name="build_hold_build_equal_pi_with_horizontal_pi.dev",
            params=ProfileParameters(
                kop_vertical_m=550.0,
                inc_entry_deg=84.0,
                inc_required_t1_t3_deg=72.0,
                inc_hold_deg=42.0,
                dls_build1_deg_per_30m=2.4,
                dls_build2_deg_per_30m=2.4,
                build1_length_m=525.0,
                hold_length_m=900.0,
                build2_length_m=525.0,
                horizontal_length_m=900.0,
                horizontal_adjust_length_m=300.0,
                horizontal_hold_length_m=600.0,
                horizontal_inc_deg=72.0,
                horizontal_dls_deg_per_30m=1.2,
                azimuth_hold_deg=12.0,
                azimuth_entry_deg=12.0,
                profile_family="unified",
            ),
            config=_config(
                entry_inc_target_deg=84.0,
                max_inc_deg=90.0,
                dls_build_max_deg_per_30m=2.4,
                dls_horizontal_max_deg_per_30m=1.5,
            ),
            meaningful_dls_values=(1.2, 2.4),
        ),
        DevTargetImportCase(
            file_name="build_hold_build_split_pi.dev",
            params=ProfileParameters(
                kop_vertical_m=500.0,
                inc_entry_deg=78.0,
                inc_required_t1_t3_deg=78.0,
                inc_hold_deg=36.0,
                dls_build1_deg_per_30m=1.2,
                dls_build2_deg_per_30m=2.1,
                build1_length_m=900.0,
                hold_length_m=950.0,
                build2_length_m=600.0,
                horizontal_length_m=700.0,
                horizontal_adjust_length_m=0.0,
                horizontal_hold_length_m=700.0,
                horizontal_inc_deg=78.0,
                horizontal_dls_deg_per_30m=0.0,
                azimuth_hold_deg=33.0,
                azimuth_entry_deg=33.0,
                profile_family="unified",
            ),
            config=_config(
                entry_inc_target_deg=78.0,
                max_inc_deg=90.0,
                dls_build_max_deg_per_30m=2.1,
                dls_horizontal_max_deg_per_30m=0.0,
            ),
            meaningful_dls_values=(1.2, 2.1),
        ),
        DevTargetImportCase(
            file_name="j_profile_constant_pi.dev",
            params=ProfileParameters(
                kop_vertical_m=550.0,
                inc_entry_deg=60.0,
                inc_required_t1_t3_deg=60.0,
                inc_hold_deg=60.0,
                dls_build1_deg_per_30m=1.8,
                dls_build2_deg_per_30m=0.0,
                build1_length_m=1000.0,
                hold_length_m=0.0,
                build2_length_m=0.0,
                horizontal_length_m=1000.0,
                horizontal_adjust_length_m=0.0,
                horizontal_hold_length_m=1000.0,
                horizontal_inc_deg=60.0,
                horizontal_dls_deg_per_30m=0.0,
                azimuth_hold_deg=25.0,
                azimuth_entry_deg=25.0,
                profile_family="j_profile",
            ),
            config=_config(
                entry_inc_target_deg=60.0,
                max_inc_deg=70.0,
                dls_build_max_deg_per_30m=1.8,
                dls_horizontal_max_deg_per_30m=0.0,
            ),
            meaningful_dls_values=(1.8,),
        ),
        DevTargetImportCase(
            file_name="j_profile_variable_pi.dev",
            params=ProfileParameters(
                kop_vertical_m=620.0,
                inc_entry_deg=60.0,
                inc_required_t1_t3_deg=60.0,
                inc_hold_deg=60.0,
                dls_build1_deg_per_30m=2.4,
                dls_build2_deg_per_30m=0.0,
                build1_length_m=0.0,
                hold_length_m=0.0,
                build2_length_m=0.0,
                horizontal_length_m=900.0,
                horizontal_adjust_length_m=0.0,
                horizontal_hold_length_m=900.0,
                horizontal_inc_deg=60.0,
                horizontal_dls_deg_per_30m=0.0,
                azimuth_hold_deg=40.0,
                azimuth_entry_deg=40.0,
                profile_family="j_profile",
                build1_controls=((450.0, 18.0, 40.0, 1.2), (525.0, 60.0, 40.0, 2.4)),
            ),
            config=_config(
                entry_inc_target_deg=60.0,
                max_inc_deg=70.0,
                dls_build_max_deg_per_30m=2.4,
                dls_horizontal_max_deg_per_30m=0.0,
            ),
            meaningful_dls_values=(1.2, 2.4),
        ),
    )


def _build_success(case: DevTargetImportCase) -> SuccessfulWellPlan:
    trajectory = _build_trajectory(case.params)
    survey = trajectory.stations(md_step_m=float(case.config.md_step_m))
    stations = add_dls(compute_positions_min_curv(survey, start=SURFACE))
    endpoints = _offset_endpoint_evaluation(
        evaluation=_evaluate_profile_endpoints(params=case.params),
        surface=SURFACE,
    )
    t1 = Point3D(endpoints.t1.east_m, endpoints.t1.north_m, endpoints.t1.tvd_m)
    t3 = Point3D(endpoints.t3.east_m, endpoints.t3.north_m, endpoints.t3.tvd_m)
    return SuccessfulWellPlan(
        name=Path(case.file_name).stem,
        surface=SURFACE,
        t1=t1,
        t3=t3,
        stations=stations,
        summary={},
        azimuth_deg=float(case.params.azimuth_entry_deg),
        md_t1_m=float(case.params.md_t1_m),
        config=case.config,
    )


def _meaningful_dls_values(well: ImportedTrajectoryWell) -> tuple[float, ...]:
    assert well.dev_export_rows is not None
    values = {
        round(float(value), 3)
        for value in well.dev_export_rows["DLS"].to_numpy(dtype=float)
        if abs(float(value)) > 1e-3
    }
    return tuple(sorted(values))


def test_dev_target_import_fixtures_match_export_path() -> None:
    for case in _cases():
        expected = build_batch_survey_dev_file([_build_success(case)])
        actual = (FIXTURE_DIR / case.file_name).read_bytes()
        assert actual == expected


def test_dev_target_import_fixtures_parse_with_expected_dls_patterns() -> None:
    for case in _cases():
        well = parse_reference_trajectory_dev_file(
            FIXTURE_DIR / case.file_name,
            kind=REFERENCE_WELL_APPROVED,
        )
        assert well.name == Path(case.file_name).stem
        assert well.dev_export_rows is not None
        assert len(well.stations.index) >= 2
        assert _meaningful_dls_values(well) == case.meaningful_dls_values
        assert all(0.6 <= value <= 2.4 for value in case.meaningful_dls_values)
