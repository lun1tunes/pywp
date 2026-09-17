"""Each uncertainty/display input must invalidate the cached AC result."""

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from pywp import ptc_core
from pywp.mcm import add_dls, compute_positions_min_curv
from pywp.models import Point3D, TrajectoryConfig
from pywp.uncertainty import PlanningUncertaintyModel
from pywp.welltrack_batch import SuccessfulWellPlan


def _success(name="A", y=0):
    surface = Point3D(0, y, 0)
    stations = add_dls(
        compute_positions_min_curv(
            pd.DataFrame(
                {
                    "MD_m": [0, 500, 1000],
                    "INC_deg": [3, 3, 3],
                    "AZI_deg": [90, 90, 90],
                    "segment": ["HOLD", "HOLD", "HOLD"],
                }
            ),
            surface,
        )
    )
    points = [Point3D(row.X_m, row.Y_m, row.Z_m) for row in stations.itertuples()]
    return SuccessfulWellPlan(
        name=name,
        surface=surface,
        t1=points[1],
        t3=points[2],
        stations=stations,
        summary={"kop_md_m": 0, "md_total_m": 1000},
        azimuth_deg=90,
        md_t1_m=500,
        config=TrajectoryConfig(),
    )


def _key(success, model):
    return ptc_core._anti_collision_cache_key(
        successes=[success],
        model=model,
        name_to_color={success.name: "#2563eb"},
        reference_wells=(),
    )


@pytest.mark.parametrize(
    "field",
    [
        f.name
        for f in fields(PlanningUncertaintyModel)
        if f.name != "iscwsa_environment"
    ],
)
def test_all_uncertainty_model_fields_participate_in_ac_cache_key(field):
    model = PlanningUncertaintyModel()
    before = getattr(model, field)
    value = "iscwsa_mwd_poor_magnetic" if field == "iscwsa_tool_code" else before + 1
    changed = replace(model, **{field: value})
    success = _success()
    assert _key(success, changed) != _key(success, model), (
        f"missing cache dependency: {field}"
    )


@pytest.mark.parametrize(
    "field", [f.name for f in fields(PlanningUncertaintyModel().iscwsa_environment)]
)
def test_all_iscwsa_environment_fields_participate_in_ac_cache_key(field):
    model = PlanningUncertaintyModel(iscwsa_tool_code="iscwsa_mwd_poor_magnetic")
    environment = replace(
        model.iscwsa_environment,
        **{field: getattr(model.iscwsa_environment, field) + 0.1},
    )
    changed = replace(model, iscwsa_environment=environment)
    success = _success()
    assert _key(success, changed) != _key(success, model)


def test_changed_isotropic_threshold_rebuilds_actual_cached_wells(monkeypatch):
    state = {}
    monkeypatch.setattr(ptc_core.st, "session_state", state)
    successes = [_success("A"), _success("B", y=30)]
    model = PlanningUncertaintyModel(near_vertical_isotropic_threshold_deg=5)
    first, _, _ = ptc_core._cached_anti_collision_view_model(
        successes=successes,
        uncertainty_model=model,
        records=[],
        parallel_workers=0,
    )
    old_cache = state["wt_anticollision_analysis_cache"]
    changed = replace(model, near_vertical_isotropic_threshold_deg=1)
    second, _, _ = ptc_core._cached_anti_collision_view_model(
        successes=successes,
        uncertainty_model=changed,
        records=[],
        parallel_workers=0,
    )
    assert state["wt_anticollision_last_run"]["cached"] is False
    assert second is not first
    assert state["wt_anticollision_last_run"]["rebuilt_well_count"] == 2
    new_cache = state["wt_anticollision_analysis_cache"]
    for name in ("A", "B"):
        new_well = new_cache["well_cache"][name][1]
        old_well = old_cache["well_cache"][name][1]
        assert new_well is not old_well
        assert new_well.overlay.model.near_vertical_isotropic_threshold_deg == 1
        assert not np.array_equal(
            new_well.overlay.samples[-1].covariance_xyz,
            old_well.overlay.samples[-1].covariance_xyz,
        )


def test_same_content_reimport_has_identical_key_and_coordinate_change_invalidates():
    model = PlanningUncertaintyModel()
    success = _success()
    repeated = _success()
    assert _key(success, model) == _key(repeated, model)
    changed = repeated.model_copy(
        update={"stations": repeated.stations.copy(deep=True)}
    )
    changed.stations.loc[1, "X_m"] = np.nextafter(
        changed.stations.loc[1, "X_m"], np.inf
    )
    assert _key(success, model) != _key(changed, model)
