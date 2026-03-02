from __future__ import annotations

from types import SimpleNamespace

from pywp import TrajectoryConfig
from pywp.ui_calc_params import CALC_PARAM_DEFAULTS, apply_calc_param_defaults
from pywp.ui_utils import dls_to_pi


def _fake_streamlit() -> SimpleNamespace:
    return SimpleNamespace(session_state={})


def test_calc_param_defaults_match_trajectory_config(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    monkeypatch.setattr(ui_calc_params, "st", _fake_streamlit())

    cfg = TrajectoryConfig()
    assert CALC_PARAM_DEFAULTS["md_step"] == float(cfg.md_step_m)
    assert CALC_PARAM_DEFAULTS["md_control"] == float(cfg.md_step_control_m)
    assert CALC_PARAM_DEFAULTS["pos_tol"] == float(cfg.pos_tolerance_m)
    assert CALC_PARAM_DEFAULTS["entry_inc_target"] == float(cfg.entry_inc_target_deg)
    assert CALC_PARAM_DEFAULTS["entry_inc_tol"] == float(cfg.entry_inc_tolerance_deg)
    assert CALC_PARAM_DEFAULTS["max_inc"] == float(cfg.max_inc_deg)
    assert CALC_PARAM_DEFAULTS["max_total_md_postcheck"] == float(
        cfg.max_total_md_postcheck_m
    )
    assert CALC_PARAM_DEFAULTS["dls_build_min"] == float(
        dls_to_pi(cfg.dls_build_min_deg_per_30m)
    )
    assert CALC_PARAM_DEFAULTS["dls_build_max"] == float(
        dls_to_pi(cfg.dls_build_max_deg_per_30m)
    )
    assert CALC_PARAM_DEFAULTS["kop_min_vertical"] == float(cfg.kop_min_vertical_m)
    assert CALC_PARAM_DEFAULTS["objective_mode"] == str(cfg.objective_mode)
    assert CALC_PARAM_DEFAULTS["turn_solver_mode"] == str(cfg.turn_solver_mode)
    assert CALC_PARAM_DEFAULTS["turn_solver_qmc_samples"] == int(
        cfg.turn_solver_qmc_samples
    )
    assert CALC_PARAM_DEFAULTS["turn_solver_local_starts"] == int(
        cfg.turn_solver_local_starts
    )
    assert CALC_PARAM_DEFAULTS["adaptive_grid_enabled"] == bool(cfg.adaptive_grid_enabled)
    assert CALC_PARAM_DEFAULTS["adaptive_dense_check_enabled"] == bool(
        cfg.adaptive_dense_check_enabled
    )
    assert CALC_PARAM_DEFAULTS["adaptive_grid_initial_size"] == int(
        cfg.adaptive_grid_initial_size
    )
    assert CALC_PARAM_DEFAULTS["adaptive_grid_refine_levels"] == int(
        cfg.adaptive_grid_refine_levels
    )
    assert CALC_PARAM_DEFAULTS["adaptive_grid_top_k"] == int(cfg.adaptive_grid_top_k)
    assert CALC_PARAM_DEFAULTS["parallel_jobs"] == int(cfg.parallel_jobs)
    assert CALC_PARAM_DEFAULTS["profile_cache_enabled"] == bool(cfg.profile_cache_enabled)


def test_apply_defaults_resyncs_when_signature_changed(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    prefix = "wt_cfg_"
    for suffix in CALC_PARAM_DEFAULTS:
        fake_st.session_state[f"{prefix}{suffix}"] = "old-value"
    fake_st.session_state[
        f"{prefix}__calc_param_defaults_signature__"
    ] = (("legacy", "signature"),)

    apply_calc_param_defaults(prefix=prefix, force=False)

    for suffix, default in CALC_PARAM_DEFAULTS.items():
        assert fake_st.session_state[f"{prefix}{suffix}"] == default
    assert (
        fake_st.session_state[f"{prefix}__calc_param_defaults_signature__"]
        != (("legacy", "signature"),)
    )

