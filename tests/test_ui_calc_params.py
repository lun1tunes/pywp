from __future__ import annotations

from types import SimpleNamespace

from pywp import TrajectoryConfig
from pywp.ui_calc_params import (
    CalcParamBinding,
    apply_calc_param_defaults,
    calc_param_defaults,
)
from pywp.ui_utils import dls_to_pi


def _fake_streamlit() -> SimpleNamespace:
    return SimpleNamespace(session_state={})


def test_calc_param_defaults_match_trajectory_config(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    monkeypatch.setattr(ui_calc_params, "st", _fake_streamlit())

    defaults = calc_param_defaults()
    cfg = TrajectoryConfig()
    assert defaults["md_step"] == float(cfg.md_step_m)
    assert defaults["md_control"] == float(cfg.md_step_control_m)
    assert defaults["pos_tol"] == float(cfg.pos_tolerance_m)
    assert defaults["entry_inc_target"] == float(cfg.entry_inc_target_deg)
    assert defaults["entry_inc_tol"] == float(cfg.entry_inc_tolerance_deg)
    assert defaults["max_inc"] == float(cfg.max_inc_deg)
    assert defaults["max_total_md_postcheck"] == float(
        cfg.max_total_md_postcheck_m
    )
    assert defaults["dls_build_max"] == float(
        dls_to_pi(cfg.dls_build_max_deg_per_30m)
    )
    assert defaults["kop_min_vertical"] == float(cfg.kop_min_vertical_m)
    assert defaults["objective_mode"] == str(cfg.objective_mode)
    assert defaults["objective_auto_switch_to_turn"] == bool(
        cfg.objective_auto_switch_to_turn
    )
    assert defaults["objective_auto_turn_threshold"] == float(
        cfg.objective_auto_turn_threshold_deg
    )
    assert defaults["turn_solver_mode"] == str(cfg.turn_solver_mode)
    assert defaults["same_direction_profile_mode"] == str(
        cfg.same_direction_profile_mode
    )
    assert defaults["turn_solver_qmc_samples"] == int(
        cfg.turn_solver_qmc_samples
    )
    assert defaults["turn_solver_local_starts"] == int(
        cfg.turn_solver_local_starts
    )
    assert defaults["adaptive_grid_enabled"] == bool(cfg.adaptive_grid_enabled)
    assert defaults["adaptive_dense_check_enabled"] == bool(
        cfg.adaptive_dense_check_enabled
    )
    assert defaults["adaptive_grid_initial_size"] == int(
        cfg.adaptive_grid_initial_size
    )
    assert defaults["adaptive_grid_refine_levels"] == int(
        cfg.adaptive_grid_refine_levels
    )
    assert defaults["adaptive_grid_top_k"] == int(cfg.adaptive_grid_top_k)
    assert defaults["parallel_jobs"] == int(cfg.parallel_jobs)
    assert defaults["profile_cache_enabled"] == bool(cfg.profile_cache_enabled)


def test_apply_defaults_resyncs_when_signature_changed(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    defaults = calc_param_defaults()
    prefix = "wt_cfg_"
    for suffix in defaults:
        fake_st.session_state[f"{prefix}{suffix}"] = "old-value"
    fake_st.session_state[
        f"{prefix}__calc_param_defaults_signature__"
    ] = (("legacy", "signature"),)

    apply_calc_param_defaults(prefix=prefix, force=False)

    for suffix, default in defaults.items():
        assert fake_st.session_state[f"{prefix}{suffix}"] == default
    assert (
        fake_st.session_state[f"{prefix}__calc_param_defaults_signature__"]
        != (("legacy", "signature"),)
    )


def test_apply_defaults_resyncs_when_schema_changed(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    defaults = calc_param_defaults()
    prefix = ""
    for suffix in defaults:
        fake_st.session_state[f"{prefix}{suffix}"] = "old-value"
    fake_st.session_state[f"{prefix}__calc_param_defaults_signature__"] = tuple(
        (key, defaults[key]) for key in sorted(defaults.keys())
    )
    fake_st.session_state[f"{prefix}__calc_param_defaults_schema_version__"] = 1

    apply_calc_param_defaults(prefix=prefix, force=False)

    for suffix, default in defaults.items():
        assert fake_st.session_state[f"{prefix}{suffix}"] == default
    assert int(fake_st.session_state[f"{prefix}__calc_param_defaults_schema_version__"]) == 4


def test_apply_defaults_resyncs_when_schema_missing(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    defaults = calc_param_defaults()
    prefix = "wt_cfg_"
    for suffix in defaults:
        fake_st.session_state[f"{prefix}{suffix}"] = "old-value"
    fake_st.session_state[f"{prefix}__calc_param_defaults_signature__"] = tuple(
        (key, defaults[key]) for key in sorted(defaults.keys())
    )
    # schema key intentionally missing

    apply_calc_param_defaults(prefix=prefix, force=False)

    for suffix, default in defaults.items():
        assert fake_st.session_state[f"{prefix}{suffix}"] == default
    assert (
        int(fake_st.session_state[f"{prefix}__calc_param_defaults_schema_version__"]) == 4
    )


def test_calc_param_binding_uses_shared_defaults(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    binding = CalcParamBinding(prefix="wt_cfg_")
    binding.apply_defaults(force=True)
    defaults = calc_param_defaults()

    for suffix, default in defaults.items():
        assert fake_st.session_state[f"wt_cfg_{suffix}"] == default
