from __future__ import annotations

from types import SimpleNamespace

from pywp.actual_fund_analysis import ActualFundKopDepthFunction
from pywp import TrajectoryConfig
from pywp.ui_calc_params import (
    CalcParamBinding,
    apply_calc_param_defaults,
    calc_param_defaults,
    clear_kop_min_vertical_function,
    kop_min_vertical_function_from_state,
    kop_min_vertical_mode,
    set_kop_min_vertical_function,
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
    assert defaults["lateral_tol"] == float(cfg.lateral_tolerance_m)
    assert defaults["vertical_tol"] == float(cfg.vertical_tolerance_m)
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
    assert defaults["optimization_mode"] == str(cfg.optimization_mode)
    assert defaults["turn_solver_max_restarts"] == int(cfg.turn_solver_max_restarts)
    assert defaults["turn_solver_mode"] == str(cfg.turn_solver_mode)


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
    assert int(fake_st.session_state[f"{prefix}__calc_param_defaults_schema_version__"]) == 11


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
        int(fake_st.session_state[f"{prefix}__calc_param_defaults_schema_version__"]) == 11
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


def test_kop_depth_function_state_roundtrip_updates_signature(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    binding = CalcParamBinding(prefix="wt_cfg_")
    binding.apply_defaults(force=True)
    baseline_signature = binding.state_signature()
    kop_function = ActualFundKopDepthFunction(
        mode="piecewise_linear",
        cluster_count=3,
        anchor_depths_tvd_m=(1600.0, 2600.0, 3600.0),
        anchor_kop_md_m=(820.0, 1260.0, 1780.0),
        note="test",
    )

    set_kop_min_vertical_function(prefix="wt_cfg_", kop_function=kop_function)

    assert kop_min_vertical_mode(prefix="wt_cfg_") == "depth_function"
    restored = kop_min_vertical_function_from_state(prefix="wt_cfg_")
    assert restored is not None
    assert restored.anchor_depths_tvd_m == kop_function.anchor_depths_tvd_m
    assert binding.state_signature() != baseline_signature

    clear_kop_min_vertical_function(prefix="wt_cfg_")
    assert kop_min_vertical_mode(prefix="wt_cfg_") == "constant"


def test_invalid_kop_depth_function_payload_reverts_to_constant(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    fake_st.session_state["wt_cfg_kop_min_vertical_mode"] = "depth_function"
    fake_st.session_state["wt_cfg_kop_min_vertical_function_payload"] = {"broken": True}

    restored = kop_min_vertical_function_from_state(prefix="wt_cfg_")

    assert restored is None
    assert kop_min_vertical_mode(prefix="wt_cfg_") == "constant"
