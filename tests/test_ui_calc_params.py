from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pywp.actual_fund_analysis import ActualFundKopDepthFunction
from pywp import TrajectoryConfig
from pywp.models import J_PROFILE_POLICY_OFF
from pywp.ui_calc_params import (
    CalcParamBinding,
    apply_calc_param_defaults,
    build_config_from_values,
    build_config_from_state,
    calc_param_defaults,
    calc_param_state_values_from_config,
    clear_kop_min_vertical_function,
    kop_min_vertical_function_from_state,
    kop_min_vertical_mode,
    render_calc_params_block,
    set_calc_param_state_values,
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
    assert defaults["max_total_md_postcheck"] == float(cfg.max_total_md_postcheck_m)
    assert defaults["dls_build_max"] == float(dls_to_pi(cfg.dls_build_max_deg_per_30m))
    assert defaults["dls_horizontal_max"] == float(
        dls_to_pi(cfg.dls_horizontal_max_deg_per_30m)
    )
    assert defaults["kop_min_vertical"] == float(cfg.kop_min_vertical_m)
    assert defaults["optimization_mode"] == str(cfg.optimization_mode)
    assert defaults["turn_solver_max_restarts"] == int(cfg.turn_solver_max_restarts)
    assert defaults["turn_solver_mode"] == str(cfg.turn_solver_mode)
    assert defaults["interpolation_method"] == str(cfg.interpolation_method)
    assert defaults["j_profile_policy"] == str(cfg.j_profile_policy)
    assert defaults["j_profile_policy"] == J_PROFILE_POLICY_OFF
    assert defaults["dls_build2_enabled"] is False
    assert defaults["dls_build2_max"] == pytest.approx(defaults["dls_build_max"])
    assert defaults["offer_j_profile"] == bool(cfg.offer_j_profile)
    assert defaults["offer_j_profile"] is False
    assert abs(float(defaults["dls_build_max"]) - 0.8) < 1e-12
    assert abs(float(defaults["dls_horizontal_max"]) - 0.5) < 1e-12


def test_apply_defaults_resyncs_when_signature_changed(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    defaults = calc_param_defaults()
    prefix = "wt_cfg_"
    for suffix in defaults:
        fake_st.session_state[f"{prefix}{suffix}"] = "old-value"
    fake_st.session_state[f"{prefix}__calc_param_defaults_signature__"] = (
        ("legacy", "signature"),
    )

    apply_calc_param_defaults(prefix=prefix, force=False)

    for suffix, default in defaults.items():
        assert fake_st.session_state[f"{prefix}{suffix}"] == default
    assert fake_st.session_state[f"{prefix}__calc_param_defaults_signature__"] != (
        ("legacy", "signature"),
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
    assert (
        int(fake_st.session_state[f"{prefix}__calc_param_defaults_schema_version__"])
        == 15
    )


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
        int(fake_st.session_state[f"{prefix}__calc_param_defaults_schema_version__"])
        == 15
    )


def test_build_config_from_state_uses_independent_horizontal_pi(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    apply_calc_param_defaults(prefix="", force=True)
    fake_st.session_state["dls_build_max"] = 1.4
    fake_st.session_state["dls_horizontal_max"] = 0.8

    config = build_config_from_state(prefix="")

    assert abs(dls_to_pi(config.dls_build_max_deg_per_30m) - 1.4) < 1e-12
    assert abs(dls_to_pi(config.dls_horizontal_max_deg_per_30m) - 0.8) < 1e-12
    assert config.dls_limits_deg_per_30m["BUILD1"] != config.dls_limits_deg_per_30m[
        "HORIZONTAL"
    ]


def test_build_config_from_state_uses_optional_independent_build2_pi(
    monkeypatch,
) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    apply_calc_param_defaults(prefix="", force=True)
    fake_st.session_state["dls_build_max"] = 0.8
    fake_st.session_state["dls_build2_enabled"] = True
    fake_st.session_state["dls_build2_max"] = 1.6

    config = build_config_from_state(prefix="")

    assert abs(dls_to_pi(config.dls_build_max_deg_per_30m) - 0.8) < 1e-12
    assert config.dls_build2_max_deg_per_30m is not None
    assert abs(dls_to_pi(config.dls_build2_max_deg_per_30m) - 1.6) < 1e-12
    assert config.dls_limits_deg_per_30m["BUILD1"] != config.dls_limits_deg_per_30m[
        "BUILD2"
    ]


def test_optional_build2_input_empty_reuses_build1(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    apply_calc_param_defaults(prefix="", force=True)
    fake_st.session_state["dls_build_max"] = 0.9
    fake_st.session_state["dls_build2_enabled"] = True
    fake_st.session_state["dls_build2_max"] = 1.4
    fake_st.session_state["dls_build2_optional_input"] = ""

    ui_calc_params._apply_build2_optional_input_state("")

    assert fake_st.session_state["dls_build2_enabled"] is False
    assert fake_st.session_state["dls_build2_max"] == pytest.approx(0.9)


def test_optional_build2_input_value_enables_separate_limit(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    apply_calc_param_defaults(prefix="", force=True)
    fake_st.session_state["dls_build2_optional_input"] = "1.7"

    ui_calc_params._apply_build2_optional_input_state("")

    assert fake_st.session_state["dls_build2_enabled"] is True
    assert fake_st.session_state["dls_build2_max"] == pytest.approx(1.7)


def test_handle_build2_optional_input_change_applies_state_and_forwards_callback(
    monkeypatch,
) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)
    callback_calls: list[str] = []

    apply_calc_param_defaults(prefix="", force=True)
    fake_st.session_state["dls_build2_optional_input"] = "1.7"

    ui_calc_params._handle_build2_optional_input_change(
        "",
        lambda: callback_calls.append("changed"),
    )

    assert fake_st.session_state["dls_build2_enabled"] is True
    assert fake_st.session_state["dls_build2_max"] == pytest.approx(1.7)
    assert callback_calls == ["changed"]


def test_calc_param_state_values_roundtrip_preserves_config() -> None:
    config = TrajectoryConfig(
        md_step_m=12.0,
        md_step_control_m=2.5,
        lateral_tolerance_m=7.0,
        vertical_tolerance_m=1.2,
        entry_inc_target_deg=86.5,
        entry_inc_tolerance_deg=0.8,
        max_inc_deg=92.0,
        max_total_md_postcheck_m=8200.0,
        dls_build_max_deg_per_30m=2.1,
        dls_build2_max_deg_per_30m=3.6,
        dls_horizontal_max_deg_per_30m=1.5,
        kop_min_vertical_m=950.0,
        optimization_mode="minimize_md",
        turn_solver_max_restarts=3,
        turn_solver_mode="de_hybrid",
        interpolation_method="slerp",
        j_profile_policy="prefer",
        offer_j_profile=True,
    )

    values = calc_param_state_values_from_config(config)
    restored = build_config_from_values(values)

    assert restored.model_dump(exclude={"dls_build2_max_deg_per_30m"}) == (
        config.model_dump(exclude={"dls_build2_max_deg_per_30m"})
    )
    assert restored.dls_build2_max_deg_per_30m == pytest.approx(
        config.dls_build2_max_deg_per_30m
    )


def test_set_calc_param_state_values_populates_prefixed_state(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    values = calc_param_defaults()
    values["dls_build_max"] = 0.6
    values["offer_j_profile"] = True

    set_calc_param_state_values(prefix="wt_local_", values=values)

    assert fake_st.session_state["wt_local_dls_build_max"] == 0.6
    assert fake_st.session_state["wt_local_offer_j_profile"] is True
    assert fake_st.session_state["wt_local_kop_min_vertical_mode"] == "constant"


def test_calc_param_binding_uses_shared_defaults(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    fake_st = _fake_streamlit()
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    binding = CalcParamBinding(prefix="wt_cfg_")
    binding.apply_defaults(force=True)
    defaults = calc_param_defaults()

    for suffix, default in defaults.items():
        assert fake_st.session_state[f"wt_cfg_{suffix}"] == default


def test_render_calc_params_block_disables_widgets(monkeypatch) -> None:
    import pywp.ui_calc_params as ui_calc_params

    calls: list[tuple[str, bool]] = []

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def number_input(self, _label, **kwargs):
            calls.append(("number_input", bool(kwargs.get("disabled"))))
            return None

        def toggle(self, _label, **kwargs):
            calls.append(("toggle", bool(kwargs.get("disabled"))))
            return None

        def text_input(self, _label, **kwargs):
            calls.append(("text_input", bool(kwargs.get("disabled"))))
            return None

        def caption(self, *args, **kwargs):
            return None

    class _FakeStreamlit(SimpleNamespace):
        def markdown(self, *args, **kwargs):
            return None

        def columns(self, spec, *args, **kwargs):
            count = int(spec) if isinstance(spec, int) else len(spec)
            return tuple(_DummyColumn() for _ in range(count))

        def number_input(self, _label, **kwargs):
            calls.append(("number_input", bool(kwargs.get("disabled"))))
            return None

        def selectbox(self, _label, **kwargs):
            calls.append(("selectbox", bool(kwargs.get("disabled"))))
            return None

        def expander(self, *args, **kwargs):
            return _DummyContext()

        def popover(self, *args, **kwargs):
            return _DummyContext()

        def caption(self, *args, **kwargs):
            return None

    fake_st = _FakeStreamlit(session_state={})
    monkeypatch.setattr(ui_calc_params, "st", fake_st)

    render_calc_params_block(disabled=True)

    assert calls
    assert all(disabled for _, disabled in calls)


def test_render_calc_params_block_registers_build2_text_input_callback(
    monkeypatch,
) -> None:
    import pywp.ui_calc_params as ui_calc_params

    captured: dict[str, object] = {}

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def number_input(self, _label, **kwargs):
            return None

        def toggle(self, _label, **kwargs):
            return None

        def text_input(self, label, **kwargs):
            if str(label) == "Макс ПИ BUILD 2, deg/10m":
                captured["on_change"] = kwargs.get("on_change")
                captured["args"] = kwargs.get("args")
            return None

        def caption(self, *args, **kwargs):
            return None

    class _FakeStreamlit(SimpleNamespace):
        def markdown(self, *args, **kwargs):
            return None

        def columns(self, spec, *args, **kwargs):
            count = int(spec) if isinstance(spec, int) else len(spec)
            return tuple(_DummyColumn() for _ in range(count))

        def number_input(self, _label, **kwargs):
            return None

        def selectbox(self, _label, **kwargs):
            return None

        def expander(self, *args, **kwargs):
            return _DummyContext()

        def popover(self, *args, **kwargs):
            return _DummyContext()

        def caption(self, *args, **kwargs):
            return None

    fake_st = _FakeStreamlit(session_state={})
    monkeypatch.setattr(ui_calc_params, "st", fake_st)
    monkeypatch.setattr(
        ui_calc_params,
        "_apply_build2_optional_input_state",
        lambda prefix="": (_ for _ in ()).throw(
            AssertionError("should not run during widget render")
        ),
    )

    render_calc_params_block()

    assert captured["on_change"] is ui_calc_params._handle_build2_optional_input_change
    assert captured["args"] == ("", None)


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


def test_j_profile_policy_control_is_rendered_above_solver_expander() -> None:
    source = Path("pywp/ui_calc_params.py").read_text(encoding="utf-8")

    assert source.index('"Макс итоговая MD (постпроверка), м"') < source.index(
        '"Режим J-профиля"'
    )
    assert source.index('"Режим J-профиля"') < source.index(
        'with st.expander("Параметры солвера"'
    )
