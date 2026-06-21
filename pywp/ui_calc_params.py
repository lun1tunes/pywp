from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import streamlit as st

from pywp.actual_fund_analysis import ActualFundKopDepthFunction
from pywp import TrajectoryConfig
from pywp.planner_config import (
    INTERPOLATION_METHOD_OPTIONS,
    J_PROFILE_POLICY_OPTIONS,
    OPTIMIZATION_OPTIONS,
    TURN_SOLVER_OPTIONS,
    build_trajectory_config,
)
from pywp.ui_utils import dls_to_pi, pi_to_dls

_FLOAT_SUFFIXES: tuple[str, ...] = (
    "md_step",
    "md_control",
    "lateral_tol",
    "vertical_tol",
    "entry_inc_target",
    "entry_inc_tol",
    "max_inc",
    "max_total_md_postcheck",
    "dls_build_max",
    "dls_build2_max",
    "dls_horizontal_max",
    "kop_min_vertical",
    "min_hold_inc",
)
_INT_SUFFIXES: tuple[str, ...] = ("turn_solver_max_restarts",)
_STR_SUFFIXES: tuple[str, ...] = (
    "optimization_mode",
    "turn_solver_mode",
    "interpolation_method",
    "j_profile_policy",
)
_BOOL_SUFFIXES: tuple[str, ...] = (
    "dls_build2_enabled",
    "min_hold_inc_enabled",
    "offer_j_profile",
    "use_fixed_kop",
)


def calc_param_defaults() -> dict[str, float | int | str | bool]:
    """Build UI defaults directly from TrajectoryConfig defaults."""

    cfg = TrajectoryConfig()
    build2_limit = (
        float(cfg.dls_build_max_deg_per_30m)
        if cfg.dls_build2_max_deg_per_30m is None
        else float(cfg.dls_build2_max_deg_per_30m)
    )
    min_hold_inc = (
        13.0
        if cfg.min_hold_inc_deg is None
        else float(cfg.min_hold_inc_deg)
    )
    return {
        "md_step": float(cfg.md_step_m),
        "md_control": float(cfg.md_step_control_m),
        "lateral_tol": float(cfg.lateral_tolerance_m),
        "vertical_tol": float(cfg.vertical_tolerance_m),
        "entry_inc_target": float(cfg.entry_inc_target_deg),
        "entry_inc_tol": float(cfg.entry_inc_tolerance_deg),
        "max_inc": float(cfg.max_inc_deg),
        "max_total_md_postcheck": float(cfg.max_total_md_postcheck_m),
        "dls_build_max": float(dls_to_pi(cfg.dls_build_max_deg_per_30m)),
        "dls_build2_max": float(dls_to_pi(build2_limit)),
        "dls_horizontal_max": float(
            dls_to_pi(cfg.dls_horizontal_max_deg_per_30m)
        ),
        "kop_min_vertical": float(cfg.kop_min_vertical_m),
        "min_hold_inc": float(min_hold_inc),
        "optimization_mode": str(cfg.optimization_mode),
        "turn_solver_max_restarts": int(cfg.turn_solver_max_restarts),
        "turn_solver_mode": str(cfg.turn_solver_mode),
        "interpolation_method": str(cfg.interpolation_method),
        "j_profile_policy": str(cfg.j_profile_policy),
        "dls_build2_enabled": bool(cfg.dls_build2_max_deg_per_30m is not None),
        "min_hold_inc_enabled": bool(cfg.min_hold_inc_deg is not None),
        "offer_j_profile": bool(cfg.offer_j_profile),
        "use_fixed_kop": bool(cfg.use_fixed_kop),
    }


_DEFAULTS_SIGNATURE_KEY_SUFFIX = "__calc_param_defaults_signature__"
_DEFAULTS_SCHEMA_KEY_SUFFIX = "__calc_param_defaults_schema_version__"
_DEFAULTS_SCHEMA_VERSION = 17
_KOP_MODE_SUFFIX = "kop_min_vertical_mode"
_KOP_FUNCTION_PAYLOAD_SUFFIX = "kop_min_vertical_function_payload"
_BUILD2_INPUT_SUFFIX = "dls_build2_optional_input"
_BUILD2_INPUT_SIGNATURE_SUFFIX = "dls_build2_optional_signature"
_MIN_HOLD_INC_INPUT_SUFFIX = "min_hold_inc_optional_input"
_MIN_HOLD_INC_INPUT_SIGNATURE_SUFFIX = "min_hold_inc_optional_signature"
KOP_MIN_VERTICAL_MODE_CONSTANT = "constant"
KOP_MIN_VERTICAL_MODE_DEPTH_FUNCTION = "depth_function"


@dataclass(frozen=True)
class CalcParamBinding:
    """Shared OOP binding for calc-parameter state and UI widgets."""

    prefix: str = ""

    def defaults(self) -> dict[str, float | int | str | bool]:
        return calc_param_defaults()

    def apply_defaults(self, *, force: bool = False) -> None:
        apply_calc_param_defaults(prefix=self.prefix, force=force)

    def preserve_state(self) -> None:
        preserve_calc_param_state(prefix=self.prefix)

    def state_signature(self) -> tuple[object, ...]:
        return calc_param_signature(prefix=self.prefix)

    def build_config(self) -> TrajectoryConfig:
        return build_config_from_state(prefix=self.prefix)

    def apply_optional_inputs(self) -> None:
        apply_optional_calc_param_inputs(prefix=self.prefix)

    def render_block(
        self,
        *,
        title: str = "Параметры расчета",
        show_solver_help: bool = True,
        on_change: Callable[[], None] | None = None,
        disabled: bool = False,
        enable_live_callbacks: bool = True,
    ) -> None:
        render_calc_params_block(
            prefix=self.prefix,
            title=title,
            show_solver_help=show_solver_help,
            on_change=on_change,
            disabled=disabled,
            enable_live_callbacks=enable_live_callbacks,
        )


def _defaults_signature() -> tuple[tuple[str, float | int | str | bool], ...]:
    defaults = calc_param_defaults()
    return tuple((key, defaults[key]) for key in sorted(defaults.keys()))


def _state_key(prefix: str, suffix: str) -> str:
    return f"{prefix}{suffix}"


def _state_value(prefix: str, suffix: str) -> float | int | str | bool:
    key = _state_key(prefix, suffix)
    if key in st.session_state:
        return st.session_state[key]
    return calc_param_defaults()[suffix]


def _kop_mode_key(prefix: str = "") -> str:
    return _state_key(prefix, _KOP_MODE_SUFFIX)


def _kop_function_payload_key(prefix: str = "") -> str:
    return _state_key(prefix, _KOP_FUNCTION_PAYLOAD_SUFFIX)


def _build2_input_key(prefix: str = "") -> str:
    return _state_key(prefix, _BUILD2_INPUT_SUFFIX)


def _build2_input_signature_key(prefix: str = "") -> str:
    return _state_key(prefix, _BUILD2_INPUT_SIGNATURE_SUFFIX)


def _min_hold_inc_input_key(prefix: str = "") -> str:
    return _state_key(prefix, _MIN_HOLD_INC_INPUT_SUFFIX)


def _min_hold_inc_input_signature_key(prefix: str = "") -> str:
    return _state_key(prefix, _MIN_HOLD_INC_INPUT_SIGNATURE_SUFFIX)


def _format_pi_input_value(value: float) -> str:
    return f"{float(value):.1f}"


def _sync_build2_optional_input_state(prefix: str = "") -> None:
    build2_enabled = bool(_state_value(prefix, "dls_build2_enabled"))
    build2_signature = (
        build2_enabled,
        float(_state_value(prefix, "dls_build2_max")) if build2_enabled else None,
    )
    signature_key = _build2_input_signature_key(prefix)
    text_key = _build2_input_key(prefix)
    if (
        text_key not in st.session_state
        or st.session_state.get(signature_key) != build2_signature
    ):
        st.session_state[text_key] = (
            _format_pi_input_value(float(_state_value(prefix, "dls_build2_max")))
            if build2_enabled
            else ""
        )
        st.session_state[signature_key] = build2_signature


def _sync_min_hold_inc_optional_input_state(prefix: str = "") -> None:
    min_hold_enabled = bool(_state_value(prefix, "min_hold_inc_enabled"))
    min_hold_signature = (
        min_hold_enabled,
        float(_state_value(prefix, "min_hold_inc")) if min_hold_enabled else None,
    )
    signature_key = _min_hold_inc_input_signature_key(prefix)
    text_key = _min_hold_inc_input_key(prefix)
    if (
        text_key not in st.session_state
        or st.session_state.get(signature_key) != min_hold_signature
    ):
        st.session_state[text_key] = (
            _format_pi_input_value(float(_state_value(prefix, "min_hold_inc")))
            if min_hold_enabled
            else ""
        )
        st.session_state[signature_key] = min_hold_signature


def _apply_build2_optional_input_state(prefix: str = "") -> None:
    raw_value = str(st.session_state.get(_build2_input_key(prefix), "")).strip()
    build1_value = float(_state_value(prefix, "dls_build_max"))
    if not raw_value:
        st.session_state[_state_key(prefix, "dls_build2_enabled")] = False
        st.session_state[_state_key(prefix, "dls_build2_max")] = float(build1_value)
        st.session_state[_build2_input_signature_key(prefix)] = (False, None)
        return
    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError):
        st.session_state[_state_key(prefix, "dls_build2_enabled")] = False
        st.session_state[_state_key(prefix, "dls_build2_max")] = float(build1_value)
        st.session_state[_build2_input_signature_key(prefix)] = (False, None)
        return
    parsed_value = max(0.1, float(parsed_value))
    st.session_state[_state_key(prefix, "dls_build2_enabled")] = True
    st.session_state[_state_key(prefix, "dls_build2_max")] = float(parsed_value)
    st.session_state[_build2_input_signature_key(prefix)] = (True, float(parsed_value))


def _apply_min_hold_inc_optional_input_state(prefix: str = "") -> None:
    raw_value = str(st.session_state.get(_min_hold_inc_input_key(prefix), "")).strip()
    default_value = float(calc_param_defaults()["min_hold_inc"])
    if not raw_value:
        st.session_state[_state_key(prefix, "min_hold_inc_enabled")] = False
        st.session_state[_state_key(prefix, "min_hold_inc")] = float(default_value)
        st.session_state[_min_hold_inc_input_signature_key(prefix)] = (False, None)
        return
    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError):
        st.session_state[_state_key(prefix, "min_hold_inc_enabled")] = False
        st.session_state[_state_key(prefix, "min_hold_inc")] = float(default_value)
        st.session_state[_min_hold_inc_input_signature_key(prefix)] = (False, None)
        return
    parsed_value = max(0.5, float(parsed_value))
    st.session_state[_state_key(prefix, "min_hold_inc_enabled")] = True
    st.session_state[_state_key(prefix, "min_hold_inc")] = float(parsed_value)
    st.session_state[_min_hold_inc_input_signature_key(prefix)] = (
        True,
        float(parsed_value),
    )


def _handle_build2_optional_input_change(
    prefix: str = "",
    on_change: Callable[[], None] | None = None,
) -> None:
    _apply_build2_optional_input_state(prefix)
    if on_change is not None:
        on_change()


def _handle_min_hold_inc_optional_input_change(
    prefix: str = "",
    on_change: Callable[[], None] | None = None,
) -> None:
    _apply_min_hold_inc_optional_input_state(prefix)
    if on_change is not None:
        on_change()


def apply_optional_calc_param_inputs(prefix: str = "") -> None:
    _apply_build2_optional_input_state(prefix)
    _apply_min_hold_inc_optional_input_state(prefix)


def kop_min_vertical_mode(prefix: str = "") -> str:
    mode = str(
        st.session_state.get(_kop_mode_key(prefix), KOP_MIN_VERTICAL_MODE_CONSTANT)
    ).strip()
    if mode not in {
        KOP_MIN_VERTICAL_MODE_CONSTANT,
        KOP_MIN_VERTICAL_MODE_DEPTH_FUNCTION,
    }:
        return KOP_MIN_VERTICAL_MODE_CONSTANT
    return mode


def kop_min_vertical_function_from_state(
    prefix: str = "",
) -> ActualFundKopDepthFunction | None:
    if kop_min_vertical_mode(prefix) != KOP_MIN_VERTICAL_MODE_DEPTH_FUNCTION:
        return None
    payload = st.session_state.get(_kop_function_payload_key(prefix))
    if isinstance(payload, ActualFundKopDepthFunction):
        return payload
    if isinstance(payload, dict):
        try:
            return ActualFundKopDepthFunction(**payload)
        except (TypeError, ValueError):
            clear_kop_min_vertical_function(prefix=prefix)
            return None
    return None


def kop_min_vertical_display_label(prefix: str = "") -> str:
    kop_function = kop_min_vertical_function_from_state(prefix)
    if kop_function is None:
        return f"{float(_state_value(prefix, 'kop_min_vertical')):.0f} м"
    if kop_function.cluster_count <= 1:
        return "Константа по фактическому фонду"
    return f"Функция KOP / TVD ({int(kop_function.cluster_count)} кластера)"


def kop_min_vertical_detail_label(prefix: str = "") -> str:
    kop_function = kop_min_vertical_function_from_state(prefix)
    if kop_function is None:
        return ""
    anchors = list(zip(kop_function.anchor_depths_tvd_m, kop_function.anchor_kop_md_m))
    preview = ", ".join(
        f"{float(depth):.0f}->{float(kop):.0f}" for depth, kop in anchors[:4]
    )
    if len(anchors) > 4:
        preview += ", …"
    return f"TVD -> KOP: {preview}"


def set_kop_min_vertical_function(
    *,
    prefix: str = "",
    kop_function: ActualFundKopDepthFunction,
) -> None:
    st.session_state[_kop_mode_key(prefix)] = KOP_MIN_VERTICAL_MODE_DEPTH_FUNCTION
    st.session_state[_kop_function_payload_key(prefix)] = kop_function.model_dump(
        mode="python"
    )
    if kop_function.anchor_kop_md_m:
        st.session_state[_state_key(prefix, "kop_min_vertical")] = float(
            min(kop_function.anchor_kop_md_m)
        )


def clear_kop_min_vertical_function(
    *,
    prefix: str = "",
    kop_min_vertical_m: float | None = None,
) -> None:
    st.session_state[_kop_mode_key(prefix)] = KOP_MIN_VERTICAL_MODE_CONSTANT
    st.session_state[_kop_function_payload_key(prefix)] = None
    if kop_min_vertical_m is not None:
        st.session_state[_state_key(prefix, "kop_min_vertical")] = float(
            kop_min_vertical_m
        )


def _setdefault_many(
    *,
    prefixes: Iterable[str],
    force: bool,
    defaults: dict[str, float | int | str | bool],
) -> None:
    for prefix in prefixes:
        for suffix, default_value in defaults.items():
            key = _state_key(prefix, suffix)
            if force or key not in st.session_state:
                st.session_state[key] = default_value


def _migrate_legacy_tolerance_keys(prefix: str = "") -> bool:
    migrated = False
    legacy_keys = (
        _state_key(prefix, "pos_tolerance_m"),
        _state_key(prefix, "pos_tol"),
    )
    legacy_value: float | None = None
    for legacy_key in legacy_keys:
        if legacy_key in st.session_state:
            if legacy_value is None:
                try:
                    legacy_value = float(st.session_state[legacy_key])
                except (TypeError, ValueError):
                    legacy_value = None
            del st.session_state[legacy_key]
            migrated = True
    if legacy_value is not None:
        lateral_key = _state_key(prefix, "lateral_tol")
        vertical_key = _state_key(prefix, "vertical_tol")
        if lateral_key not in st.session_state:
            st.session_state[lateral_key] = float(legacy_value)
        if vertical_key not in st.session_state:
            st.session_state[vertical_key] = float(legacy_value)
    return migrated


def apply_calc_param_defaults(prefix: str = "", *, force: bool = False) -> None:
    legacy_migrated = _migrate_legacy_tolerance_keys(prefix=prefix)
    defaults = calc_param_defaults()
    signature_key = _state_key(prefix, _DEFAULTS_SIGNATURE_KEY_SUFFIX)
    schema_key = _state_key(prefix, _DEFAULTS_SCHEMA_KEY_SUFFIX)
    current_signature = _defaults_signature()
    signature_changed = st.session_state.get(signature_key) != current_signature
    schema_changed = int(st.session_state.get(schema_key, 0)) < int(
        _DEFAULTS_SCHEMA_VERSION
    )
    effective_force = bool(
        force or signature_changed or schema_changed or legacy_migrated
    )
    _setdefault_many(prefixes=(prefix,), force=effective_force, defaults=defaults)
    mode_key = _kop_mode_key(prefix)
    payload_key = _kop_function_payload_key(prefix)
    if effective_force or mode_key not in st.session_state:
        st.session_state[mode_key] = KOP_MIN_VERTICAL_MODE_CONSTANT
    if effective_force or payload_key not in st.session_state:
        st.session_state[payload_key] = None
    st.session_state[signature_key] = current_signature
    st.session_state[schema_key] = int(_DEFAULTS_SCHEMA_VERSION)


def preserve_calc_param_state(prefix: str = "") -> None:
    """Interrupt Streamlit widget cleanup for calc-param keys.

    Recommended by Streamlit docs for multipage/dynamic rendering scenarios.
    """

    defaults = calc_param_defaults()
    for suffix in defaults:
        key = _state_key(prefix, suffix)
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def calc_param_signature(prefix: str = "") -> tuple[object, ...]:
    signature: list[object] = []
    for suffix in _FLOAT_SUFFIXES:
        signature.append(float(_state_value(prefix, suffix)))
    for suffix in _INT_SUFFIXES:
        signature.append(int(_state_value(prefix, suffix)))
    for suffix in _STR_SUFFIXES:
        signature.append(str(_state_value(prefix, suffix)))
    for suffix in _BOOL_SUFFIXES:
        signature.append(bool(_state_value(prefix, suffix)))
    signature.append(kop_min_vertical_mode(prefix))
    kop_function = kop_min_vertical_function_from_state(prefix)
    if kop_function is not None:
        signature.append(
            tuple(float(value) for value in kop_function.anchor_depths_tvd_m)
        )
        signature.append(tuple(float(value) for value in kop_function.anchor_kop_md_m))
    return tuple(signature)


def calc_param_state_values_from_config(
    config: TrajectoryConfig,
) -> dict[str, float | int | str | bool]:
    build2_limit = (
        float(config.dls_build_max_deg_per_30m)
        if config.dls_build2_max_deg_per_30m is None
        else float(config.dls_build2_max_deg_per_30m)
    )
    return {
        "md_step": float(config.md_step_m),
        "md_control": float(config.md_step_control_m),
        "lateral_tol": float(config.lateral_tolerance_m),
        "vertical_tol": float(config.vertical_tolerance_m),
        "entry_inc_target": float(config.entry_inc_target_deg),
        "entry_inc_tol": float(config.entry_inc_tolerance_deg),
        "max_inc": float(config.max_inc_deg),
        "max_total_md_postcheck": float(config.max_total_md_postcheck_m),
        "dls_build_max": float(dls_to_pi(config.dls_build_max_deg_per_30m)),
        "dls_build2_max": float(dls_to_pi(build2_limit)),
        "dls_horizontal_max": float(
            dls_to_pi(config.dls_horizontal_max_deg_per_30m)
        ),
        "kop_min_vertical": float(config.kop_min_vertical_m),
        "min_hold_inc": float(
            13.0 if config.min_hold_inc_deg is None else config.min_hold_inc_deg
        ),
        "optimization_mode": str(config.optimization_mode),
        "turn_solver_max_restarts": int(config.turn_solver_max_restarts),
        "turn_solver_mode": str(config.turn_solver_mode),
        "interpolation_method": str(config.interpolation_method),
        "j_profile_policy": str(config.j_profile_policy),
        "dls_build2_enabled": bool(config.dls_build2_max_deg_per_30m is not None),
        "min_hold_inc_enabled": bool(config.min_hold_inc_deg is not None),
        "offer_j_profile": bool(config.offer_j_profile),
        "use_fixed_kop": bool(config.use_fixed_kop),
    }


def set_calc_param_state_values(
    *,
    prefix: str = "",
    values: dict[str, float | int | str | bool],
) -> None:
    apply_calc_param_defaults(prefix=prefix, force=False)
    defaults = calc_param_defaults()
    for suffix, default_value in defaults.items():
        st.session_state[_state_key(prefix, suffix)] = values.get(suffix, default_value)
    st.session_state[_kop_mode_key(prefix)] = KOP_MIN_VERTICAL_MODE_CONSTANT
    st.session_state[_kop_function_payload_key(prefix)] = None
    _sync_build2_optional_input_state(prefix)
    _sync_min_hold_inc_optional_input_state(prefix)


def _build_config_kwargs_from_values(
    values: dict[str, float | int | str | bool],
) -> dict[str, float | int | str | bool]:
    build2_enabled = bool(values.get("dls_build2_enabled", False))
    return {
        "md_step_m": float(values["md_step"]),
        "md_step_control_m": float(values["md_control"]),
        "lateral_tolerance_m": float(values["lateral_tol"]),
        "vertical_tolerance_m": float(values["vertical_tol"]),
        "entry_inc_target_deg": float(values["entry_inc_target"]),
        "entry_inc_tolerance_deg": float(values["entry_inc_tol"]),
        "max_inc_deg": float(values["max_inc"]),
        "max_total_md_postcheck_m": float(values["max_total_md_postcheck"]),
        "dls_build_max_deg_per_30m": pi_to_dls(float(values["dls_build_max"])),
        "dls_build2_max_deg_per_30m": (
            pi_to_dls(float(values["dls_build2_max"])) if build2_enabled else None
        ),
        "dls_horizontal_max_deg_per_30m": pi_to_dls(
            float(values["dls_horizontal_max"])
        ),
        "kop_min_vertical_m": float(values["kop_min_vertical"]),
        "use_fixed_kop": bool(values.get("use_fixed_kop", False)),
        "min_hold_inc_deg": (
            float(values["min_hold_inc"])
            if bool(values.get("min_hold_inc_enabled", False))
            else None
        ),
        "optimization_mode": str(values["optimization_mode"]),
        "turn_solver_max_restarts": int(values["turn_solver_max_restarts"]),
        "turn_solver_mode": str(values["turn_solver_mode"]),
        "interpolation_method": str(values["interpolation_method"]),
        "j_profile_policy": str(values["j_profile_policy"]),
        "offer_j_profile": bool(values["offer_j_profile"]),
    }


def build_config_from_values(
    values: dict[str, float | int | str | bool],
) -> TrajectoryConfig:
    defaults = calc_param_defaults()
    resolved = {
        suffix: values.get(suffix, defaults[suffix]) for suffix in defaults
    }
    return build_trajectory_config(**_build_config_kwargs_from_values(resolved))


def build_config_from_state(prefix: str = "") -> TrajectoryConfig:
    defaults = calc_param_defaults()
    resolved = {
        suffix: _state_value(prefix, suffix) for suffix in defaults
    }
    return build_trajectory_config(**_build_config_kwargs_from_values(resolved))


def render_calc_params_block(
    prefix: str = "",
    *,
    title: str = "Параметры расчета",
    show_solver_help: bool = True,
    on_change: Callable[[], None] | None = None,
    disabled: bool = False,
    enable_live_callbacks: bool = True,
) -> None:
    # Guard against any code path that renders widgets before page-level init.
    apply_calc_param_defaults(prefix=prefix, force=False)
    widget_change_kwargs = (
        {"on_change": on_change}
        if on_change is not None and enable_live_callbacks
        else {}
    )
    widget_state_kwargs = {"disabled": bool(disabled)}
    widget_kwargs = {**widget_change_kwargs, **widget_state_kwargs}
    optional_input_callback_kwargs = (
        {
            "on_change": _handle_build2_optional_input_change,
            "args": (prefix, on_change),
        }
        if enable_live_callbacks
        else {}
    )
    min_hold_callback_kwargs = (
        {
            "on_change": _handle_min_hold_inc_optional_input_change,
            "args": (prefix, on_change),
        }
        if enable_live_callbacks
        else {}
    )
    if title:
        st.markdown(f"### {title}")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7, gap="small")
    c1.number_input(
        "Шаг MD, м",
        key=_state_key(prefix, "md_step"),
        min_value=1.0,
        step=1.0,
        help="Шаг выходной инклинометрии. Меньше шаг = подробнее профиль.",
        **widget_kwargs,
    )
    c2.number_input(
        "Контрольный шаг MD, м",
        key=_state_key(prefix, "md_control"),
        min_value=0.5,
        step=0.5,
        help="Внутренний шаг проверки ограничений и качества решения.",
        **widget_kwargs,
    )
    c3.number_input(
        "Допуск по латерали, м",
        key=_state_key(prefix, "lateral_tol"),
        min_value=0.1,
        step=1.0,
        help="Максимально допустимый горизонтальный промах по t1 и t3: sqrt(dX² + dY²).",
        **widget_kwargs,
    )
    c4.number_input(
        "Допуск по вертикали, м",
        key=_state_key(prefix, "vertical_tol"),
        min_value=0.1,
        step=0.1,
        help="Максимально допустимый вертикальный промах по t1 и t3: |dZ|.",
        **widget_kwargs,
    )
    c5.number_input(
        "Целевой INC на t1, deg",
        key=_state_key(prefix, "entry_inc_target"),
        min_value=70.0,
        max_value=89.0,
        step=0.5,
        help="Плановый угол входа в пласт в точке t1.",
        **widget_kwargs,
    )
    c6.number_input(
        "Допуск INC на t1, deg",
        key=_state_key(prefix, "entry_inc_tol"),
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        help="Допустимое отклонение от целевого INC в точке t1.",
        **widget_kwargs,
    )
    c7.number_input(
        "Макс INC по стволу, deg",
        key=_state_key(prefix, "max_inc"),
        min_value=80.0,
        max_value=120.0,
        step=0.5,
        help="Глобальное ограничение по зенитному углу по всей траектории.",
        **widget_kwargs,
    )

    _sync_build2_optional_input_state(prefix)
    build2_enabled = bool(_state_value(prefix, "dls_build2_enabled"))
    d1, d2, d3, d4 = st.columns(4, gap="small")
    d1.number_input(
        "Макс ПИ BUILD 1, deg/10m" if build2_enabled else "Макс ПИ BUILD 1/2, deg/10m",
        key=_state_key(prefix, "dls_build_max"),
        min_value=0.1,
        step=0.1,
        help=(
            "Верхняя граница поиска ПИ до входа в t1. "
            "Если BUILD 2 не задан отдельно, это общий лимит для BUILD1 и BUILD2."
        ),
        **widget_kwargs,
    )
    d2.text_input(
        "Макс ПИ BUILD 2, deg/10m",
        key=_build2_input_key(prefix),
        placeholder=_format_pi_input_value(float(_state_value(prefix, "dls_build_max"))),
        help=(
            "Оставьте пустым, чтобы BUILD2 использовал тот же лимит, что и BUILD1. "
            "Если указано значение, BUILD2 считается с отдельным лимитом."
        ),
        **optional_input_callback_kwargs,
        **widget_state_kwargs,
    )
    d3.number_input(
        "Макс ПИ HORIZONTAL, deg/10m",
        key=_state_key(prefix, "dls_horizontal_max"),
        min_value=0.1,
        step=0.1,
        help=(
            "Лимит ПИ для участка после t1: HORIZONTAL-переход к t3 и "
            "HORIZONTAL_BUILD между уровнями MULTIHORIZONTAL."
        ),
        **widget_kwargs,
    )
    (
        d4.number_input(
            "Мин VERTICAL до KOP, м",
            key=_state_key(prefix, "kop_min_vertical"),
            min_value=0.0,
            step=50.0,
            help="Минимальный вертикальный участок от S до начала BUILD1.",
            **widget_kwargs,
        )
        if kop_min_vertical_mode(prefix) == KOP_MIN_VERTICAL_MODE_CONSTANT
        else d4.text_input(
            "Мин VERTICAL до KOP, м",
            value=kop_min_vertical_display_label(prefix),
            disabled=True,
            help=(
                "Для расчета активна функция KOP / TVD по фактическому фонду. "
                "Пересчет по скважинам будет подставлять свой KOP по глубине t1."
            ),
        )
    )
    if kop_min_vertical_mode(prefix) == KOP_MIN_VERTICAL_MODE_DEPTH_FUNCTION:
        d4.caption(kop_min_vertical_detail_label(prefix))
    _sync_min_hold_inc_optional_input_state(prefix)
    e1, e2 = st.columns(2, gap="small")
    e1.checkbox(
        "Использовать фиксированный KOP",
        key=_state_key(prefix, "use_fixed_kop"),
        help=(
            "Если включено, указанное значение KOP используется как точная глубина "
            "начала BUILD. Если выключено, KOP остается минимальным ограничением, "
            "и солвер может опустить его глубже. Для функции KOP / TVD фиксируется "
            "рассчитанный KOP конкретной скважины."
        ),
        **widget_kwargs,
    )
    e2.text_input(
        "Мин. угол стабилизации, deg",
        key=_min_hold_inc_input_key(prefix),
        help=(
            "Оставьте пустым, чтобы не ограничивать угол HOLD дополнительно. "
            "Если указано значение, солвер ищет решение с HOLD не ниже этого угла."
        ),
        **min_hold_callback_kwargs,
        **widget_state_kwargs,
    )
    st.number_input(
        "Макс итоговая MD (постпроверка), м",
        key=_state_key(prefix, "max_total_md_postcheck"),
        min_value=100.0,
        step=100.0,
        help=(
            "Порог итоговой длины ствола по MD для финальной проверки. "
            "На сам поиск решения не влияет."
        ),
        **widget_kwargs,
    )
    st.selectbox(
        "Режим J-профиля",
        options=list(J_PROFILE_POLICY_OPTIONS.keys()),
        format_func=lambda value: J_PROFILE_POLICY_OPTIONS.get(str(value), str(value)),
        key=_state_key(prefix, "j_profile_policy"),
        help=(
            "J-профиль: вертикаль → один набор → t3. "
            "«Предлагать» — кандидат среди вариантов. "
            "«Предпочитать» — принять J, если он допустим."
        ),
        **widget_kwargs,
    )

    with st.expander("Параметры солвера", expanded=False):
        if show_solver_help:
            with st.popover("Что означают параметры солвера", icon=":material/tune:"):
                st.markdown(
                    "Солвер строит траекторию S → t1 → t3 с ограничениями по ПИ.\n\n"
                    "**Оптимизация** — сократить MD или поднять KOP после нахождения решения.\n\n"
                    "**Метод** — Least Squares быстрее, DE Hybrid надёжнее на сложной геометрии.\n\n"
                    "**Рестарты** — повторные попытки при неудаче; больше — дольше, но надёжнее."
                )

        st.caption(
            "По умолчанию солвер ищет допустимую траекторию без оптимизации. "
            "Ниже можно включить оптимизацию, выбрать метод и число рестартов."
        )
        st.selectbox(
            "Оптимизация",
            options=list(OPTIMIZATION_OPTIONS.keys()),
            key=_state_key(prefix, "optimization_mode"),
            format_func=lambda key: OPTIMIZATION_OPTIONS[str(key)],
            help=(
                "Без оптимизации — найти допустимый профиль. "
                "Мин. MD — сократить длину ствола. "
                "Мин. KOP — поднять точку набора."
            ),
            **widget_kwargs,
        )
        st.selectbox(
            "Метод решателя",
            options=list(TURN_SOLVER_OPTIONS.keys()),
            key=_state_key(prefix, "turn_solver_mode"),
            format_func=lambda key: TURN_SOLVER_OPTIONS[str(key)],
            help=(
                "Least Squares — быстрый, подходит для большинства профилей. "
                "DE Hybrid — медленнее, но надёжнее на сложной геометрии."
            ),
            **widget_kwargs,
        )
        st.selectbox(
            "Интерполяция BUILD",
            options=list(INTERPOLATION_METHOD_OPTIONS.keys()),
            key=_state_key(prefix, "interpolation_method"),
            format_func=lambda key: INTERPOLATION_METHOD_OPTIONS[str(key)],
            help=(
                "Rodrigues — численно стабильная формула вращения, рекомендуется. "
                "SLERP — классическая сферическая линейная интерполяция."
            ),
            **widget_kwargs,
        )
        st.number_input(
            "Макс рестартов решателя",
            key=_state_key(prefix, "turn_solver_max_restarts"),
            min_value=0,
            max_value=6,
            step=1,
            help=(
                "Число повторных попыток при неудаче. "
                "Каждый рестарт увеличивает сетку поиска — дольше, но надёжнее."
            ),
            **widget_kwargs,
        )
