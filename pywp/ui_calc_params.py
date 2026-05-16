from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import streamlit as st

from pywp.actual_fund_analysis import ActualFundKopDepthFunction
from pywp import TrajectoryConfig
from pywp.planner_config import (
    INTERPOLATION_METHOD_OPTIONS,
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
    "kop_min_vertical",
)
_INT_SUFFIXES: tuple[str, ...] = ("turn_solver_max_restarts",)
_STR_SUFFIXES: tuple[str, ...] = (
    "optimization_mode",
    "turn_solver_mode",
    "interpolation_method",
)
_BOOL_SUFFIXES: tuple[str, ...] = ("offer_j_profile",)


def calc_param_defaults() -> dict[str, float | int | str | bool]:
    """Build UI defaults directly from TrajectoryConfig defaults."""

    cfg = TrajectoryConfig()
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
        "kop_min_vertical": float(cfg.kop_min_vertical_m),
        "optimization_mode": str(cfg.optimization_mode),
        "turn_solver_max_restarts": int(cfg.turn_solver_max_restarts),
        "turn_solver_mode": str(cfg.turn_solver_mode),
        "interpolation_method": str(cfg.interpolation_method),
        "offer_j_profile": bool(cfg.offer_j_profile),
    }


_DEFAULTS_SIGNATURE_KEY_SUFFIX = "__calc_param_defaults_signature__"
_DEFAULTS_SCHEMA_KEY_SUFFIX = "__calc_param_defaults_schema_version__"
_DEFAULTS_SCHEMA_VERSION = 12
_KOP_MODE_SUFFIX = "kop_min_vertical_mode"
_KOP_FUNCTION_PAYLOAD_SUFFIX = "kop_min_vertical_function_payload"
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

    def render_block(
        self,
        *,
        title: str = "Параметры расчета",
        show_solver_help: bool = True,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        render_calc_params_block(
            prefix=self.prefix,
            title=title,
            show_solver_help=show_solver_help,
            on_change=on_change,
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


def build_config_from_state(prefix: str = "") -> TrajectoryConfig:
    return build_trajectory_config(
        md_step_m=float(_state_value(prefix, "md_step")),
        md_step_control_m=float(_state_value(prefix, "md_control")),
        lateral_tolerance_m=float(_state_value(prefix, "lateral_tol")),
        vertical_tolerance_m=float(_state_value(prefix, "vertical_tol")),
        entry_inc_target_deg=float(_state_value(prefix, "entry_inc_target")),
        entry_inc_tolerance_deg=float(_state_value(prefix, "entry_inc_tol")),
        max_inc_deg=float(_state_value(prefix, "max_inc")),
        max_total_md_postcheck_m=float(_state_value(prefix, "max_total_md_postcheck")),
        dls_build_max_deg_per_30m=pi_to_dls(
            float(_state_value(prefix, "dls_build_max"))
        ),
        kop_min_vertical_m=float(_state_value(prefix, "kop_min_vertical")),
        optimization_mode=str(_state_value(prefix, "optimization_mode")),
        turn_solver_max_restarts=int(_state_value(prefix, "turn_solver_max_restarts")),
        turn_solver_mode=str(_state_value(prefix, "turn_solver_mode")),
        interpolation_method=str(_state_value(prefix, "interpolation_method")),
        offer_j_profile=bool(_state_value(prefix, "offer_j_profile")),
    )


def render_calc_params_block(
    prefix: str = "",
    *,
    title: str = "Параметры расчета",
    show_solver_help: bool = True,
    on_change: Callable[[], None] | None = None,
) -> None:
    # Guard against any code path that renders widgets before page-level init.
    apply_calc_param_defaults(prefix=prefix, force=False)
    widget_change_kwargs = {"on_change": on_change} if on_change is not None else {}
    if title:
        st.markdown(f"### {title}")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7, gap="small")
    c1.number_input(
        "Шаг MD, м",
        key=_state_key(prefix, "md_step"),
        min_value=1.0,
        step=1.0,
        help="Шаг выходной инклинометрии. Меньше шаг = подробнее профиль.",
        **widget_change_kwargs,
    )
    c2.number_input(
        "Контрольный шаг MD, м",
        key=_state_key(prefix, "md_control"),
        min_value=0.5,
        step=0.5,
        help="Внутренний шаг проверки ограничений и качества решения.",
        **widget_change_kwargs,
    )
    c3.number_input(
        "Допуск по латерали, м",
        key=_state_key(prefix, "lateral_tol"),
        min_value=0.1,
        step=1.0,
        help="Максимально допустимый горизонтальный промах по t1 и t3: sqrt(dX² + dY²).",
        **widget_change_kwargs,
    )
    c4.number_input(
        "Допуск по вертикали, м",
        key=_state_key(prefix, "vertical_tol"),
        min_value=0.1,
        step=0.1,
        help="Максимально допустимый вертикальный промах по t1 и t3: |dZ|.",
        **widget_change_kwargs,
    )
    c5.number_input(
        "Целевой INC на t1, deg",
        key=_state_key(prefix, "entry_inc_target"),
        min_value=70.0,
        max_value=89.0,
        step=0.5,
        help="Плановый угол входа в пласт в точке t1.",
        **widget_change_kwargs,
    )
    c6.number_input(
        "Допуск INC на t1, deg",
        key=_state_key(prefix, "entry_inc_tol"),
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        help="Допустимое отклонение от целевого INC в точке t1.",
        **widget_change_kwargs,
    )
    c7.number_input(
        "Макс INC по стволу, deg",
        key=_state_key(prefix, "max_inc"),
        min_value=80.0,
        max_value=120.0,
        step=0.5,
        help="Глобальное ограничение по зенитному углу по всей траектории.",
        **widget_change_kwargs,
    )

    d1, d2, d3 = st.columns(3, gap="small")
    d1.number_input(
        "Макс ПИ BUILD, deg/10m",
        key=_state_key(prefix, "dls_build_max"),
        min_value=0.1,
        step=0.1,
        help=(
            "Верхняя граница поиска ПИ на BUILD-сегментах. "
            "Внутри решателя автоматически переводится во внутренние единицы."
        ),
        **widget_change_kwargs,
    )
    (
        d2.number_input(
            "Мин VERTICAL до KOP, м",
            key=_state_key(prefix, "kop_min_vertical"),
            min_value=0.0,
            step=50.0,
            help="Минимальный вертикальный участок от S до начала BUILD1.",
            **widget_change_kwargs,
        )
        if kop_min_vertical_mode(prefix) == KOP_MIN_VERTICAL_MODE_CONSTANT
        else d2.text_input(
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
        d2.caption(kop_min_vertical_detail_label(prefix))
    d3.number_input(
        "Макс итоговая MD (постпроверка), м",
        key=_state_key(prefix, "max_total_md_postcheck"),
        min_value=100.0,
        step=100.0,
        help=(
            "Порог итоговой длины ствола по MD для финальной проверки. "
            "На сам поиск решения не влияет."
        ),
        **widget_change_kwargs,
    )
    st.checkbox(
        "Предлагать J-образную траекторию",
        key=_state_key(prefix, "offer_j_profile"),
        help=(
            "Если включено, планнер сначала пробует простую J-модель: "
            "VERTICAL -> один BUILD -> участок к t3. ПИ по зениту и азимуту "
            "подбирается без превышения максимального ПИ. В режиме без оптимизации "
            "допустимый J-профиль принимается сразу; при минимизации MD финальный "
            "выбор всё равно остаётся за более коротким допустимым профилем."
        ),
        **widget_change_kwargs,
    )

    with st.expander("Параметры солвера", expanded=False):
        if show_solver_help:
            with st.popover("Что означают параметры солвера", icon=":material/tune:"):
                st.markdown(
                    "**Общий принцип:** солвер строит траекторию через точки S → t1 → t3, "
                    "соблюдая ограничения по ПИ и DLS.\n\n"
                    "**Оптимизация** — необязательный второй этап. Если включена, "
                    "солвер пытается сократить MD или поднять KOP поверх уже найденного решения.\n\n"
                    "**Метод решателя** — влияет на сложные профили с поворотом по азимуту. "
                    "`Least Squares` быстрее, `DE Hybrid` надёжнее на нестандартной геометрии.\n\n"
                    "**Рестарты** — при неудаче солвер автоматически перезапускается "
                    "с более плотной сеткой начальных приближений. Больше рестартов — выше шанс "
                    "найти решение, но дольше расчёт."
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
            **widget_change_kwargs,
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
            **widget_change_kwargs,
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
            **widget_change_kwargs,
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
            **widget_change_kwargs,
        )
