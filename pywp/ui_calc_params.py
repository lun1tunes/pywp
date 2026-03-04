from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import streamlit as st

from pywp import TrajectoryConfig
from pywp.planner_config import (
    OBJECTIVE_OPTIONS,
    SAME_DIRECTION_PROFILE_OPTIONS,
    TURN_SOLVER_OPTIONS,
    build_trajectory_config,
)
from pywp.ui_utils import dls_to_pi, pi_to_dls

_FLOAT_SUFFIXES: tuple[str, ...] = (
    "md_step",
    "md_control",
    "pos_tol",
    "entry_inc_target",
    "entry_inc_tol",
    "max_inc",
    "max_total_md_postcheck",
    "dls_build_max",
    "kop_min_vertical",
    "objective_auto_turn_threshold",
)
_INT_SUFFIXES: tuple[str, ...] = (
    "turn_solver_qmc_samples",
    "turn_solver_local_starts",
    "adaptive_grid_initial_size",
    "adaptive_grid_refine_levels",
    "adaptive_grid_top_k",
    "parallel_jobs",
)
_STR_SUFFIXES: tuple[str, ...] = (
    "objective_mode",
    "turn_solver_mode",
    "same_direction_profile_mode",
)
_BOOL_SUFFIXES: tuple[str, ...] = (
    "objective_auto_switch_to_turn",
    "adaptive_grid_enabled",
    "adaptive_dense_check_enabled",
    "profile_cache_enabled",
)

def calc_param_defaults() -> dict[str, float | int | str | bool]:
    """Build UI defaults directly from TrajectoryConfig defaults."""

    cfg = TrajectoryConfig()
    return {
        "md_step": float(cfg.md_step_m),
        "md_control": float(cfg.md_step_control_m),
        "pos_tol": float(cfg.pos_tolerance_m),
        "entry_inc_target": float(cfg.entry_inc_target_deg),
        "entry_inc_tol": float(cfg.entry_inc_tolerance_deg),
        "max_inc": float(cfg.max_inc_deg),
        "max_total_md_postcheck": float(cfg.max_total_md_postcheck_m),
        "dls_build_max": float(dls_to_pi(cfg.dls_build_max_deg_per_30m)),
        "kop_min_vertical": float(cfg.kop_min_vertical_m),
        "objective_mode": str(cfg.objective_mode),
        "objective_auto_switch_to_turn": bool(cfg.objective_auto_switch_to_turn),
        "objective_auto_turn_threshold": float(cfg.objective_auto_turn_threshold_deg),
        "turn_solver_mode": str(cfg.turn_solver_mode),
        "same_direction_profile_mode": str(cfg.same_direction_profile_mode),
        "turn_solver_qmc_samples": int(cfg.turn_solver_qmc_samples),
        "turn_solver_local_starts": int(cfg.turn_solver_local_starts),
        "adaptive_grid_enabled": bool(cfg.adaptive_grid_enabled),
        "adaptive_dense_check_enabled": bool(cfg.adaptive_dense_check_enabled),
        "adaptive_grid_initial_size": int(cfg.adaptive_grid_initial_size),
        "adaptive_grid_refine_levels": int(cfg.adaptive_grid_refine_levels),
        "adaptive_grid_top_k": int(cfg.adaptive_grid_top_k),
        "parallel_jobs": int(cfg.parallel_jobs),
        "profile_cache_enabled": bool(cfg.profile_cache_enabled),
    }


_DEFAULTS_SIGNATURE_KEY_SUFFIX = "__calc_param_defaults_signature__"
_DEFAULTS_SCHEMA_KEY_SUFFIX = "__calc_param_defaults_schema_version__"
_DEFAULTS_SCHEMA_VERSION = 4


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
    ) -> None:
        render_calc_params_block(
            prefix=self.prefix,
            title=title,
            show_solver_help=show_solver_help,
        )


def _defaults_signature() -> tuple[tuple[str, float | int | str | bool], ...]:
    defaults = calc_param_defaults()
    return tuple(
        (key, defaults[key]) for key in sorted(defaults.keys())
    )


def _state_key(prefix: str, suffix: str) -> str:
    return f"{prefix}{suffix}"


def _state_value(prefix: str, suffix: str) -> float | int | str | bool:
    key = _state_key(prefix, suffix)
    if key in st.session_state:
        return st.session_state[key]
    return calc_param_defaults()[suffix]


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


def apply_calc_param_defaults(prefix: str = "", *, force: bool = False) -> None:
    defaults = calc_param_defaults()
    signature_key = _state_key(prefix, _DEFAULTS_SIGNATURE_KEY_SUFFIX)
    schema_key = _state_key(prefix, _DEFAULTS_SCHEMA_KEY_SUFFIX)
    current_signature = _defaults_signature()
    signature_changed = st.session_state.get(signature_key) != current_signature
    schema_changed = int(st.session_state.get(schema_key, 0)) < int(
        _DEFAULTS_SCHEMA_VERSION
    )
    effective_force = bool(force or signature_changed or schema_changed)
    _setdefault_many(prefixes=(prefix,), force=effective_force, defaults=defaults)
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
    return tuple(signature)


def build_config_from_state(prefix: str = "") -> TrajectoryConfig:
    return build_trajectory_config(
        md_step_m=float(_state_value(prefix, "md_step")),
        md_step_control_m=float(_state_value(prefix, "md_control")),
        pos_tolerance_m=float(_state_value(prefix, "pos_tol")),
        entry_inc_target_deg=float(_state_value(prefix, "entry_inc_target")),
        entry_inc_tolerance_deg=float(_state_value(prefix, "entry_inc_tol")),
        max_inc_deg=float(_state_value(prefix, "max_inc")),
        max_total_md_postcheck_m=float(_state_value(prefix, "max_total_md_postcheck")),
        dls_build_min_deg_per_30m=0.0,
        dls_build_max_deg_per_30m=pi_to_dls(float(_state_value(prefix, "dls_build_max"))),
        kop_min_vertical_m=float(_state_value(prefix, "kop_min_vertical")),
        objective_mode=str(_state_value(prefix, "objective_mode")),
        objective_auto_switch_to_turn=bool(
            _state_value(prefix, "objective_auto_switch_to_turn")
        ),
        objective_auto_turn_threshold_deg=float(
            _state_value(prefix, "objective_auto_turn_threshold")
        ),
        turn_solver_mode=str(_state_value(prefix, "turn_solver_mode")),
        same_direction_profile_mode=str(
            _state_value(prefix, "same_direction_profile_mode")
        ),
        turn_solver_qmc_samples=int(_state_value(prefix, "turn_solver_qmc_samples")),
        turn_solver_local_starts=int(_state_value(prefix, "turn_solver_local_starts")),
        adaptive_grid_enabled=bool(_state_value(prefix, "adaptive_grid_enabled")),
        adaptive_dense_check_enabled=bool(
            _state_value(prefix, "adaptive_dense_check_enabled")
        ),
        adaptive_grid_initial_size=int(_state_value(prefix, "adaptive_grid_initial_size")),
        adaptive_grid_refine_levels=int(
            _state_value(prefix, "adaptive_grid_refine_levels")
        ),
        adaptive_grid_top_k=int(_state_value(prefix, "adaptive_grid_top_k")),
        parallel_jobs=int(_state_value(prefix, "parallel_jobs")),
        profile_cache_enabled=bool(_state_value(prefix, "profile_cache_enabled")),
    )


def render_calc_params_block(
    prefix: str = "",
    *,
    title: str = "Параметры расчета",
    show_solver_help: bool = True,
) -> None:
    # Guard against any code path that renders widgets before page-level init.
    apply_calc_param_defaults(prefix=prefix, force=False)
    if title:
        st.markdown(f"### {title}")
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    c1.number_input(
        "Шаг MD, м",
        key=_state_key(prefix, "md_step"),
        min_value=1.0,
        step=1.0,
        help="Шаг выходной инклинометрии. Меньше шаг = подробнее профиль.",
    )
    c2.number_input(
        "Контрольный шаг MD, м",
        key=_state_key(prefix, "md_control"),
        min_value=0.5,
        step=0.5,
        help="Внутренний шаг проверки ограничений и качества решения.",
    )
    c3.number_input(
        "Допуск по позиции, м",
        key=_state_key(prefix, "pos_tol"),
        min_value=0.1,
        step=0.1,
        help="Максимально допустимый промах по t1 и t3.",
    )
    c4.number_input(
        "Целевой INC на t1, deg",
        key=_state_key(prefix, "entry_inc_target"),
        min_value=70.0,
        max_value=89.0,
        step=0.5,
        help="Плановый угол входа в пласт в точке t1.",
    )
    c5.number_input(
        "Допуск INC на t1, deg",
        key=_state_key(prefix, "entry_inc_tol"),
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        help="Допустимое отклонение от целевого INC в точке t1.",
    )
    c6.number_input(
        "Макс INC по стволу, deg",
        key=_state_key(prefix, "max_inc"),
        min_value=80.0,
        max_value=120.0,
        step=0.5,
        help="Глобальное ограничение по зенитному углу по всей траектории.",
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
    )
    d2.number_input(
        "Мин VERTICAL до KOP, м",
        key=_state_key(prefix, "kop_min_vertical"),
        min_value=0.0,
        step=10.0,
        help="Минимальный вертикальный участок от S до начала BUILD1.",
    )
    d3.number_input(
        "Макс итоговая MD (постпроверка), м",
        key=_state_key(prefix, "max_total_md_postcheck"),
        min_value=100.0,
        step=50.0,
        help=(
            "Порог итоговой длины ствола по MD для финальной проверки. "
            "На сам поиск решения не влияет."
        ),
    )

    with st.expander("Параметры солвера", expanded=False):
        if show_solver_help:
            with st.popover("Что означают параметры солвера", icon=":material/tune:"):
                st.markdown(
                    "- `Целевая функция` задает критерий выбора из допустимых профилей.\n"
                    "- `Тип профиля` управляет вариантом единой модели `J Profile + Continious Build`.\n"
                    "- `Минимизировать азимутальный доворот` полезно для сглаживания траектории в сложной геометрии.\n"
                    "- `Автопереключение objective` (для режима максимизации HOLD) переводит выбор на минимизацию доворота, если найден «конский» TURN выше порога.\n"
                    "- `Минимизировать итоговую MD` уменьшает проходку и объем бурения.\n"
                    "- `TURN` параметры влияют только на некомпланарные случаи.\n"
                    "- `Сетка поиска` — это набор пробных значений KOP и BUILD ПИ.\n"
                    "- `Adaptive` сначала считает по редкой сетке, затем уплотняет ее около лучших решений.\n"
                    "- `Dense-check` — финальная проверка на полной сетке после adaptive-фаз.\n"
                    "- `Parallel jobs` использует процессы в coplanar-поиске.\n"
                    "- `Кэш профилей` сокращает повторные вычисления внутри оптимизации."
                )

        e1, e2, e3, e4 = st.columns(4, gap="small")
        e1.selectbox(
            "Целевая функция",
            options=list(OBJECTIVE_OPTIONS.keys()),
            key=_state_key(prefix, "objective_mode"),
            format_func=lambda key: OBJECTIVE_OPTIONS[str(key)],
            help=(
                "Определяет приоритет среди допустимых решений. "
                "«Минимизировать азимутальный доворот» уменьшает резкие повороты "
                "в некомпланарных кейсах. "
                "«Минимизировать итоговую MD» уменьшает длину проходки."
            ),
        )
        e2.selectbox(
            "Метод TURN-решателя",
            options=list(TURN_SOLVER_OPTIONS.keys()),
            key=_state_key(prefix, "turn_solver_mode"),
            format_func=lambda key: TURN_SOLVER_OPTIONS[str(key)],
            help=(
                "Least Squares (TRF) — быстрый дефолт. "
                "DE Hybrid — тяжелее, но может помочь на сложной геометрии."
            ),
        )
        e3.number_input(
            "TURN QMC samples",
            key=_state_key(prefix, "turn_solver_qmc_samples"),
            min_value=0,
            step=4,
            help=(
                "Дополнительные стартовые точки (Latin Hypercube). "
                "Больше = устойчивее в сложных случаях, но медленнее."
            ),
        )
        e4.selectbox(
            "Тип профиля (J Profile + Continious Build)",
            options=list(SAME_DIRECTION_PROFILE_OPTIONS.keys()),
            key=_state_key(prefix, "same_direction_profile_mode"),
            format_func=lambda key: SAME_DIRECTION_PROFILE_OPTIONS[str(key)],
            help=(
                "Авто: включает J-профиль, когда цель близко к устью и геометрия "
                "допускает проход одним BUILD. "
                "Классический: всегда 2 BUILD + HOLD. "
                "J-профиль: форсирует 1 BUILD до t1."
            ),
        )
        auto_obj_1, auto_obj_2 = st.columns(2, gap="small")
        auto_obj_1.toggle(
            "Автопереключение objective по TURN",
            key=_state_key(prefix, "objective_auto_switch_to_turn"),
            help=(
                "Работает при целевой функции «Максимизировать длину HOLD». "
                "Если итоговый azimuth TURN превышает порог, солвер автоматически "
                "выберет решение с минимальным доворотом."
            ),
        )
        auto_obj_2.number_input(
            "Порог TURN для автопереключения, deg",
            key=_state_key(prefix, "objective_auto_turn_threshold"),
            min_value=0.0,
            max_value=180.0,
            step=1.0,
            help=(
                "Порог «конского» доворота для автопереключения objective. "
                "Если TURN выше этого значения, применяется "
                "«Минимизировать азимутальный доворот»."
            ),
        )
        st.number_input(
            "TURN local starts",
            key=_state_key(prefix, "turn_solver_local_starts"),
            min_value=1,
            step=1,
            help=(
                "Сколько лучших стартовых точек запускать локальным решателем."
            ),
        )

        st.markdown("**Производительность и поиск**")
        st.caption(
            "Сетка — это точки, в которых солвер пробует кандидаты. "
            "Плотнее сетка обычно точнее, но медленнее. "
            "Adaptive ускоряет поиск: грубо в начале, детально рядом с лучшими вариантами."
        )
        p1, p2, p3 = st.columns(3, gap="small")
        p1.toggle(
            "Умная сетка (coarse → fine)",
            key=_state_key(prefix, "adaptive_grid_enabled"),
            help="Включает адаптивное уточнение сетки рядом с лучшими кандидатами.",
        )
        p2.toggle(
            "Кэш профилей",
            key=_state_key(prefix, "profile_cache_enabled"),
            help="Кэширует промежуточные профильные расчеты внутри оптимизации.",
        )
        p3.toggle(
            "Финальный dense-check",
            key=_state_key(prefix, "adaptive_dense_check_enabled"),
            help=(
                "После adaptive-фаз запускает финальную проверку по плотной сетке. "
                "Надежнее, но медленнее."
            ),
        )
        p4, p5, p6 = st.columns(3, gap="small")
        p4.number_input(
            "Стартовая сетка (точек)",
            key=_state_key(prefix, "adaptive_grid_initial_size"),
            min_value=2,
            step=1,
            help="Размер первого coarse-прохода.",
        )
        p5.number_input(
            "Шагов уточнения",
            key=_state_key(prefix, "adaptive_grid_refine_levels"),
            min_value=0,
            step=1,
            help="Сколько refinement-проходов делать после стартовой сетки.",
        )
        p6.number_input(
            "Лучших зон для уточнения (top-k)",
            key=_state_key(prefix, "adaptive_grid_top_k"),
            min_value=1,
            step=1,
            help="Сколько лучших кандидатов использовать как центры следующего refinement.",
        )
        st.number_input(
            "Параллельные процессы (jobs)",
            key=_state_key(prefix, "parallel_jobs"),
            min_value=1,
            step=1,
            help=(
                "Количество процессов для параллельной проверки кандидатов. "
                "1 = без параллелизма."
            ),
        )
