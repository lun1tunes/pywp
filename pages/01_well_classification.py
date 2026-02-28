from __future__ import annotations

import pandas as pd
import streamlit as st

from pywp.classification import (
    classify_well,
    complexity_label,
    interpolate_limits,
    reference_table_rows,
    trajectory_type_label,
)
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_utils import arrow_safe_text_dataframe


def _reference_frame() -> pd.DataFrame:
    return arrow_safe_text_dataframe(pd.DataFrame(reference_table_rows()))


def _interpolated_frame(gv_m: float) -> pd.DataFrame:
    limits = interpolate_limits(gv_m=gv_m)
    frame = pd.DataFrame(
        [
            {"Параметр": "ГВ (интерполированная), м", "Значение": f"{float(limits.gv_m):.2f}"},
            {
                "Параметр": "Окно обратного направления, м",
                "Значение": "Не допускается"
                if not limits.reverse_allowed
                else f"{float(limits.reverse_min_m):.1f} - {float(limits.reverse_max_m):.1f}",
            },
            {"Параметр": "Отход t1: Обычная (до), м", "Значение": f"{float(limits.ordinary_offset_max_m):.2f}"},
            {"Параметр": "Отход t1: Сложная (до), м", "Значение": f"{float(limits.complex_offset_max_m):.2f}"},
            {"Параметр": "ЗУ HOLD: Обычная (до), deg", "Значение": f"{float(limits.hold_ordinary_max_deg):.2f}"},
            {"Параметр": "ЗУ HOLD: Сложная (до), deg", "Значение": f"{float(limits.hold_complex_max_deg):.2f}"},
        ]
    )
    return arrow_safe_text_dataframe(frame)


def _classification_result_frame(gv_m: float, horizontal_offset_t1_m: float, hold_inc_deg: float) -> pd.DataFrame:
    result = classify_well(gv_m=gv_m, horizontal_offset_t1_m=horizontal_offset_t1_m, hold_inc_deg=hold_inc_deg)
    frame = pd.DataFrame(
        [
            {"Параметр": "Тип траектории", "Значение": trajectory_type_label(result.trajectory_type)},
            {"Параметр": "Сложность", "Значение": complexity_label(result.complexity)},
            {"Параметр": "Сложность по отходу", "Значение": complexity_label(result.complexity_by_offset)},
            {"Параметр": "Сложность по ЗУ HOLD", "Значение": complexity_label(result.complexity_by_hold)},
        ]
    )
    return arrow_safe_text_dataframe(frame)


def run_page() -> None:
    st.set_page_config(page_title="Классификация скважин", layout="wide")
    apply_page_style(max_width_px=1500)
    render_hero(
        title="Классификация сложности скважин",
        subtitle="Базовая таблица порогов и линейная интерполяция по фактической ГВ t1.",
    )

    st.markdown("### Базовая таблица")
    st.dataframe(_reference_frame(), width="stretch", hide_index=True)

    st.markdown("### Интерполяция и оценка для конкретной скважины")
    c1, c2, c3 = st.columns(3, gap="small")
    gv_m = c1.number_input("ГВ t1, м", min_value=1000.0, max_value=3600.0, value=2400.0, step=10.0)
    horizontal_offset_t1_m = c2.number_input("Горизонтальный отход t1, м", min_value=0.0, value=1000.0, step=10.0)
    hold_inc_deg = c3.number_input("ЗУ секции HOLD, deg", min_value=0.0, max_value=89.0, value=20.0, step=0.5)
    render_small_note("Тип траектории и класс сложности определяются по наихудшему из критериев (отход/ЗУ HOLD).")

    st.dataframe(_interpolated_frame(gv_m=float(gv_m)), width="stretch", hide_index=True)
    st.dataframe(
        _classification_result_frame(
            gv_m=float(gv_m),
            horizontal_offset_t1_m=float(horizontal_offset_t1_m),
            hold_inc_deg=float(hold_inc_deg),
        ),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    run_page()
