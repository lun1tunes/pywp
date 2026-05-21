from __future__ import annotations

import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from pywp.coordinate_integration import (
    CRS_LABEL_BY_VALUE,
    CSV_CRS_OPTIONS,
    DEFAULT_CRS,
    INPUT_CRS_LABEL_BY_VALUE,
    INPUT_CRS_OPTIONS,
    can_transform_crs,
    transform_xy_to_crs,
)
from pywp.coordinate_systems import CoordinateSystem
from pywp.ui_theme import apply_page_style, render_hero

_DEFAULT_OUTPUT_CRS = CoordinateSystem.WGS84_UTM_ZONE_43N
_DEFAULT_X = 600_010.6
_DEFAULT_Y = 7_407_421.0


def _labels(options: list[tuple[str, CoordinateSystem]]) -> list[str]:
    return [label for label, _crs in options]


def _crs_by_label(
    label: str,
    options: list[tuple[str, CoordinateSystem]],
    fallback: CoordinateSystem,
) -> CoordinateSystem:
    return next((crs for item_label, crs in options if item_label == label), fallback)


def _index_for_crs(
    crs: CoordinateSystem,
    options: list[tuple[str, CoordinateSystem]],
) -> int:
    return next((idx for idx, (_label, item) in enumerate(options) if item == crs), 0)


def _format_value(value: float, crs: CoordinateSystem) -> str:
    decimals = 8 if crs.is_geographic() else 3
    return f"{float(value):.{decimals}f}"


def _result_frame(
    *,
    x_in: float,
    y_in: float,
    x_out: float,
    y_out: float,
    input_crs: CoordinateSystem,
    output_crs: CoordinateSystem,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CRS": INPUT_CRS_LABEL_BY_VALUE.get(input_crs, input_crs.name),
                "X": _format_value(x_in, input_crs),
                "Y": _format_value(y_in, input_crs),
                "Роль": "Вход",
            },
            {
                "CRS": CRS_LABEL_BY_VALUE.get(output_crs, output_crs.name),
                "X": _format_value(x_out, output_crs),
                "Y": _format_value(y_out, output_crs),
                "Роль": "Выход",
            },
        ]
    )


def run_page() -> None:
    st.set_page_config(page_title="CRS calculator", layout="wide")
    apply_page_style(max_width_px=1100)
    render_hero(
        title="CRS calculator",
        subtitle="Быстрая проверка пересчёта X/Y той же функцией, что используется в CSV-выгрузке.",
    )

    input_labels = _labels(INPUT_CRS_OPTIONS)
    output_labels = _labels(CSV_CRS_OPTIONS)
    c1, c2 = st.columns(2, gap="small")
    input_label = c1.selectbox(
        "Входная CRS",
        options=input_labels,
        index=_index_for_crs(DEFAULT_CRS, INPUT_CRS_OPTIONS),
        key="crs_calc_input_crs",
    )
    output_label = c2.selectbox(
        "Выходная CRS",
        options=output_labels,
        index=_index_for_crs(_DEFAULT_OUTPUT_CRS, CSV_CRS_OPTIONS),
        key="crs_calc_output_crs",
    )

    input_crs = _crs_by_label(input_label, INPUT_CRS_OPTIONS, DEFAULT_CRS)
    output_crs = _crs_by_label(output_label, CSV_CRS_OPTIONS, _DEFAULT_OUTPUT_CRS)

    x_col, y_col = st.columns(2, gap="small")
    x_in = x_col.number_input(
        "X",
        value=float(_DEFAULT_X),
        step=1.0,
        format="%.3f",
        key="crs_calc_x",
    )
    y_in = y_col.number_input(
        "Y",
        value=float(_DEFAULT_Y),
        step=1.0,
        format="%.3f",
        key="crs_calc_y",
    )

    can_transform = can_transform_crs(input_crs, output_crs)
    x_out, y_out = transform_xy_to_crs(
        float(x_in),
        float(y_in),
        input_crs,
        output_crs,
    )

    if input_crs != output_crs and not can_transform:
        st.warning(
            "Для выбранной пары CRS нет безопасного прямого пересчёта; "
            "значения оставлены как есть."
        )

    st.dataframe(
        _result_frame(
            x_in=float(x_in),
            y_in=float(y_in),
            x_out=float(x_out),
            y_out=float(y_out),
            input_crs=input_crs,
            output_crs=output_crs,
        ),
        hide_index=True,
        width="stretch",
    )

    out_x_col, out_y_col = st.columns(2, gap="small")
    out_x_col.metric("X output", _format_value(x_out, output_crs))
    out_y_col.metric("Y output", _format_value(y_out, output_crs))


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(
            "Запустите приложение командой `streamlit run pages/04_crs_calculator.py`."
        )
    run_page()
