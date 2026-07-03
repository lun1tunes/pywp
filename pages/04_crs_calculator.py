from __future__ import annotations

import logging
import re

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
_DEFAULT_WGS84_LON = 72.23553333333334
_DEFAULT_WGS84_LAT = 71.17081666666667
_DEFAULT_WGS84_DMS = "N 71 10 14.94; E 72 14 7.92"
_BATCH_EDITOR_COLUMNS = ("X", "Y")
_WGS84_INPUT_LABEL = "WGS84 (градусы)"
_CALCULATOR_INPUT_CRS_OPTIONS = [
    *INPUT_CRS_OPTIONS,
    (_WGS84_INPUT_LABEL, CoordinateSystem.WGS84),
]
_SWAP_CRS_BUTTON_LABEL = "⇄"


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


def _default_input_label() -> str:
    return _labels(_CALCULATOR_INPUT_CRS_OPTIONS)[
        _index_for_crs(DEFAULT_CRS, _CALCULATOR_INPUT_CRS_OPTIONS)
    ]


def _default_output_label() -> str:
    return _labels(CSV_CRS_OPTIONS)[
        _index_for_crs(_DEFAULT_OUTPUT_CRS, CSV_CRS_OPTIONS)
    ]


def _swap_crs_labels_supported(
    input_label: str,
    output_label: str,
) -> bool:
    input_labels = set(_labels(_CALCULATOR_INPUT_CRS_OPTIONS))
    output_labels = set(_labels(CSV_CRS_OPTIONS))
    return str(input_label) in output_labels and str(output_label) in input_labels


def _swap_crs_selection_state() -> None:
    input_label = str(
        st.session_state.get("crs_calc_input_crs", _default_input_label())
    )
    output_label = str(
        st.session_state.get("crs_calc_output_crs", _default_output_label())
    )
    if not _swap_crs_labels_supported(input_label, output_label):
        return
    st.session_state["crs_calc_input_crs"] = output_label
    st.session_state["crs_calc_output_crs"] = input_label


def _format_value(value: float, crs: CoordinateSystem) -> str:
    decimals = 8 if crs.is_geographic() else 3
    return f"{float(value):.{decimals}f}"


def _crs_display_name(crs: CoordinateSystem, *, output: bool) -> str:
    if output:
        return CRS_LABEL_BY_VALUE.get(crs, crs.name)
    return INPUT_CRS_LABEL_BY_VALUE.get(crs, crs.name)


def _default_batch_editor_frame() -> pd.DataFrame:
    return pd.DataFrame([{column: None for column in _BATCH_EDITOR_COLUMNS}])


def _parse_dms_component(text: str) -> tuple[str, float]:
    tokens = re.findall(r"[NSEW]|[-+]?\d+(?:[.,]\d+)?", str(text).upper())
    if len(tokens) != 4:
        raise ValueError("ожидается формат вида `N 71 10 14.94`.")
    direction = str(tokens[0])
    if direction not in {"N", "S", "E", "W"}:
        raise ValueError("неверное направление: используйте N/S/E/W.")
    degrees = float(str(tokens[1]).replace(",", "."))
    minutes = float(str(tokens[2]).replace(",", "."))
    seconds = float(str(tokens[3]).replace(",", "."))
    if degrees < 0.0:
        raise ValueError("градусы должны быть неотрицательными.")
    if minutes < 0.0 or minutes >= 60.0:
        raise ValueError("минуты должны быть в диапазоне [0, 60).")
    if seconds < 0.0 or seconds >= 60.0:
        raise ValueError("секунды должны быть в диапазоне [0, 60).")
    limit = 90.0 if direction in {"N", "S"} else 180.0
    if degrees > limit:
        raise ValueError(f"градусы для {direction} должны быть <= {limit:.0f}.")
    if degrees == limit and (minutes > 0.0 or seconds > 0.0):
        raise ValueError(
            f"для {direction} при {limit:.0f}° минуты и секунды должны быть равны 0."
        )
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if direction in {"S", "W"}:
        decimal *= -1.0
    return direction, decimal


def _parse_wgs84_dms_input(text: str) -> tuple[float, float]:
    """Parse WGS84 DMS text and return calculator X/Y order: (longitude, latitude)."""
    parts = [part.strip() for part in re.split(r"[;\n]+", str(text)) if part.strip()]
    if len(parts) != 2:
        raise ValueError(
            "ожидаются две части через `;`: сначала широта, затем долгота."
        )
    parsed = dict(_parse_dms_component(part) for part in parts)
    lat = parsed.get("N", parsed.get("S"))
    lon = parsed.get("E", parsed.get("W"))
    if lat is None or lon is None:
        raise ValueError("нужны широта `N/S` и долгота `E/W`.")
    x_lon_deg = float(lon)
    y_lat_deg = float(lat)
    return x_lon_deg, y_lat_deg


def _normalize_batch_editor_frame(
    editor_value: object,
) -> tuple[pd.DataFrame, int]:
    if editor_value is None:
        return pd.DataFrame(columns=list(_BATCH_EDITOR_COLUMNS)), 0

    frame = pd.DataFrame(editor_value).copy()
    for column in _BATCH_EDITOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame.loc[:, list(_BATCH_EDITOR_COLUMNS)].copy()
    raw_x = frame["X"]
    raw_y = frame["Y"]
    non_blank_mask = ~(
        raw_x.isna()
        | raw_x.astype(str).str.strip().eq("")
    ) | ~(
        raw_y.isna()
        | raw_y.astype(str).str.strip().eq("")
    )
    if not bool(non_blank_mask.any()):
        return pd.DataFrame(columns=list(_BATCH_EDITOR_COLUMNS)), 0

    result = frame.loc[non_blank_mask].copy()
    result["X"] = pd.to_numeric(result["X"], errors="coerce")
    result["Y"] = pd.to_numeric(result["Y"], errors="coerce")
    valid_mask = result["X"].notna() & result["Y"].notna()
    invalid_count = int((~valid_mask).sum())
    result = result.loc[valid_mask].copy()
    if result.empty:
        return pd.DataFrame(columns=list(_BATCH_EDITOR_COLUMNS)), invalid_count
    return result.reset_index(drop=True), invalid_count


def _batch_result_frame(
    *,
    points: pd.DataFrame,
    input_crs: CoordinateSystem,
    output_crs: CoordinateSystem,
) -> pd.DataFrame:
    input_label = _crs_display_name(input_crs, output=False)
    output_label = _crs_display_name(output_crs, output=True)
    columns = {
        f"X input ({input_label})": [],
        f"Y input ({input_label})": [],
        f"X output ({output_label})": [],
        f"Y output ({output_label})": [],
    }
    if points.empty:
        return pd.DataFrame(columns=columns.keys())

    rows: list[dict[str, str]] = []
    for x_in, y_in in points[["X", "Y"]].itertuples(index=False, name=None):
        x_out, y_out = transform_xy_to_crs(
            float(x_in),
            float(y_in),
            input_crs,
            output_crs,
        )
        rows.append(
            {
                f"X input ({input_label})": _format_value(float(x_in), input_crs),
                f"Y input ({input_label})": _format_value(float(y_in), input_crs),
                f"X output ({output_label})": _format_value(float(x_out), output_crs),
                f"Y output ({output_label})": _format_value(float(y_out), output_crs),
            }
        )
    return pd.DataFrame(rows, columns=columns.keys())


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

    input_labels = _labels(_CALCULATOR_INPUT_CRS_OPTIONS)
    output_labels = _labels(CSV_CRS_OPTIONS)
    current_input_label = str(
        st.session_state.get("crs_calc_input_crs", _default_input_label())
    )
    current_output_label = str(
        st.session_state.get("crs_calc_output_crs", _default_output_label())
    )
    can_swap_crs_labels = _swap_crs_labels_supported(
        current_input_label,
        current_output_label,
    )
    c1, c_swap, c2 = st.columns([1.0, 0.18, 1.0], gap="small", vertical_alignment="bottom")
    input_label = c1.selectbox(
        "Входная CRS",
        options=input_labels,
        index=_index_for_crs(DEFAULT_CRS, _CALCULATOR_INPUT_CRS_OPTIONS),
        key="crs_calc_input_crs",
    )
    c_swap.button(
        _SWAP_CRS_BUTTON_LABEL,
        key="crs_calc_swap_crs",
        help=(
            "Инвертировать входную и выходную CRS."
            if can_swap_crs_labels
            else "Инверсия доступна только для CRS, которые есть и во входном, и в выходном списке."
        ),
        disabled=not can_swap_crs_labels,
        on_click=_swap_crs_selection_state,
        width="stretch",
    )
    output_label = c2.selectbox(
        "Выходная CRS",
        options=output_labels,
        index=_index_for_crs(_DEFAULT_OUTPUT_CRS, CSV_CRS_OPTIONS),
        key="crs_calc_output_crs",
    )

    input_crs = _crs_by_label(
        input_label,
        _CALCULATOR_INPUT_CRS_OPTIONS,
        DEFAULT_CRS,
    )
    output_crs = _crs_by_label(output_label, CSV_CRS_OPTIONS, _DEFAULT_OUTPUT_CRS)

    can_transform = can_transform_crs(input_crs, output_crs)
    if input_crs != output_crs and not can_transform:
        st.warning(
            "Для выбранной пары CRS нет безопасного прямого пересчёта; "
            "значения оставлены как есть."
        )
    single_tab, batch_tab = st.tabs(["Одна точка", "Таблица точек"])

    with single_tab:
        x_in: float | None = None
        y_in: float | None = None
        if input_crs.is_geographic():
            input_mode = st.radio(
                "Формат ввода WGS84",
                options=["DMS", "Decimal"],
                horizontal=True,
                key="crs_calc_wgs84_input_mode",
            )
            if str(input_mode) == "DMS":
                dms_value = st.text_input(
                    "WGS84 DMS",
                    value=_DEFAULT_WGS84_DMS,
                    key="crs_calc_wgs84_dms",
                    placeholder="N 71 10 14.94; E 72 14 7.92",
                )
                try:
                    lon_deg, lat_deg = _parse_wgs84_dms_input(dms_value)
                except ValueError as exc:
                    st.warning(f"Не удалось разобрать WGS84 DMS: {exc}")
                else:
                    x_in = lon_deg
                    y_in = lat_deg
                    st.caption(
                        "Распознано как "
                        f"широта {lat_deg:.8f}°, долгота {lon_deg:.8f}°."
                    )
            else:
                x_col, y_col = st.columns(2, gap="small")
                x_in = x_col.number_input(
                    "Долгота (E)",
                    value=float(_DEFAULT_WGS84_LON),
                    step=0.000001,
                    format="%.8f",
                    key="crs_calc_x",
                )
                y_in = y_col.number_input(
                    "Широта (N)",
                    value=float(_DEFAULT_WGS84_LAT),
                    step=0.000001,
                    format="%.8f",
                    key="crs_calc_y",
                )
        else:
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
        if x_in is None or y_in is None:
            st.info("Введите корректную точку WGS84, чтобы увидеть результат пересчёта.")
        else:
            x_out, y_out = transform_xy_to_crs(
                float(x_in),
                float(y_in),
                input_crs,
                output_crs,
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

    with batch_tab:
        editor_format = "%.8f" if input_crs.is_geographic() else "%.3f"
        if input_crs.is_geographic():
            st.caption(
                "Для таблицы WGS84 используйте decimal degrees: X = долгота, Y = широта."
            )
        edited_points = st.data_editor(
            _default_batch_editor_frame(),
            key="crs_calc_batch_points",
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "X": st.column_config.NumberColumn("X", format=editor_format),
                "Y": st.column_config.NumberColumn("Y", format=editor_format),
            },
        )
        batch_points, invalid_row_count = _normalize_batch_editor_frame(edited_points)
        if invalid_row_count:
            st.warning(
                f"Пропущены строки без полной числовой пары X/Y: {invalid_row_count}."
            )
        st.dataframe(
            _batch_result_frame(
                points=batch_points,
                input_crs=input_crs,
                output_crs=output_crs,
            ),
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(
            "Запустите приложение командой `streamlit run pages/04_crs_calculator.py`."
        )
    run_page()
