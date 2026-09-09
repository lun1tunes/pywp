from __future__ import annotations

import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.runtime.scriptrunner_utils.script_run_context import (  # noqa: E402
    get_script_run_ctx,
)

from pywp.coordinate_integration import (  # noqa: E402
    CALCULATOR_INPUT_CRS_LABEL_BY_VALUE,
    CALCULATOR_INPUT_CRS_OPTIONS,
    CALCULATOR_OUTPUT_CRS_LABEL_BY_VALUE,
    CALCULATOR_OUTPUT_CRS_OPTIONS,
    CoordinateSystemError,
    DEFAULT_CRS,
    can_transform_crs,
    strict_transform_xy_to_crs,
)
from pywp.coordinate_systems import CoordinateSystem  # noqa: E402
from pywp.ui_theme import apply_page_style  # noqa: E402

_DEFAULT_OUTPUT_CRS = CoordinateSystem.WGS84_UTM_ZONE_43N
_DEFAULT_X = 600_010.6
_DEFAULT_Y = 7_407_421.0
_BATCH_EDITOR_COLUMNS = ("X", "Y")
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
    return _labels(CALCULATOR_INPUT_CRS_OPTIONS)[
        _index_for_crs(DEFAULT_CRS, CALCULATOR_INPUT_CRS_OPTIONS)
    ]


def _default_output_label() -> str:
    return _labels(CALCULATOR_OUTPUT_CRS_OPTIONS)[
        _index_for_crs(_DEFAULT_OUTPUT_CRS, CALCULATOR_OUTPUT_CRS_OPTIONS)
    ]


def _swap_crs_labels_supported(
    input_label: str,
    output_label: str,
) -> bool:
    input_labels = set(_labels(CALCULATOR_INPUT_CRS_OPTIONS))
    output_labels = set(_labels(CALCULATOR_OUTPUT_CRS_OPTIONS))
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
        return CALCULATOR_OUTPUT_CRS_LABEL_BY_VALUE.get(crs, crs.name)
    return CALCULATOR_INPUT_CRS_LABEL_BY_VALUE.get(crs, crs.name)


def _default_batch_editor_frame() -> pd.DataFrame:
    return pd.DataFrame([{column: None for column in _BATCH_EDITOR_COLUMNS}])


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
        x_out, y_out = strict_transform_xy_to_crs(
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
                "CRS": CALCULATOR_INPUT_CRS_LABEL_BY_VALUE.get(
                    input_crs, input_crs.name
                ),
                "X": _format_value(x_in, input_crs),
                "Y": _format_value(y_in, input_crs),
                "Роль": "Вход",
            },
            {
                "CRS": CALCULATOR_OUTPUT_CRS_LABEL_BY_VALUE.get(
                    output_crs, output_crs.name
                ),
                "X": _format_value(x_out, output_crs),
                "Y": _format_value(y_out, output_crs),
                "Роль": "Выход",
            },
        ]
    )


def run_page() -> None:
    st.set_page_config(page_title="Калькулятор СК", layout="wide")
    apply_page_style(max_width_px=1100)

    input_labels = _labels(CALCULATOR_INPUT_CRS_OPTIONS)
    output_labels = _labels(CALCULATOR_OUTPUT_CRS_OPTIONS)
    current_input_label = str(
        st.session_state.get("crs_calc_input_crs", _default_input_label())
    )
    current_output_label = str(
        st.session_state.get("crs_calc_output_crs", _default_output_label())
    )
    if current_input_label not in input_labels:
        st.session_state.pop("crs_calc_input_crs", None)
        st.session_state["crs_calc_x"] = None
        st.session_state["crs_calc_y"] = None
        st.session_state.pop("crs_calc_batch_points", None)
        current_input_label = _default_input_label()
        st.warning(
            "Прежняя входная CRS больше недоступна. "
            "Выберите прямоугольную CRS и заново введите координаты в метрах."
        )
    if current_output_label not in output_labels:
        st.session_state.pop("crs_calc_output_crs", None)
        current_output_label = _default_output_label()
    can_swap_crs_labels = _swap_crs_labels_supported(
        current_input_label,
        current_output_label,
    )
    c1, c_swap, c2 = st.columns([1.0, 0.18, 1.0], gap="small", vertical_alignment="bottom")
    input_label = c1.selectbox(
        "Входная CRS",
        options=input_labels,
        index=_index_for_crs(DEFAULT_CRS, CALCULATOR_INPUT_CRS_OPTIONS),
        key="crs_calc_input_crs",
    )
    c_swap.button(
        _SWAP_CRS_BUTTON_LABEL,
        key="crs_calc_swap_crs",
        disabled=not can_swap_crs_labels,
        on_click=_swap_crs_selection_state,
        width="stretch",
    )
    output_label = c2.selectbox(
        "Выходная CRS",
        options=output_labels,
        index=_index_for_crs(_DEFAULT_OUTPUT_CRS, CALCULATOR_OUTPUT_CRS_OPTIONS),
        key="crs_calc_output_crs",
    )

    input_crs = _crs_by_label(
        input_label,
        CALCULATOR_INPUT_CRS_OPTIONS,
        DEFAULT_CRS,
    )
    output_crs = _crs_by_label(
        output_label, CALCULATOR_OUTPUT_CRS_OPTIONS, _DEFAULT_OUTPUT_CRS
    )
    if input_crs.name.startswith("GSK_2011_ZONE_"):
        zone = int(input_crs.name.rsplit("_", 1)[1])
        st.caption(
            f"ГСК2011: шестиградусная зона {zone}, {input_crs.value}. "
            f"Восточный отсчёт X включает номер зоны в миллионах; "
            f"на центральном меридиане X = {zone * 1_000_000 + 500_000:,} м."
        )
    if output_crs.is_geographic():
        st.caption("Географический выход: X = долгота (E), Y = широта (N), градусы.")

    can_transform = can_transform_crs(input_crs, output_crs)
    if input_crs != output_crs and not can_transform:
        st.warning(
            "Для выбранной пары CRS нет безопасного прямого пересчёта; "
            "результат пересчёта недоступен."
        )
    single_tab, batch_tab = st.tabs(["Одна точка", "Таблица точек"])

    with single_tab:
        st.caption(
            f"{_crs_display_name(input_crs, output=False)}: "
            "X = восточная координата (E), Y = северная координата (N), метры."
        )
        x_col, y_col = st.columns(2, gap="small")
        x_in = x_col.number_input(
            "X (м)",
            value=st.session_state.get("crs_calc_x", float(_DEFAULT_X)),
            step=1.0,
            format="%.3f",
            key="crs_calc_x",
        )
        y_in = y_col.number_input(
            "Y (м)",
            value=st.session_state.get("crs_calc_y", float(_DEFAULT_Y)),
            step=1.0,
            format="%.3f",
            key="crs_calc_y",
        )
        if x_in is None or y_in is None:
            st.info(
                "Введите корректную прямоугольную точку X/Y в метрах, "
                "чтобы увидеть результат пересчёта."
            )
        else:
            try:
                x_out, y_out = strict_transform_xy_to_crs(
                    float(x_in),
                    float(y_in),
                    input_crs,
                    output_crs,
                )
            except CoordinateSystemError as exc:
                st.error(f"Пересчёт не выполнен: {exc}")
            else:
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
        editor_format = "%.3f"
        st.caption(
            "Входные X/Y для всех CRS задаются в метрах: "
            "X = восточная, Y = северная координата."
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
        try:
            batch_result = _batch_result_frame(
                points=batch_points,
                input_crs=input_crs,
                output_crs=output_crs,
            )
        except CoordinateSystemError as exc:
            st.error(f"Пересчёт таблицы не выполнен: {exc}")
            batch_result = pd.DataFrame()
        st.dataframe(batch_result, hide_index=True, width="stretch")


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(
            "Запустите приложение командой `streamlit run pages/04_crs_calculator.py`."
        )
    run_page()
