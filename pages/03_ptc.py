from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import streamlit as st

from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_well_panels import render_run_log_panel
from pywp.ui_well_result import (
    SingleWellResultView,
    render_key_metrics,
    render_result_plots,
    render_result_tables,
)


@lru_cache(maxsize=4)
def _load_welltrack_import_module_cached(
    page_path_str: str,
    mtime_ns: int,
) -> ModuleType:
    module_name = f"_ptc_welltrack_import_shared_{mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, page_path_str)
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось загрузить модуль WELLTRACK import.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_welltrack_import_module() -> ModuleType:
    page_path = Path(__file__).with_name("02_welltrack_import.py")
    return _load_welltrack_import_module_cached(
        str(page_path),
        int(page_path.stat().st_mtime_ns),
    )


def _force_ptc_defaults(wt: ModuleType) -> None:
    st.session_state["wt_log_verbosity"] = str(wt.WT_LOG_COMPACT)
    st.session_state["wt_3d_render_mode"] = str(wt.WT_3D_RENDER_DETAIL)
    st.session_state["wt_3d_backend"] = str(wt.WT_3D_BACKEND_THREE_LOCAL)
    if st.session_state.get("wt_prepared_recommendation_snapshot"):
        st.session_state["wt_prepared_well_overrides"] = {}
        st.session_state["wt_prepared_override_message"] = ""
        st.session_state["wt_prepared_recommendation_id"] = ""
        st.session_state["wt_anticollision_prepared_cluster_id"] = ""
        st.session_state["wt_prepared_recommendation_snapshot"] = None


def _render_ptc_import_section(wt: ModuleType) -> None:
    st.markdown("## 1. Импорт целей")
    st.caption(
        "Загрузите целевые точки скважин в формате WELLTRACK. После успешного "
        "импорта цели будут доступны для расчёта и визуализации."
    )
    source_payload = wt._render_source_input()
    parse_clicked = st.button(
        "Импорт целей",
        type="primary",
        icon=":material/upload_file:",
        width="content",
    )
    wt._handle_import_actions(
        source_payload=source_payload,
        parse_clicked=bool(parse_clicked),
        clear_clicked=False,
        reset_params_clicked=False,
    )


def _render_ptc_reference_kind_import_block(wt: ModuleType, *, kind: str) -> None:
    title = wt._reference_kind_title(kind)
    state_mode_key = f"ptc_reference_source_mode::{kind}"
    st.session_state.setdefault(state_mode_key, "Путь к WELLTRACK")
    mode = st.radio(
        f"Источник для {title.lower()}",
        options=["Путь к WELLTRACK", "Загрузить WELLTRACK"],
        key=state_mode_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.caption(str(wt._reference_kind_help(kind)))

    uploaded_file = None
    if mode == "Путь к WELLTRACK":
        st.text_input(
            "Путь к WELLTRACK",
            key=wt._reference_welltrack_path_key(kind),
            placeholder="tests/test_data/WELLTRACKS3.INC",
        )
    else:
        uploaded_file = st.file_uploader(
            f"WELLTRACK файл для {title.lower()}",
            type=["inc", "txt", "data", "ecl"],
            key=f"ptc_reference_{kind}_welltrack_file",
        )

    action_col, clear_col = st.columns([1.5, 1.0], gap="small")
    import_clicked = action_col.button(
        f"Загрузить {title.lower()}",
        key=f"ptc_reference_import_{kind}",
        type="primary",
        icon=":material/upload_file:",
        width="stretch",
    )
    clear_clicked = clear_col.button(
        f"Очистить {title.lower()}",
        key=f"ptc_reference_clear_{kind}",
        icon=":material/delete:",
        width="stretch",
    )

    if import_clicked:
        with st.status(f"Импорт {title.lower()}...", expanded=True) as status:
            try:
                if mode == "Путь к WELLTRACK":
                    parsed = wt.parse_reference_trajectory_welltrack_text(
                        wt._read_welltrack_file(
                            str(
                                st.session_state.get(
                                    wt._reference_welltrack_path_key(kind),
                                    "",
                                )
                            )
                        ),
                        kind=kind,
                    )
                else:
                    payload = b"" if uploaded_file is None else uploaded_file.getvalue()
                    parsed = wt.parse_reference_trajectory_welltrack_text(
                        wt._decode_welltrack_payload(
                            payload,
                            source_label=(
                                f"WELLTRACK `{getattr(uploaded_file, 'name', 'uploaded')}`"
                            ),
                        ),
                        kind=kind,
                    )
                wt._set_reference_wells_for_kind(kind=kind, wells=parsed)
                wt._reset_anticollision_view_state(clear_prepared=True)
                status.write(f"Загружено скважин: {len(parsed)}.")
                status.update(
                    label=f"{title} импортированы",
                    state="complete",
                    expanded=False,
                )
                st.rerun()
            except wt.WelltrackParseError as exc:
                status.write(str(exc))
                status.update(
                    label=f"Ошибка импорта: {title.lower()}",
                    state="error",
                    expanded=True,
                )

    if clear_clicked:
        wt._set_reference_wells_for_kind(kind=kind, wells=())
        st.session_state[wt._reference_welltrack_path_key(kind)] = ""
        wt._reset_anticollision_view_state(clear_prepared=True)
        st.rerun()

    current_wells = tuple(wt._reference_kind_wells(kind))
    if current_wells:
        st.caption(f"Загружено {len(current_wells)} скважин.")
    else:
        st.caption("Скважины этого типа не загружены.")


def _render_ptc_reference_section(wt: ModuleType) -> None:
    st.markdown("## 3. Загрузка фактического фонда")
    st.caption(
        "Если на месторождении уже есть фактический фонд или утверждённый "
        "проектный (проработанный в ЦСБ) - загрузите его в формате welltrack."
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("### Фактический фонд")
        _render_ptc_reference_kind_import_block(wt, kind=wt.REFERENCE_WELL_ACTUAL)
    with c2:
        st.markdown("### Утверждённый проектный фонд")
        _render_ptc_reference_kind_import_block(wt, kind=wt.REFERENCE_WELL_APPROVED)

    reference_wells = tuple(wt._reference_wells_from_state())
    if reference_wells:
        m1, m2, m3 = st.columns(3, gap="small")
        m1.metric("Доп. скважин", f"{len(reference_wells)}")
        m2.metric(
            "Фактических",
            f"{sum(1 for item in reference_wells if str(item.kind) == wt.REFERENCE_WELL_ACTUAL)}",
        )
        m3.metric(
            "Проектных утверждённых",
            f"{sum(1 for item in reference_wells if str(item.kind) == wt.REFERENCE_WELL_APPROVED)}",
        )

        st.dataframe(
            wt.arrow_safe_text_dataframe(
                pd.DataFrame(
                    [
                        {
                            "Скважина": wt.reference_well_display_label(item),
                            "Точек": int(len(item.stations)),
                            "MD max, м": float(item.stations["MD_m"].iloc[-1]),
                        }
                        for item in reference_wells
                    ]
                )
            ),
            width="stretch",
            hide_index=True,
        )

    actual_wells = tuple(wt._reference_kind_wells(wt.REFERENCE_WELL_ACTUAL))
    if actual_wells:
        analyses = wt._actual_fund_analyses(actual_wells)
        wt._render_actual_fund_analysis_panel(analyses=analyses)


def _render_ptc_run_section(wt: ModuleType, *, records: list[object]) -> None:
    st.markdown("## 4. Расчёт траекторий")
    st.caption("Детализация лога зафиксирована: краткий режим.")
    summary_rows = st.session_state.get("wt_summary_rows")
    wt._render_batch_selection_status(records=records, summary_rows=summary_rows)
    all_names, _ = wt._sync_selection_state(records=records)
    pads, _, well_names_by_pad_id = wt._pad_membership(records)
    pad_ids = [str(pad.pad_id) for pad in pads]
    if pad_ids and str(st.session_state.get("wt_batch_select_pad_id", "")).strip() not in pad_ids:
        st.session_state["wt_batch_select_pad_id"] = pad_ids[0]

    with st.form("ptc_run_form", clear_on_submit=False):
        select_col, pad_col, action_col, pad_add_col, pad_only_col = st.columns(
            [5.0, 2.4, 1.2, 1.45, 1.45],
            gap="small",
            vertical_alignment="bottom",
        )
        with select_col:
            st.multiselect(
                "Скважины для расчёта",
                options=all_names,
                key="wt_selected_names",
            )
        with action_col:
            select_all_clicked = st.form_submit_button(
                "Выбрать все",
                icon=":material/done_all:",
                width="stretch",
            )
        with pad_col:
            if len(pad_ids) > 1:
                st.selectbox(
                    "Куст",
                    options=pad_ids,
                    format_func=lambda value: wt._pad_display_label(
                        next(pad for pad in pads if str(pad.pad_id) == str(value))
                    ),
                    key="wt_batch_select_pad_id",
                )
        with pad_add_col:
            add_pad_clicked = (
                st.form_submit_button(
                    "Добавить куст",
                    icon=":material/filter_alt:",
                    width="stretch",
                )
                if len(pad_ids) > 1
                else False
            )
        with pad_only_col:
            replace_with_pad_clicked = (
                st.form_submit_button(
                    "Только куст",
                    icon=":material/rule:",
                    width="stretch",
                )
                if len(pad_ids) > 1
                else False
            )

        with st.expander("Параметры расчёта", expanded=False):
            config = wt._build_config_form(binding=wt.WT_CALC_PARAMS, title="")

        run_clicked = st.form_submit_button(
            "Рассчитать траектории",
            type="primary",
            icon=":material/play_arrow:",
        )

    if select_all_clicked:
        st.session_state["wt_pending_selected_names"] = list(all_names)
        st.rerun()
    if add_pad_clicked:
        selected_pad_id = str(st.session_state.get("wt_batch_select_pad_id", "")).strip()
        current_selected = [
            str(name) for name in st.session_state.get("wt_selected_names", [])
        ]
        st.session_state["wt_pending_selected_names"] = list(
            dict.fromkeys([*current_selected, *well_names_by_pad_id.get(selected_pad_id, ())])
        )
        st.rerun()
    if replace_with_pad_clicked:
        selected_pad_id = str(st.session_state.get("wt_batch_select_pad_id", "")).strip()
        st.session_state["wt_pending_selected_names"] = list(
            well_names_by_pad_id.get(selected_pad_id, ())
        )
        st.rerun()

    wt._run_batch_if_clicked(
        requests=[
            wt._BatchRunRequest(
                selected_names=list(st.session_state.get("wt_selected_names", [])),
                config=config,
                run_clicked=bool(run_clicked),
            )
        ],
        records=records,
    )


def _render_ptc_anticollision_panel(
    wt: ModuleType,
    *,
    successes: list[object],
    records: list[object],
    focus_pad_id: str,
) -> None:
    reference_wells = wt._reference_wells_from_state()
    if len(successes) + len(reference_wells) < 2:
        st.info("Для anti-collision нужно минимум две успешно рассчитанные скважины.")
        return

    custom_actual_fund_model = wt._actual_fund_custom_model_from_state()
    preset_options = list(wt.UNCERTAINTY_PRESET_OPTIONS.keys())
    if custom_actual_fund_model is not None:
        preset_options.append(wt.UNCERTAINTY_PRESET_CUSTOM_ACTUAL_FUND)
    normalized_preset = wt.normalize_uncertainty_preset(
        st.session_state.get(
            "wt_anticollision_uncertainty_preset",
            wt.DEFAULT_UNCERTAINTY_PRESET,
        ),
        allow_custom=custom_actual_fund_model is not None,
    )
    if normalized_preset not in preset_options:
        normalized_preset = wt.DEFAULT_UNCERTAINTY_PRESET
    st.session_state["wt_anticollision_uncertainty_preset"] = normalized_preset
    selected_preset = st.selectbox(
        "Пресет неопределенности для anti-collision",
        options=preset_options,
        format_func=wt.uncertainty_preset_label,
        key="wt_anticollision_uncertainty_preset",
    )
    uncertainty_model = wt.planning_uncertainty_model_for_preset(
        selected_preset,
        custom_model=custom_actual_fund_model,
    )
    analysis, recommendations, clusters = wt._cached_anti_collision_view_model(
        successes=successes,
        uncertainty_model=uncertainty_model,
        records=records,
        reference_wells=reference_wells,
    )
    focus_pad_well_names = wt._focus_pad_well_names(
        records=records,
        focus_pad_id=focus_pad_id,
    )
    visible_clusters = wt._clusters_touching_focus_pad(
        clusters=clusters,
        focus_pad_well_names=focus_pad_well_names,
    )
    visible_recommendations = wt._recommendations_for_clusters(
        recommendations=recommendations,
        clusters=visible_clusters,
    )
    focus_anticollision_well_names = wt._anticollision_focus_well_names(
        clusters=visible_clusters,
        focus_pad_well_names=focus_pad_well_names,
    )

    st.caption(
        f"Пресет: {wt.uncertainty_preset_label(selected_preset)}. "
        f"{wt.anti_collision_method_caption(uncertainty_model)}"
    )

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Проверено пар", f"{int(analysis.pair_count)}")
    m2.metric("Пар с overlap", f"{int(analysis.overlapping_pair_count)}")
    m3.metric("Пересечения в t1/t3", f"{int(analysis.target_overlap_pair_count)}")
    worst_sf = analysis.worst_separation_factor
    m4.metric("Минимальный SF", "—" if worst_sf is None else f"{float(worst_sf):.2f}")
    with st.expander("Что такое SF?", expanded=False):
        st.markdown(wt._sf_help_markdown())

    chart_col1, chart_col2 = st.columns(2, gap="medium")
    anticollision_3d_figure = wt._all_wells_anticollision_3d_figure(
        analysis,
        previous_successes_by_name={},
        focus_well_names=focus_anticollision_well_names or focus_pad_well_names,
        render_mode=wt.WT_3D_RENDER_DETAIL,
    )
    wt._render_plotly_or_three_3d(
        container=chart_col1,
        figure=anticollision_3d_figure,
        backend=wt.WT_3D_BACKEND_THREE_LOCAL,
        height=660,
        payload_overrides=wt._anticollision_three_payload_overrides(
            records=records,
            analysis=analysis,
        ),
    )
    chart_col2.plotly_chart(
        wt._all_wells_anticollision_plan_figure(
            analysis,
            previous_successes_by_name={},
            focus_well_names=focus_anticollision_well_names or focus_pad_well_names,
        ),
        width="stretch",
    )

    if not analysis.zones:
        st.success(
            "Пересечения 2σ конусов неопределенности не обнаружены для рассчитанного набора."
        )
        return

    target_zones = [zone for zone in analysis.zones if int(zone.priority_rank) < 2]
    if target_zones:
        st.warning(
            "Найдены пересечения, затрагивающие точки целей t1/t3. Они вынесены "
            "в начало отчета и должны разбираться в первую очередь."
        )
    else:
        st.warning("Найдены пересечения 2σ конусов неопределенности по траекториям.")

    report_rows = (
        wt._report_rows_from_recommendations(visible_recommendations)
        if focus_pad_well_names
        else wt.anti_collision_report_rows(analysis)
    )
    st.markdown("### Отчет по anti-collision")
    st.dataframe(
        wt.arrow_safe_text_dataframe(pd.DataFrame(report_rows)),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Рекомендации")
    st.dataframe(
        wt.arrow_safe_text_dataframe(
            pd.DataFrame(wt.anti_collision_recommendation_rows(visible_recommendations))
        ),
        width="stretch",
        hide_index=True,
    )


def _render_ptc_success_tabs(
    wt: ModuleType,
    *,
    successes: list[object],
    records: list[object],
    summary_rows: list[dict[str, object]],
) -> None:
    name_to_color = wt._well_color_map(records)
    reference_wells = wt._reference_wells_from_state()
    target_only_wells = wt._failed_target_only_wells(
        records=records,
        summary_rows=summary_rows,
    )
    view_mode = st.radio(
        "Режим просмотра результатов",
        options=["Отдельная скважина", "Все скважины"],
        key="wt_results_view_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    if str(view_mode) == "Отдельная скважина":
        selected_name = st.selectbox("Скважина", options=[item.name for item in successes])
        selected = wt._ensure_selected_success_baseline(
            selected_name=str(selected_name),
            successes=successes,
        )
        well_view = SingleWellResultView(
            well_name=str(selected.name),
            surface=selected.surface,
            t1=selected.t1,
            t3=selected.t3,
            stations=selected.stations,
            summary=selected.summary,
            config=selected.config,
            azimuth_deg=float(selected.azimuth_deg),
            md_t1_m=float(selected.md_t1_m),
            runtime_s=selected.runtime_s,
            baseline_summary=selected.baseline_summary,
            baseline_runtime_s=selected.baseline_runtime_s,
            issue_messages=(
                (str(selected.md_postcheck_message),)
                if str(selected.md_postcheck_message).strip()
                else ()
            ),
            trajectory_line_dash="dash" if bool(selected.md_postcheck_exceeded) else "solid",
        )
        t1_horizontal_offset_m = render_key_metrics(
            view=well_view,
            title="Ключевые показатели",
            border=True,
        )
        render_result_plots(
            view=well_view,
            title_trajectory=None,
            title_plan=None,
            border=True,
        )
        render_result_tables(
            view=well_view,
            t1_horizontal_offset_m=t1_horizontal_offset_m,
            summary_tab_label="Сводка",
            survey_tab_label="Инклинометрия",
            survey_file_name=f"{selected_name}_survey.csv",
            show_validation_section=False,
            show_solver_diagnostics_section=False,
        )
        return

    all_view_mode = st.radio(
        "Режим отображения всех скважин",
        options=["Траектории", "Anti-collision"],
        key="wt_results_all_view_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    pads, _, _ = wt._pad_membership(records)
    if len(pads) > 1:
        focus_options = [wt.WT_PAD_FOCUS_ALL, *(str(pad.pad_id) for pad in pads)]
        normalized_focus_pad_id = wt._normalize_focus_pad_id(
            records=records,
            requested_pad_id=st.session_state.get("wt_results_focus_pad_id"),
        )
        if normalized_focus_pad_id != str(st.session_state.get("wt_results_focus_pad_id", "")):
            st.session_state["wt_results_focus_pad_id"] = normalized_focus_pad_id
        st.selectbox(
            "Фокус камеры по кусту",
            options=focus_options,
            format_func=lambda value: (
                "Все кусты"
                if str(value) == wt.WT_PAD_FOCUS_ALL
                else wt._pad_display_label(
                    next(pad for pad in pads if str(pad.pad_id) == str(value))
                )
            ),
            key="wt_results_focus_pad_id",
        )
    focus_pad_id = wt._normalize_focus_pad_id(
        records=records,
        requested_pad_id=st.session_state.get("wt_results_focus_pad_id"),
    )
    focus_pad_well_names = wt._focus_pad_well_names(
        records=records,
        focus_pad_id=focus_pad_id,
    )

    if str(all_view_mode) == "Траектории":
        if target_only_wells:
            st.caption(
                "Для непростроенных скважин на обзорных графиках показаны только "
                "точки S/t1/t3, без траектории."
            )
        if reference_wells:
            st.caption(
                "Дополнительные фактические и утвержденные скважины показаны как "
                "reference-траектории: серые и красные линии без точек S/t1/t3."
            )
        c1, c2 = st.columns(2, gap="medium")
        overview_3d_figure = wt._all_wells_3d_figure(
            successes,
            target_only_wells=target_only_wells,
            reference_wells=reference_wells,
            name_to_color=name_to_color,
            focus_well_names=focus_pad_well_names,
            render_mode=wt.WT_3D_RENDER_DETAIL,
        )
        wt._render_plotly_or_three_3d(
            container=c1,
            figure=overview_3d_figure,
            backend=wt.WT_3D_BACKEND_THREE_LOCAL,
            height=620,
            payload_overrides=wt._trajectory_three_payload_overrides(
                records=records,
                successes=successes,
                target_only_wells=target_only_wells,
                name_to_color=name_to_color,
            ),
        )
        c2.plotly_chart(
            wt._all_wells_plan_figure(
                successes,
                target_only_wells=target_only_wells,
                reference_wells=reference_wells,
                name_to_color=name_to_color,
                focus_well_names=focus_pad_well_names,
            ),
            width="stretch",
        )
        return

    _render_ptc_anticollision_panel(
        wt,
        successes=successes,
        records=records,
        focus_pad_id=focus_pad_id,
    )


def run_page() -> None:
    wt = _load_welltrack_import_module()
    st.set_page_config(page_title="PTC", layout="wide")
    wt._init_state()
    _force_ptc_defaults(wt)
    apply_page_style(max_width_px=1700)
    render_hero(title="PTC", subtitle="Prototype trajectory constructor")

    _render_ptc_import_section(wt)
    records = st.session_state.get("wt_records")
    if records is None:
        st.info("Загрузите цели и нажмите «Импорт целей».")
        return
    if not records:
        st.warning("В источнике не найдено ни одной скважины.")
        return

    wt._render_records_overview(records=records)
    wt._render_raw_records_table(records=records)

    st.markdown("## 2. Кусты и расчёт устьев")
    wt._render_t1_t3_order_panel(records=records)
    wt._render_pad_layout_panel(records=records)

    _render_ptc_reference_section(wt)
    _render_ptc_run_section(wt, records=records)

    if st.session_state.get("wt_last_error"):
        wt.render_solver_diagnostics(st.session_state["wt_last_error"])

    st.markdown("## 5. Результаты расчёта")
    render_run_log_panel(
        st.session_state.get("wt_last_run_log_lines"),
        border=False,
    )

    summary_rows = st.session_state.get("wt_summary_rows")
    successes = st.session_state.get("wt_successes")
    if not summary_rows:
        render_small_note("Результаты расчёта появятся после запуска расчёта траекторий.")
        return
    wt._render_batch_summary(summary_rows=summary_rows)
    if not successes:
        st.warning("Все выбранные скважины завершились ошибками расчёта.")
        return
    _render_ptc_success_tabs(
        wt,
        successes=successes,
        records=list(records),
        summary_rows=list(summary_rows),
    )


if __name__ == "__main__":
    run_page()
