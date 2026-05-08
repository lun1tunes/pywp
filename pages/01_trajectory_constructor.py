from __future__ import annotations

import logging

# Suppress noisy Streamlit warnings BEFORE importing streamlit
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from pywp import ptc_core as wt
from pywp.solver_diagnostics_ui import render_solver_diagnostics
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_well_panels import render_run_log_panel
from pywp.coordinate_integration import (
    DEFAULT_CRS,
    csv_export_crs,
    get_crs_display_suffix,
    get_selected_crs,
    render_crs_sidebar,
    should_auto_convert,
    transform_stations_to_crs,
)
from pywp.models import TrajectoryConfig
from pywp.ui_well_result import (
    SingleWellResultView,
    render_key_metrics,
    render_result_plots,
    render_result_tables,
)

PTC_CALC_PARAMS_EXPAND_ONCE_KEY = "ptc_calc_params_expand_once"


def _force_ptc_defaults() -> None:
    if str(st.session_state.get("wt_log_verbosity", "")).strip() not in set(
        wt.WT_LOG_LEVEL_OPTIONS
    ):
        st.session_state["wt_log_verbosity"] = str(wt.WT_LOG_COMPACT)
    if str(st.session_state.get("wt_3d_render_mode", "")).strip() not in set(
        wt.WT_3D_RENDER_OPTIONS
    ):
        st.session_state["wt_3d_render_mode"] = str(wt.WT_3D_RENDER_DETAIL)
    if str(st.session_state.get("wt_3d_backend", "")).strip() not in set(
        wt.WT_3D_BACKEND_OPTIONS
    ):
        st.session_state["wt_3d_backend"] = str(wt.WT_3D_BACKEND_THREE_LOCAL)
    if wt._pending_edit_target_names():
        st.session_state["wt_results_view_mode"] = "Все скважины"
    st.session_state["wt_results_all_view_mode"] = "Anti-collision"
    if st.session_state.get("wt_prepared_recommendation_snapshot"):
        st.session_state["wt_prepared_well_overrides"] = {}
        st.session_state["wt_prepared_override_message"] = ""
        st.session_state["wt_prepared_recommendation_id"] = ""
        st.session_state["wt_anticollision_prepared_cluster_id"] = ""
        st.session_state["wt_prepared_recommendation_snapshot"] = None


def _keep_ptc_calc_params_expanded() -> None:
    st.session_state[PTC_CALC_PARAMS_EXPAND_ONCE_KEY] = True


def _render_ptc_calc_params_panel() -> TrajectoryConfig:
    expanded = bool(st.session_state.pop(PTC_CALC_PARAMS_EXPAND_ONCE_KEY, False))
    with st.expander("Параметры расчёта", expanded=expanded):
        return wt._build_config_form(
            binding=wt.WT_CALC_PARAMS,
            title="",
            on_change=_keep_ptc_calc_params_expanded,
        )


def _render_ptc_import_section() -> None:
    st.markdown("## 1. Импорт целей")

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


def _render_ptc_reference_kind_import_block(*, kind: str) -> None:
    title = wt._reference_kind_title(kind)
    state_mode_key = f"ptc_reference_source_mode::{kind}"
    source_options = [
        "Загрузить .dev",
        "Путь к WELLTRACK",
        "Загрузить WELLTRACK",
    ]
    if state_mode_key not in st.session_state:
        legacy_mode = str(
            st.session_state.get(wt._reference_source_mode_key(kind), "")
        ).strip()
        st.session_state[state_mode_key] = (
            legacy_mode if legacy_mode in source_options else "Загрузить .dev"
        )
    if str(st.session_state.get(state_mode_key, "")).strip() not in set(
        source_options
    ):
        st.session_state[state_mode_key] = "Загрузить .dev"
    mode = st.radio(
        f"Источник для {title.lower()}",
        options=source_options,
        key=state_mode_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state[wt._reference_source_mode_key(kind)] = mode
    # st.caption(str(wt._reference_kind_help(kind)))

    uploaded_file = None
    if mode == "Загрузить .dev":
        folder_count_key = wt._reference_dev_folder_count_key(kind)
        try:
            folder_count = int(st.session_state.get(folder_count_key, 1))
        except (TypeError, ValueError):
            folder_count = 1
        folder_count = max(1, folder_count)
        st.session_state[folder_count_key] = folder_count
        for index in range(folder_count):
            st.text_input(
                "Папка с .dev файлами"
                if index == 0
                else f"Папка с .dev файлами #{index + 1}",
                key=wt._reference_dev_folder_path_key(kind, index),
                placeholder="tests/test_data/dev_fact",
            )
        if st.button(
            "Добавить ещё папку",
            key=f"ptc_reference_{kind}_add_dev_folder",
            icon=":material/create_new_folder:",
            use_container_width=True,
        ):
            st.session_state[folder_count_key] = folder_count + 1
            st.rerun()
        st.caption(
            "Импортируются все `.dev` файлы из папок. "
            "Имя берется из файла без `.dev`, "
            "координаты - из колонок `MD X Y Z`."
        )
    elif mode == "Путь к WELLTRACK":
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

    action_col, clear_col = st.columns(2, gap="small")
    import_clicked = action_col.button(
        f"Загрузить {title.lower()}",
        key=f"ptc_reference_import_{kind}",
        type="primary",
        icon=":material/upload_file:",
        use_container_width=True,
    )
    clear_clicked = clear_col.button(
        f"Очистить {title.lower()}",
        key=f"ptc_reference_clear_{kind}",
        icon=":material/delete:",
        use_container_width=True,
    )

    if import_clicked:
        with st.status(f"Импорт {title.lower()}...", expanded=True) as status:
            try:
                if mode == "Загрузить .dev":
                    parsed = wt.parse_reference_trajectory_dev_directories(
                        wt._reference_dev_folder_paths(kind),
                        kind=kind,
                    )
                elif mode == "Путь к WELLTRACK":
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
                    payload = (
                        b""
                        if uploaded_file is None
                        else uploaded_file.getvalue()
                    )
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
        wt._clear_reference_dev_folder_state(kind)
        wt._reset_anticollision_view_state(clear_prepared=True)
        st.rerun()

    current_wells = tuple(wt._reference_kind_wells(kind))
    if current_wells:
        st.caption(f"Загружено {len(current_wells)} скважин.")
    else:
        st.caption("Скважины этого типа не загружены.")


def _render_ptc_reference_section() -> None:
    st.markdown("## 3. Загрузка фактического фонда")
    st.caption(
        "Если на месторождении уже есть фактический фонд или утверждённый "
        "проектный (проработанный в ЦСБ) - загрузите его из папок `.dev` "
        "или в формате WELLTRACK."
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("### Фактический фонд")
        _render_ptc_reference_kind_import_block(kind=wt.REFERENCE_WELL_ACTUAL)
    with c2:
        st.markdown("### Утверждённый проектный фонд")
        _render_ptc_reference_kind_import_block(
            kind=wt.REFERENCE_WELL_APPROVED
        )

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

        with st.expander(
            "Список загруженных фактических/ проектных скважин",
            expanded=False,
        ):
            st.dataframe(
                wt.arrow_safe_text_dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Скважина": wt.reference_well_display_label(
                                    item
                                ),
                                "Точек": int(len(item.stations)),
                                "MD max, м": float(
                                    item.stations["MD_m"].iloc[-1]
                                ),
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

    approved_wells = tuple(
        wt._reference_kind_wells(wt.REFERENCE_WELL_APPROVED)
    )
    if approved_wells:
        with st.expander(
            "Просмотр загруженных утверждённых проектных скважин",
            expanded=False,
        ):
            try:
                approved_analyses = wt.build_actual_fund_well_analyses(
                    approved_wells
                )
            except Exception as exc:
                st.error(
                    "Не удалось построить просмотр утверждённого проектного фонда."
                )
                st.caption(f"{type(exc).__name__}: {exc}")
            else:
                wt._render_reference_well_detail(
                    approved_analyses,
                    select_label="Просмотр утвержденной проектной скважины",
                    selected_key="wt_approved_fund_selected_well",
                )


def _render_ptc_run_section(*, records: list[object]) -> None:
    st.markdown("## 4. Расчёт траекторий")
    summary_rows = st.session_state.get("wt_summary_rows")
    wt._render_batch_selection_status(
        records=records, summary_rows=summary_rows
    )
    all_names, _ = wt._sync_selection_state(records=records)
    pads, _, well_names_by_pad_id = wt._pad_membership(records)
    pad_ids = [str(pad.pad_id) for pad in pads]
    if (
        pad_ids
        and str(st.session_state.get("wt_batch_select_pad_id", "")).strip()
        not in pad_ids
    ):
        st.session_state["wt_batch_select_pad_id"] = pad_ids[0]

    config = _render_ptc_calc_params_panel()

    with st.form("ptc_run_form", clear_on_submit=False):
        select_col, pad_col, action_col, pad_add_col, pad_only_col = (
            st.columns(
                [5.0, 2.4, 1.2, 1.45, 1.45],
                gap="small",
                vertical_alignment="bottom",
            )
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
                        next(
                            pad
                            for pad in pads
                            if str(pad.pad_id) == str(value)
                        )
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

        _parallel_options = [
            ("Без Multiprocessing", 0),
            *((f"{n} процессов", n) for n in (2, 4, 6, 8, 12, 16, 24, 32)),
        ]
        _parallel_labels = [label for label, _ in _parallel_options]
        _parallel_values = {label: value for label, value in _parallel_options}
        _parallel_label = st.selectbox(
            "Параллельный расчёт",
            options=_parallel_labels,
            index=0,
            key="wt_parallel_workers_label_01_constructor",
            help=(
                "Количество параллельных процессов для batch-расчёта. "
                "Ускоряет расчёт при большом числе скважин за счёт "
                "использования нескольких ядер CPU."
            ),
        )
        _parallel_workers = _parallel_values.get(str(_parallel_label), 0)

        run_clicked = st.form_submit_button(
            "Рассчитать траектории",
            type="primary",
            icon=":material/play_arrow:",
        )

    if select_all_clicked:
        st.session_state["wt_pending_selected_names"] = list(all_names)
        st.rerun()
    if add_pad_clicked:
        selected_pad_id = str(
            st.session_state.get("wt_batch_select_pad_id", "")
        ).strip()
        current_selected = [
            str(name) for name in st.session_state.get("wt_selected_names", [])
        ]
        st.session_state["wt_pending_selected_names"] = list(
            dict.fromkeys(
                [
                    *current_selected,
                    *well_names_by_pad_id.get(selected_pad_id, ()),
                ]
            )
        )
        st.rerun()
    if replace_with_pad_clicked:
        selected_pad_id = str(
            st.session_state.get("wt_batch_select_pad_id", "")
        ).strip()
        st.session_state["wt_pending_selected_names"] = list(
            well_names_by_pad_id.get(selected_pad_id, ())
        )
        st.rerun()

    wt._run_batch_if_clicked(
        requests=[
            wt._BatchRunRequest(
                selected_names=list(
                    st.session_state.get("wt_selected_names", [])
                ),
                config=config,
                run_clicked=bool(run_clicked),
                parallel_workers=int(_parallel_workers),
            )
        ],
        records=records,
    )
    if (
        run_clicked
        and st.session_state.get("wt_summary_rows")
        and not st.session_state.get("wt_last_error")
    ):
        st.session_state["wt_results_view_mode"] = "Все скважины"
        st.session_state["wt_results_all_view_mode"] = "Anti-collision"


def _render_ptc_anticollision_panel(
    *,
    successes: list[object],
    records: list[object],
    focus_pad_id: str,
    focus_pad_well_names: list[str],
) -> None:
    pending_edit_names = wt._pending_edit_target_names()
    if pending_edit_names:
        st.info(
            "Anti-collision анализ приостановлен: есть изменённые в 3D точки "
            "t1/t3, которые ещё не пересчитаны."
        )
        st.caption(
            "Сначала пересчитайте скважины: "
            + ", ".join(pending_edit_names)
            + ". Предыдущий anti-collision расчёт сохранён и не будет "
            "перезапускаться по неполному набору."
        )
        return

    reference_wells = wt._reference_wells_from_state()
    if len(successes) + len(reference_wells) < 2:
        st.info(
            "Для anti-collision нужно минимум две успешно рассчитанные скважины."
        )
        return

    preset_options = list(wt.UNCERTAINTY_PRESET_OPTIONS.keys())
    normalized_preset = wt.normalize_uncertainty_preset(
        st.session_state.get(
            "wt_anticollision_uncertainty_preset",
            wt.DEFAULT_UNCERTAINTY_PRESET,
        ),
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
    uncertainty_model = wt.planning_uncertainty_model_for_preset(selected_preset)

    anti_collision_progress = st.progress(
        8, text="Подготовка anti-collision анализа..."
    )

    def _anti_collision_progress_update(value: int, text: str) -> None:
        anti_collision_progress.progress(int(value), text=text)

    try:
        analysis, recommendations, clusters = (
            wt._cached_anti_collision_view_model(
                successes=successes,
                uncertainty_model=uncertainty_model,
                records=records,
                reference_wells=reference_wells,
                progress_callback=_anti_collision_progress_update,
            )
        )
    except Exception as exc:
        anti_collision_progress.empty()
        wt._store_anticollision_failure_state(exc)
        st.error(
            "Не удалось построить anti-collision анализ. Проверьте лог расчёта ниже."
        )
        wt._render_status_run_log(
            title="Лог расчёта Anti-collision",
            state_payload=st.session_state.get("wt_anticollision_last_run"),
            empty_message="Anti-collision анализ ещё не запускался.",
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return
    anti_collision_progress.empty()
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

    # st.caption(
    #     f"Пресет: {wt.uncertainty_preset_label(selected_preset)}. "
    #     f"{wt.anti_collision_method_caption(uncertainty_model)}"
    # )
    wt._render_status_run_log(
        title="Лог расчёта Anti-collision",
        state_payload=st.session_state.get("wt_anticollision_last_run"),
        empty_message="Anti-collision анализ ещё не запускался.",
    )

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Проверено пар", f"{int(analysis.pair_count)}")
    m2.metric("Пар с overlap", f"{int(analysis.overlapping_pair_count)}")
    m3.metric(
        "Пересечения в t1/t3", f"{int(analysis.target_overlap_pair_count)}"
    )
    worst_sf = analysis.worst_separation_factor
    m4.metric(
        "Минимальный SF", "—" if worst_sf is None else f"{float(worst_sf):.2f}"
    )
    with st.expander("Что такое SF?", expanded=False):
        st.markdown(wt._sf_help_markdown())

    chart_col1, chart_col2 = st.columns(2, gap="medium")
    try:
        anticollision_3d_figure = wt._all_wells_anticollision_3d_figure(
            analysis,
            previous_successes_by_name={},
            focus_well_names=focus_anticollision_well_names
            or focus_pad_well_names,
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
                successes=successes,
            ),
        )
        chart_col2.plotly_chart(
            wt._all_wells_anticollision_plan_figure(
                analysis,
                previous_successes_by_name={},
                focus_well_names=focus_anticollision_well_names
                or focus_pad_well_names,
            ),
            width="stretch",
        )
    except Exception as exc:
        wt._store_anticollision_failure_state(exc)
        st.error(
            "Не удалось отрисовать anti-collision визуализацию. Проверьте лог расчёта ниже."
        )
        wt._render_status_run_log(
            title="Лог расчёта Anti-collision",
            state_payload=st.session_state.get("wt_anticollision_last_run"),
            empty_message="Anti-collision анализ ещё не запускался.",
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    if not analysis.zones:
        st.success(
            "Пересечения 2σ конусов неопределенности не обнаружены для рассчитанного набора."
        )
        return

    target_zones = [
        zone for zone in analysis.zones if int(zone.priority_rank) < 2
    ]
    if target_zones:
        st.warning(
            "Найдены пересечения, затрагивающие точки целей t1/t3. Они вынесены "
            "в начало отчета и должны разбираться в первую очередь."
        )
    else:
        st.warning(
            "Найдены пересечения 2σ конусов неопределенности по траекториям."
        )

    report_rows = (
        wt._report_rows_from_recommendations(visible_recommendations, analysis)
        if visible_recommendations
        else wt.anti_collision_report_rows(analysis)
    )
    st.markdown("### Отчет по anti-collision")
    st.dataframe(
        wt.arrow_safe_text_dataframe(pd.DataFrame(report_rows)),
        width="stretch",
        hide_index=True,
    )

    if focus_pad_well_names and len(focus_pad_well_names) >= 2:
        st.caption(
            "Скважины, для которых зафиксирован порядок на кусте, не участвуют в оптимизации."
        )
        fixed_pad_well_names = wt._focus_pad_fixed_well_names(
            records=records,
            focus_pad_id=focus_pad_id,
        )
        movable_pad_well_names = sorted(
            set(str(name) for name in focus_pad_well_names)
            - set(str(name) for name in fixed_pad_well_names)
        )
        if fixed_pad_well_names:
            st.caption(
                "Фиксированный порядок не изменяется: "
                + ", ".join(str(name) for name in fixed_pad_well_names)
                + ". Оптимизатор переставляет только остальные скважины."
            )
        optimization_disabled = len(movable_pad_well_names) < 2
        if optimization_disabled:
            st.info(
                "Для оптимизации нужно минимум две незафиксированные скважины "
                "на выбранном кусте."
            )
        if st.button(
            "✨ Оптимизировать порядок скважин на кусте",
            type="primary",
            use_container_width=True,
            disabled=optimization_disabled,
        ):
            from pywp.pad_optimization import optimize_pad_order

            opt_progress = st.progress(0, text="Инициализация оптимизации...")

            def opt_callback(percent: int, msg: str) -> None:
                opt_progress.progress(int(percent), text=msg)

            config_by_name = {
                s.name: s.config
                for s in successes
                if s.name in focus_pad_well_names
            }
            success_dict = {s.name: s for s in successes}

            new_records, new_success_dict, improved = optimize_pad_order(
                records=records,
                success_dict=success_dict,
                pad_well_names=focus_pad_well_names,
                uncertainty_model=uncertainty_model,
                reference_wells=list(reference_wells),
                config_by_name=config_by_name,
                progress_callback=opt_callback,
                fixed_well_names=set(fixed_pad_well_names),
            )

            if improved:
                from pywp.welltrack_batch import (
                    WelltrackBatchPlanner,
                    merge_batch_results,
                )

                new_rows = []
                for r in new_records:
                    if r.name in new_success_dict:
                        new_rows.append(
                            WelltrackBatchPlanner._row_from_success(
                                record=r, success=new_success_dict[r.name]
                            )
                        )

                merged_rows, merged_successes = merge_batch_results(
                    records=new_records,
                    existing_rows=st.session_state.get("wt_summary_rows"),
                    existing_successes=st.session_state.get("wt_successes"),
                    new_rows=new_rows,
                    new_successes=list(new_success_dict.values()),
                )

                st.session_state["wt_records"] = new_records
                st.session_state["wt_successes"] = merged_successes
                st.session_state["wt_summary_rows"] = merged_rows
                wt._reset_anticollision_view_state(clear_prepared=True)
                opt_progress.progress(
                    100, text="Оптимизация завершена! Применяем изменения..."
                )
                st.rerun()
            else:
                opt_progress.progress(
                    100,
                    text="Текущий порядок уже оптимален (или улучшить не удалось).",
                )


def _render_ptc_success_tabs(
    successes: list[object],
    records: list[object],
    summary_rows: list[dict[str, object]],
) -> None:
    """Render success tabs with coordinate system support."""
    # Get selected CRS for coordinate transformation
    selected_crs = get_selected_crs()
    auto_convert = should_auto_convert()
    name_to_color = wt._well_color_map(list(records))
    reference_wells = tuple(wt._reference_wells_from_state())
    target_only_wells = wt._failed_target_only_wells(
        records=list(records),
        summary_rows=list(summary_rows),
    )
    view_mode = st.radio(
        "Режим просмотра результатов",
        options=["Отдельная скважина", "Все скважины"],
        key="wt_results_view_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    if str(view_mode) == "Отдельная скважина":
        selected_name = st.selectbox(
            "Скважина", options=[item.name for item in successes]
        )
        selected = wt._find_selected_success(
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
            issue_messages=(
                (str(selected.md_postcheck_message),)
                if str(selected.md_postcheck_message).strip()
                else ()
            ),
            trajectory_line_dash=(
                "dash" if bool(selected.md_postcheck_exceeded) else "solid"
            ),
        )
        survey_export_stations = None
        survey_export_xy_label_suffix = ""
        survey_export_xy_unit = "м"
        survey_export_crs = csv_export_crs(
            selected_crs,
            DEFAULT_CRS,
            auto_convert=auto_convert,
        )
        if (
            auto_convert
            and selected_crs != DEFAULT_CRS
            and survey_export_crs == selected_crs
        ):
            survey_export_stations = transform_stations_to_crs(
                selected.stations,
                selected_crs,
                DEFAULT_CRS,
                rename_columns=False,
            )
        if auto_convert and selected_crs != DEFAULT_CRS:
            survey_export_xy_label_suffix = get_crs_display_suffix(survey_export_crs)
            survey_export_xy_unit = (
                "deg" if survey_export_crs.is_geographic() else "м"
            )
        single_well_name_to_color = {
            str(selected.name): str(
                name_to_color.get(str(selected.name), "#2563eb")
            )
        }

        def render_3d_override(container: object, figure: object) -> None:
            wt._render_plotly_or_three_3d(
                container=container,
                figure=figure,
                backend=wt.WT_3D_BACKEND_THREE_LOCAL,
                height=560,
                payload_overrides={
                    "component_key": f"ptc-single-well-{selected.name}",
                    "edit_wells": wt._build_edit_wells_payload(
                        [selected],
                        single_well_name_to_color,
                    ),
                },
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
            render_3d_override=render_3d_override,
        )
        render_result_tables(
            view=well_view,
            t1_horizontal_offset_m=t1_horizontal_offset_m,
            summary_tab_label="Сводка",
            survey_tab_label="Инклинометрия",
            survey_file_name=f"{selected_name}_survey.csv",
            survey_export_stations=survey_export_stations,
            survey_export_xy_label_suffix=survey_export_xy_label_suffix,
            survey_export_xy_unit=survey_export_xy_unit,
            show_validation_section=False,
            show_solver_diagnostics_section=False,
        )
        return

    st.session_state["wt_results_all_view_mode"] = "Anti-collision"
    pads, _, _ = wt._pad_membership(records)
    if len(pads) > 1:
        focus_options = [
            wt.WT_PAD_FOCUS_ALL,
            *(str(pad.pad_id) for pad in pads),
        ]
        normalized_focus_pad_id = wt._normalize_focus_pad_id(
            records=records,
            requested_pad_id=st.session_state.get("wt_results_focus_pad_id"),
        )
        if normalized_focus_pad_id != str(
            st.session_state.get("wt_results_focus_pad_id", "")
        ):
            st.session_state["wt_results_focus_pad_id"] = (
                normalized_focus_pad_id
            )
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

    _render_ptc_anticollision_panel(
        successes=successes,
        records=records,
        focus_pad_id=focus_pad_id,
        focus_pad_well_names=focus_pad_well_names,
    )


def run_page() -> None:
    st.set_page_config(page_title="PTC", layout="wide")
    wt._init_state()
    _force_ptc_defaults()

    # Coordinate system selection in sidebar
    render_crs_sidebar()
    apply_page_style(max_width_px=1700)
    render_hero(
        title="PTC",
        subtitle="Prototype trajectory constructor",
        centered=True,
        max_content_width_px=760,
    )

    _render_ptc_import_section()
    _edit_applied = st.session_state.pop("wt_edit_targets_applied", None)
    if _edit_applied:
        st.success(
            f"Точки обновлены из 3D-редактора: {', '.join(_edit_applied)}. "
            "Проверьте подсветку t1/t3 ниже и запустите пересчёт."
        )
        st.toast(
            f"Цели обновлены из 3D-редактора: {', '.join(_edit_applied)}. "
            "Запустите пересчёт для уточнения траекторий.",
            icon=":material/edit:",
        )
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

    _render_ptc_reference_section()
    _render_ptc_run_section(records=records)

    if st.session_state.get("wt_last_error"):
        render_solver_diagnostics(st.session_state["wt_last_error"])

    st.markdown("## 5. Результаты расчёта")
    render_run_log_panel(
        st.session_state.get("wt_last_run_log_lines"),
        border=False,
    )

    summary_rows = st.session_state.get("wt_summary_rows")
    successes = st.session_state.get("wt_successes")
    if not summary_rows:
        render_small_note(
            "Результаты расчёта появятся после запуска расчёта траекторий."
        )
        return
    selected_crs = get_selected_crs()
    auto_convert = should_auto_convert()
    wt._render_batch_summary(
        summary_rows=summary_rows,
        target_crs=selected_crs,
        auto_convert=auto_convert,
    )
    if not successes:
        st.warning("Все выбранные скважины завершились ошибками расчёта.")
        return
    _render_ptc_success_tabs(
        successes=successes,
        records=list(records),
        summary_rows=list(summary_rows),
    )


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(
            "Запустите приложение командой `streamlit run pages/01_trajectory_constructor.py`."
        )
    run_page()
