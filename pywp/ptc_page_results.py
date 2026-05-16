from __future__ import annotations

import pandas as pd
import streamlit as st

from pywp import ptc_core as wt
from pywp import ptc_anticollision_params
from pywp import ptc_reference_state as reference_state
from pywp.anticollision import anti_collision_report_rows
from pywp.coordinate_integration import (
    DEFAULT_CRS,
    csv_export_crs,
    get_crs_display_suffix,
    get_selected_crs,
    should_auto_convert,
    transform_stations_to_crs,
)
from pywp.models import Point3D
from pywp.pilot_wells import is_pilot_name, pilot_name_key_for_parent, well_name_key
from pywp.ui_well_result import (
    SingleWellResultView,
    render_key_metrics,
    render_result_plots,
    render_result_tables,
)
from pywp.uncertainty import UNCERTAINTY_PRESET_OPTIONS
from pywp.uncertainty import planning_uncertainty_model_for_preset

__all__ = ["render_failed_target_only_results", "render_success_tabs"]


def _single_well_pilot_context(
    *,
    parent_success: object,
    successes: list[object],
    records: list[object],
) -> tuple[str | None, pd.DataFrame | None, tuple[Point3D, ...]]:
    pilot_success = _pilot_success_for_parent(
        parent_success=parent_success,
        successes=successes,
    )
    pilot_record = _pilot_record_for_parent(
        parent_name=getattr(parent_success, "name", ""),
        pilot_name=getattr(pilot_success, "name", None),
        records=records,
    )
    pilot_name = str(
        getattr(pilot_success, "name", "")
        or getattr(pilot_record, "name", "")
        or ""
    ).strip()
    if not pilot_name:
        return None, None, ()
    study_points = (
        _pilot_study_points_from_record(pilot_record)
        if pilot_record is not None
        else _pilot_study_points_from_success(pilot_success)
    )
    return pilot_name, getattr(pilot_success, "stations", None), study_points


def _pilot_success_for_parent(
    *,
    parent_success: object,
    successes: list[object],
) -> object | None:
    parent_name = getattr(parent_success, "name", "")
    expected_names = {pilot_name_key_for_parent(parent_name)}
    summary = getattr(parent_success, "summary", {}) or {}
    pilot_name = str(
        summary.get("pilot_well_name", "") if hasattr(summary, "get") else ""
    ).strip()
    if pilot_name:
        expected_names.add(well_name_key(pilot_name))
    return next(
        (
            success
            for success in successes
            if is_pilot_name(getattr(success, "name", ""))
            and well_name_key(getattr(success, "name", "")) in expected_names
        ),
        None,
    )


def _pilot_record_for_parent(
    *,
    parent_name: object,
    pilot_name: object | None,
    records: list[object],
) -> object | None:
    expected_names = {pilot_name_key_for_parent(parent_name)}
    if pilot_name:
        expected_names.add(well_name_key(pilot_name))
    return next(
        (
            record
            for record in records
            if is_pilot_name(getattr(record, "name", ""))
            and well_name_key(getattr(record, "name", "")) in expected_names
        ),
        None,
    )


def _pilot_study_points_from_record(record: object) -> tuple[Point3D, ...]:
    points = tuple(getattr(record, "points", ()) or ())
    return tuple(
        Point3D(x=float(point.x), y=float(point.y), z=float(point.z))
        for point in points[1:]
    )


def _pilot_study_points_from_success(
    pilot_success: object | None,
) -> tuple[Point3D, ...]:
    if pilot_success is None:
        return ()
    result: list[Point3D] = []
    for attr_name in ("t1", "t3"):
        point = getattr(pilot_success, attr_name, None)
        if point is None:
            continue
        candidate = Point3D(x=float(point.x), y=float(point.y), z=float(point.z))
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def _pilot_study_points_by_name(records: list[object]) -> dict[str, tuple[Point3D, ...]]:
    return {
        str(getattr(record, "name", "")): _pilot_study_points_from_record(record)
        for record in records
        if is_pilot_name(getattr(record, "name", ""))
    }


def _render_anticollision_panel(
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

    reference_wells = reference_state.reference_wells_from_state()
    if len(successes) + len(reference_wells) < 2:
        st.info("Для anti-collision нужно минимум две успешно рассчитанные скважины.")
        return

    if _render_full_anticollision_recalc_button():
        return

    preset_options = list(UNCERTAINTY_PRESET_OPTIONS.keys())
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
    uncertainty_model = planning_uncertainty_model_for_preset(selected_preset)
    reference_uncertainty_models_by_name = (
        ptc_anticollision_params.reference_uncertainty_models_from_state(
            reference_wells
        )
    )
    anti_collision_parallel_workers = int(
        st.session_state.get("wt_last_parallel_workers") or 0
    )

    anti_collision_progress = st.progress(
        8, text="Подготовка anti-collision анализа..."
    )

    def _anti_collision_progress_update(value: int, text: str) -> None:
        anti_collision_progress.progress(int(value), text=text)

    try:
        analysis, recommendations, clusters = wt._cached_anti_collision_view_model(
            successes=successes,
            uncertainty_model=uncertainty_model,
            records=records,
            reference_wells=reference_wells,
            reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
            progress_callback=_anti_collision_progress_update,
            parallel_workers=anti_collision_parallel_workers,
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

    wt._render_status_run_log(
        title="Лог расчёта Anti-collision",
        state_payload=st.session_state.get("wt_anticollision_last_run"),
        empty_message="Anti-collision анализ ещё не запускался.",
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
    try:
        anticollision_3d_payload = wt._all_wells_anticollision_three_payload(
            analysis,
            previous_successes_by_name={},
            pilot_study_points_by_name=_pilot_study_points_by_name(records),
            focus_well_names=focus_anticollision_well_names or focus_pad_well_names,
            render_mode=wt.WT_3D_RENDER_DETAIL,
        )
        wt._render_three_payload(
            container=chart_col1,
            payload=anticollision_3d_payload,
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
                focus_well_names=focus_anticollision_well_names or focus_pad_well_names,
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

    target_zones = [zone for zone in analysis.zones if int(zone.priority_rank) < 2]
    if target_zones:
        st.warning(
            "Найдены пересечения, затрагивающие точки целей t1/t3. Они вынесены "
            "в начало отчета и должны разбираться в первую очередь."
        )
    else:
        st.warning("Найдены пересечения 2σ конусов неопределенности по траекториям.")

    report_rows = (
        wt._report_rows_from_recommendations(visible_recommendations, analysis)
        if visible_recommendations
        else anti_collision_report_rows(analysis)
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
                s.name: s.config for s in successes if s.name in focus_pad_well_names
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


def _render_full_anticollision_recalc_button() -> bool:
    if not st.button(
        "Полный пересчёт anti-collision",
        help=(
            "Сбрасывает кэш скважин и пар anti-collision. Используйте после "
            "смены набора данных или если нужен гарантированный расчёт с нуля."
        ),
        use_container_width=True,
    ):
        return False
    wt._reset_anticollision_view_state(clear_prepared=True)
    st.rerun()
    return True


def _render_target_edit_overview(
    *,
    successes: list[object],
    records: list[object],
    summary_rows: list[dict[str, object]],
    title: str,
    empty_message: str,
    focus_pad_well_names: list[str] | None = None,
    show_focus_selector: bool = True,
) -> bool:
    target_only_wells = wt._failed_target_only_wells(
        records=list(records),
        summary_rows=list(summary_rows),
    )
    if not successes and not target_only_wells:
        st.info(empty_message)
        return False

    name_to_color = wt._well_color_map(list(records))
    st.markdown(title)
    if target_only_wells:
        st.caption(
            "Скважины без успешной траектории показаны как исходные точки. "
            "Их можно выбрать в легенде 3D и подвинуть через редактор целей."
        )

    if focus_pad_well_names is None:
        pads, _, _ = wt._pad_membership(records)
        if show_focus_selector and len(pads) > 1:
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
    payload = wt._all_wells_three_payload(
        list(successes),
        target_only_wells=target_only_wells,
        name_to_color=name_to_color,
        pilot_study_points_by_name=_pilot_study_points_by_name(list(records)),
        focus_well_names=tuple(focus_pad_well_names),
        render_mode=wt.WT_3D_RENDER_DETAIL,
    )
    chart_col1, chart_col2 = st.columns(2, gap="medium")
    wt._render_three_payload(
        container=chart_col1,
        payload=payload,
        height=660,
        payload_overrides=wt._trajectory_three_payload_overrides(
            records=list(records),
            successes=list(successes),
            target_only_wells=target_only_wells,
            name_to_color=name_to_color,
        ),
    )
    chart_col2.plotly_chart(
        wt._all_wells_plan_figure(
            list(successes),
            target_only_wells=target_only_wells,
            name_to_color=name_to_color,
            focus_well_names=tuple(focus_pad_well_names),
        ),
        width="stretch",
    )
    return True


def render_failed_target_only_results(
    *,
    records: list[object],
    summary_rows: list[dict[str, object]],
) -> None:
    _render_target_edit_overview(
        successes=[],
        records=records,
        summary_rows=summary_rows,
        title="### Исходные точки для правки",
        empty_message=(
            "Не удалось подготовить исходные точки для 3D-редактора. "
            "Проверьте статус импорта целей выше."
        ),
    )


def render_success_tabs(
    successes: list[object],
    records: list[object],
    summary_rows: list[dict[str, object]],
) -> None:
    selected_crs = get_selected_crs()
    auto_convert = should_auto_convert()
    name_to_color = wt._well_color_map(list(records))
    view_mode = st.radio(
        "Режим просмотра результатов",
        options=["Отдельная скважина", "Все скважины"],
        key="wt_results_view_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    if str(view_mode) == "Отдельная скважина":
        visible_successes = [
            item for item in successes if not is_pilot_name(getattr(item, "name", ""))
        ]
        if not visible_successes:
            st.info("Нет рассчитанных продуктивных стволов для просмотра.")
            return
        selected_name = st.selectbox(
            "Скважина", options=[item.name for item in visible_successes]
        )
        selected = wt._find_selected_success(
            selected_name=str(selected_name),
            successes=successes,
        )
        pilot_name, pilot_stations, pilot_study_points = _single_well_pilot_context(
            parent_success=selected,
            successes=successes,
            records=records,
        )
        well_view = SingleWellResultView(
            well_name=str(selected.name),
            surface=selected.surface,
            t1=selected.t1,
            t3=selected.t3,
            target_pairs=tuple(getattr(selected, "target_pairs", ()) or ()),
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
            pilot_name=pilot_name,
            pilot_stations=pilot_stations,
            pilot_study_points=pilot_study_points,
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
            survey_export_xy_unit = "deg" if survey_export_crs.is_geographic() else "м"
        single_well_name_to_color = {
            str(selected.name): str(name_to_color.get(str(selected.name), "#2563eb"))
        }

        def render_3d_override(container: object, payload: dict[str, object]) -> None:
            wt._render_three_payload(
                container=container,
                payload=payload,
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
    target_only_wells = wt._failed_target_only_wells(
        records=list(records),
        summary_rows=list(summary_rows),
    )
    if target_only_wells:
        _render_target_edit_overview(
            successes=list(successes),
            records=records,
            summary_rows=summary_rows,
            title="### Все скважины и точки без траектории",
            empty_message="Нет данных для 3D-обзора целей.",
            focus_pad_well_names=list(focus_pad_well_names),
            show_focus_selector=False,
        )

    _render_anticollision_panel(
        successes=successes,
        records=records,
        focus_pad_id=focus_pad_id,
        focus_pad_well_names=focus_pad_well_names,
    )
