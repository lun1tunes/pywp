from __future__ import annotations

import pandas as pd
import streamlit as st

from pywp import ptc_core as wt
from pywp import ptc_anticollision_params
from pywp import ptc_reference_state as reference_state
from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionWellSegment,
    anti_collision_report_rows,
)
from pywp.coordinate_integration import (
    csv_export_crs,
    get_crs_display_suffix,
    get_input_crs,
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


def _render_anticollision_action_button(*, has_current_analysis: bool) -> bool:
    return bool(
        st.button(
            (
                "Пересчёт пересечений"
                if has_current_analysis
                else "Расчёт пересечений"
            ),
            type="primary",
            use_container_width=True,
            help=(
                "Запускает anti-collision анализ для текущего набора рассчитанных "
                "траекторий. При наличии предыдущего кэша будут переиспользованы "
                "неизменённые скважины и пары."
            ),
        )
    )


def _prepare_anticollision_incremental_rerun() -> None:
    cache = st.session_state.get("wt_anticollision_analysis_cache")
    if not isinstance(cache, dict) or not cache:
        return
    refreshed_cache = dict(cache)
    refreshed_cache["key"] = ""
    st.session_state["wt_anticollision_analysis_cache"] = refreshed_cache


def _render_anticollision_panel(
    *,
    successes: list[object],
    records: list[object],
    summary_rows: list[dict[str, object]],
    focus_pad_id: str,
    focus_pad_well_names: list[str],
    show_visualization: bool = True,
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
        if _render_cached_anticollision_snapshot_for_pending_edits(
            successes=successes,
            records=records,
            summary_rows=summary_rows,
            focus_pad_well_names=focus_pad_well_names,
            show_visualization=show_visualization,
        ):
            return
        return

    reference_wells = reference_state.reference_wells_from_state()
    if len(successes) + len(reference_wells) < 2:
        st.info("Для anti-collision нужно минимум две успешно рассчитанные скважины.")
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
    current_snapshot = wt._current_anti_collision_cache_snapshot(
        successes=list(successes),
        uncertainty_model=uncertainty_model,
        records=list(records),
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
    )
    has_current_analysis = current_snapshot is not None
    has_stale_snapshot = (
        current_snapshot is None and _cached_anticollision_snapshot() is not None
    )

    st.markdown("### Anti-collision и пересечения")
    run_requested = _render_anticollision_action_button(
        has_current_analysis=has_current_analysis
    )
    if (
        has_current_analysis or has_stale_snapshot
    ) and _render_full_anticollision_recalc_button():
        return

    if not has_current_analysis and not run_requested:
        if has_stale_snapshot:
            st.info(
                "Траектории обновились после прошлого anti-collision расчёта. "
                "Нажмите 'Расчёт пересечений', чтобы пересчитать конуса и "
                "пересечения для актуального набора."
            )
        else:
            st.info(
                "Траектории рассчитаны. Запустите anti-collision отдельным "
                "шагом, когда будете готовы проверить конуса и пересечения."
            )
        wt._render_status_run_log(
            title="Лог расчёта Anti-collision",
            state_payload=st.session_state.get("wt_anticollision_last_run"),
            empty_message="Anti-collision анализ ещё не запускался.",
        )
        return

    if run_requested and has_current_analysis:
        _prepare_anticollision_incremental_rerun()

    anti_collision_parallel_workers = int(
        st.session_state.get("wt_last_parallel_workers") or 0
    )

    anti_collision_progress = (
        None
        if has_current_analysis and not run_requested
        else st.progress(8, text="Подготовка anti-collision анализа...")
    )

    def _anti_collision_progress_update(value: int, text: str) -> None:
        if anti_collision_progress is not None:
            anti_collision_progress.progress(int(value), text=text)

    if current_snapshot is not None and not run_requested:
        analysis, recommendations, clusters = current_snapshot
    else:
        try:
            analysis, recommendations, clusters = (
                wt._cached_anti_collision_view_model(
                    successes=successes,
                    uncertainty_model=uncertainty_model,
                    records=records,
                    reference_wells=reference_wells,
                    reference_uncertainty_models_by_name=(
                        reference_uncertainty_models_by_name
                    ),
                    progress_callback=_anti_collision_progress_update,
                    parallel_workers=anti_collision_parallel_workers,
                )
            )
        except Exception as exc:
            if anti_collision_progress is not None:
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
    if anti_collision_progress is not None:
        anti_collision_progress.empty()
    if run_requested and not show_visualization:
        st.rerun()
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
    if show_visualization:
        try:
            _render_anticollision_visual_overview(
                analysis=analysis,
                successes=successes,
                records=records,
                summary_rows=summary_rows,
                focus_pad_well_names=focus_pad_well_names,
                focus_anticollision_well_names=focus_anticollision_well_names,
                title=None,
                caption=None,
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
            ac_cache = st.session_state.get("wt_anticollision_analysis_cache")
            ac_cache = ac_cache if isinstance(ac_cache, dict) else {}

            new_records, new_success_dict, improved = optimize_pad_order(
                records=records,
                success_dict=success_dict,
                pad_well_names=focus_pad_well_names,
                uncertainty_model=uncertainty_model,
                reference_wells=list(reference_wells),
                config_by_name=config_by_name,
                progress_callback=opt_callback,
                fixed_well_names=set(fixed_pad_well_names),
                initial_analysis=analysis,
                previous_well_cache=ac_cache.get("well_cache"),
                previous_pair_cache=ac_cache.get("pair_cache"),
                well_signature_by_name=ac_cache.get("well_signature_by_name"),
                reference_uncertainty_models_by_name=(
                    reference_uncertainty_models_by_name
                ),
                parallel_workers=int(
                    st.session_state.get("wt_last_parallel_workers") or 0
                ),
            )

            if improved:
                _apply_pad_order_optimization_result(
                    new_records=new_records,
                    new_success_dict=new_success_dict,
                )
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


def _apply_pad_order_optimization_result(
    *,
    new_records: list[object],
    new_success_dict: dict[str, object],
) -> None:
    from pywp.welltrack_batch import (
        WelltrackBatchPlanner,
        merge_batch_results,
    )

    new_rows = []
    for record in new_records:
        name = str(getattr(record, "name", ""))
        if name in new_success_dict:
            new_rows.append(
                WelltrackBatchPlanner._row_from_success(
                    record=record, success=new_success_dict[name]
                )
            )

    merged_rows, merged_successes = merge_batch_results(
        records=new_records,
        existing_rows=st.session_state.get("wt_summary_rows"),
        existing_successes=st.session_state.get("wt_successes"),
        new_rows=new_rows,
        new_successes=list(new_success_dict.values()),
    )

    previous_ac_cache = st.session_state.get("wt_anticollision_analysis_cache")
    st.session_state["wt_records"] = list(new_records)
    st.session_state["wt_records_original"] = list(new_records)
    st.session_state["wt_successes"] = merged_successes
    st.session_state["wt_summary_rows"] = merged_rows
    wt._reset_anticollision_view_state(clear_prepared=True)
    if isinstance(previous_ac_cache, dict) and previous_ac_cache:
        st.session_state["wt_anticollision_analysis_cache"] = previous_ac_cache


def _cached_anticollision_snapshot() -> tuple[
    AntiCollisionAnalysis,
    tuple[object, ...],
    tuple[object, ...],
] | None:
    cache = st.session_state.get("wt_anticollision_analysis_cache")
    if not isinstance(cache, dict):
        return None
    analysis = cache.get("analysis")
    if not isinstance(analysis, AntiCollisionAnalysis):
        return None
    recommendations = cache.get("recommendations")
    clusters = cache.get("clusters")
    return (
        analysis,
        recommendations if isinstance(recommendations, tuple) else (),
        clusters if isinstance(clusters, tuple) else (),
    )


def _filter_cached_anticollision_snapshot_for_pending_edits(
    *,
    analysis: AntiCollisionAnalysis,
    recommendations: tuple[object, ...],
    clusters: tuple[object, ...],
    pending_edit_names: list[str],
) -> tuple[AntiCollisionAnalysis, tuple[object, ...], tuple[object, ...]]:
    edited_names = {
        str(name).strip() for name in pending_edit_names if str(name).strip()
    }
    if not edited_names:
        return analysis, recommendations, clusters
    filtered_wells = tuple(
        well
        for well in analysis.wells
        if str(getattr(well, "name", "")).strip() not in edited_names
    )
    filtered_corridors = tuple(
        corridor
        for corridor in analysis.corridors
        if not {
            str(getattr(corridor, "well_a", "")).strip(),
            str(getattr(corridor, "well_b", "")).strip(),
        }.intersection(edited_names)
    )
    filtered_zones = tuple(
        zone
        for zone in analysis.zones
        if not {
            str(getattr(zone, "well_a", "")).strip(),
            str(getattr(zone, "well_b", "")).strip(),
        }.intersection(edited_names)
    )
    filtered_segments = _cached_snapshot_segments_from_corridors(
        corridors=filtered_corridors,
        wells=filtered_wells,
    )
    pair_keys = {
        tuple(
            sorted(
                (
                    str(getattr(corridor, "well_a", "")).strip(),
                    str(getattr(corridor, "well_b", "")).strip(),
                )
            )
        )
        for corridor in filtered_corridors
    }
    target_pair_keys = {
        tuple(
            sorted(
                (
                    str(getattr(corridor, "well_a", "")).strip(),
                    str(getattr(corridor, "well_b", "")).strip(),
                )
            )
        )
        for corridor in filtered_corridors
        if int(getattr(corridor, "priority_rank", 99)) < 2
    }
    worst_sf_values = [
        float(getattr(zone, "separation_factor"))
        for zone in filtered_zones
        if getattr(zone, "separation_factor", None) is not None
    ]
    filtered_analysis = AntiCollisionAnalysis(
        wells=filtered_wells,
        corridors=filtered_corridors,
        well_segments=filtered_segments,
        zones=filtered_zones,
        pair_count=_cached_snapshot_pair_count(filtered_wells),
        overlapping_pair_count=int(len(pair_keys)),
        target_overlap_pair_count=int(len(target_pair_keys)),
        worst_separation_factor=(
            min(worst_sf_values) if worst_sf_values else None
        ),
    )
    filtered_recommendations = tuple(
        recommendation
        for recommendation in recommendations
        if not _recommendation_touches_names(recommendation, edited_names)
    )
    filtered_clusters = tuple(
        cluster
        for cluster in clusters
        if not _cluster_touches_names(cluster, edited_names)
    )
    return filtered_analysis, filtered_recommendations, filtered_clusters


def _cached_snapshot_pair_count(wells: tuple[object, ...]) -> int:
    return max(len(wells) * (len(wells) - 1) // 2, 0)


def _cached_snapshot_segments_from_corridors(
    *,
    corridors: tuple[object, ...],
    wells: tuple[object, ...],
) -> tuple[AntiCollisionWellSegment, ...]:
    if not corridors:
        return ()
    step_tolerance_m = _cached_snapshot_segment_step_tolerance_m(wells)
    raw_segments: list[AntiCollisionWellSegment] = []
    for corridor in corridors:
        raw_segments.append(
            AntiCollisionWellSegment(
                well_name=str(getattr(corridor, "well_a")),
                md_start_m=float(getattr(corridor, "md_a_start_m")),
                md_end_m=float(getattr(corridor, "md_a_end_m")),
                classification=str(getattr(corridor, "classification", "")),
                priority_rank=int(getattr(corridor, "priority_rank", 99)),
            )
        )
        raw_segments.append(
            AntiCollisionWellSegment(
                well_name=str(getattr(corridor, "well_b")),
                md_start_m=float(getattr(corridor, "md_b_start_m")),
                md_end_m=float(getattr(corridor, "md_b_end_m")),
                classification=str(getattr(corridor, "classification", "")),
                priority_rank=int(getattr(corridor, "priority_rank", 99)),
            )
        )
    merged: list[AntiCollisionWellSegment] = []
    for well_name in sorted({segment.well_name for segment in raw_segments}):
        well_segments = sorted(
            [segment for segment in raw_segments if segment.well_name == well_name],
            key=lambda segment: (float(segment.md_start_m), float(segment.md_end_m)),
        )
        current = well_segments[0]
        for segment in well_segments[1:]:
            if float(segment.md_start_m) <= float(current.md_end_m) + step_tolerance_m:
                current = AntiCollisionWellSegment(
                    well_name=current.well_name,
                    md_start_m=float(current.md_start_m),
                    md_end_m=max(float(current.md_end_m), float(segment.md_end_m)),
                    classification=(
                        current.classification
                        if int(current.priority_rank) <= int(segment.priority_rank)
                        else segment.classification
                    ),
                    priority_rank=min(
                        int(current.priority_rank), int(segment.priority_rank)
                    ),
                )
                continue
            merged.append(current)
            current = segment
        merged.append(current)
    return tuple(merged)


def _cached_snapshot_segment_step_tolerance_m(wells: tuple[object, ...]) -> float:
    step_values: list[float] = []
    for well in wells:
        samples = tuple(getattr(well, "samples", ()) or ())
        md_values = [
            float(getattr(sample, "md_m"))
            for sample in samples
            if getattr(sample, "md_m", None) is not None
        ]
        if len(md_values) < 2:
            continue
        md_values = sorted(set(md_values))
        diffs = [
            right - left
            for left, right in zip(md_values, md_values[1:], strict=False)
            if right > left
        ]
        if diffs:
            step_values.append(min(diffs))
    return float(max(step_values, default=100.0) * 1.05)


def _recommendation_touches_names(
    recommendation: object,
    edited_names: set[str],
) -> bool:
    names = {
        str(getattr(recommendation, "well_a", "")).strip(),
        str(getattr(recommendation, "well_b", "")).strip(),
        *(
            str(name).strip()
            for name in tuple(getattr(recommendation, "affected_wells", ()) or ())
        ),
    }
    return bool(names.intersection(edited_names))


def _cluster_touches_names(cluster: object, edited_names: set[str]) -> bool:
    names = {
        str(name).strip()
        for name in tuple(getattr(cluster, "well_names", ()) or ())
    }
    names.update(
        str(name).strip()
        for name in tuple(getattr(cluster, "affected_wells", ()) or ())
    )
    return bool(names.intersection(edited_names))


def _render_cached_anticollision_snapshot_for_pending_edits(
    *,
    successes: list[object],
    records: list[object],
    summary_rows: list[dict[str, object]],
    focus_pad_well_names: list[str],
    show_visualization: bool = True,
    show_report: bool = True,
) -> bool:
    snapshot = _cached_anticollision_snapshot()
    if snapshot is None:
        return False
    analysis, recommendations, clusters = snapshot
    analysis, recommendations, clusters = (
        _filter_cached_anticollision_snapshot_for_pending_edits(
            analysis=analysis,
            recommendations=recommendations,
            clusters=clusters,
            pending_edit_names=wt._pending_edit_target_names(),
        )
    )
    visible_focus_names = tuple(focus_pad_well_names)
    visible_clusters = wt._clusters_touching_focus_pad(
        clusters=clusters,
        focus_pad_well_names=visible_focus_names,
    )
    focus_anticollision_well_names = wt._anticollision_focus_well_names(
        clusters=visible_clusters,
        focus_pad_well_names=visible_focus_names,
    )
    target_only_wells = wt._failed_target_only_wells(
        records=list(records),
        summary_rows=list(summary_rows),
    )
    if show_visualization:
        _render_anticollision_visual_overview(
            analysis=analysis,
            successes=successes,
            records=records,
            summary_rows=summary_rows,
            focus_pad_well_names=visible_focus_names,
            focus_anticollision_well_names=focus_anticollision_well_names,
            title="### Все скважины, конуса и пересечения",
            caption=(
                "Ниже показан последний anti-collision снимок до пересчёта: "
                "старые конусы и пересечения скрыты для изменённых скважин, а "
                "фактический/утверждённый фонд и неизменённые скважины остаются "
                "на экране как ориентир."
            ),
            target_only_wells=target_only_wells,
        )

    visible_recommendations = wt._recommendations_for_clusters(
        recommendations=recommendations,
        clusters=visible_clusters,
    )
    if show_report and visible_recommendations:
        st.markdown("### Отчет по предыдущему anti-collision")
        report_rows = wt._report_rows_from_recommendations(
            visible_recommendations,
            analysis,
        )
        st.dataframe(
            wt.arrow_safe_text_dataframe(pd.DataFrame(report_rows)),
            width="stretch",
            hide_index=True,
        )
    return True


def _render_anticollision_visual_overview(
    *,
    analysis: AntiCollisionAnalysis,
    successes: list[object],
    records: list[object],
    summary_rows: list[dict[str, object]],
    focus_pad_well_names: list[str] | tuple[str, ...],
    focus_anticollision_well_names: list[str] | tuple[str, ...],
    title: str | None,
    caption: str | None,
    target_only_wells: list[object] | None = None,
) -> None:
    visible_focus_names = tuple(str(name) for name in focus_pad_well_names)
    focus_names = tuple(str(name) for name in focus_anticollision_well_names)
    resolved_target_only_wells = (
        list(target_only_wells)
        if target_only_wells is not None
        else wt._failed_target_only_wells(
            records=list(records),
            summary_rows=list(summary_rows),
        )
    )
    name_to_color = wt._well_color_map(list(records))
    display_name_by_well_name = wt._well_label_display_names(list(records))
    reference_wells = reference_state.reference_wells_from_state()
    if title:
        st.markdown(title)
    if caption:
        st.caption(caption)

    anticollision_3d_payload = wt._all_wells_anticollision_three_payload(
        analysis,
        previous_successes_by_name={},
        target_only_wells=resolved_target_only_wells,
        reference_wells=reference_wells,
        name_to_color=name_to_color,
        display_name_by_well_name=display_name_by_well_name,
        pilot_study_points_by_name=_pilot_study_points_by_name(list(records)),
        focus_well_names=focus_names or visible_focus_names,
        render_mode=wt.WT_3D_RENDER_DETAIL,
        show_sidetrack_relative_cones=False,
    )
    wt._render_three_payload(
        container=st.container(),
        payload=anticollision_3d_payload,
        height=660,
        payload_overrides=wt._anticollision_three_payload_overrides(
            records=list(records),
            analysis=analysis,
            successes=list(successes),
            target_only_wells=resolved_target_only_wells,
            target_only_name_to_color=name_to_color,
        ),
    )


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
    display_name_by_well_name = wt._well_label_display_names(list(records))
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
    reference_wells = reference_state.reference_wells_from_state()
    payload = wt._all_wells_three_payload(
        list(successes),
        target_only_wells=target_only_wells,
        reference_wells=reference_wells,
        name_to_color=name_to_color,
        display_name_by_well_name=display_name_by_well_name,
        pilot_study_points_by_name=_pilot_study_points_by_name(list(records)),
        focus_well_names=tuple(focus_pad_well_names),
        render_mode=wt.WT_3D_RENDER_FAST,
    )
    wt._render_three_payload(
        container=st.container(),
        payload=payload,
        height=660,
        payload_overrides=wt._trajectory_three_payload_overrides(
            records=list(records),
            successes=list(successes),
            target_only_wells=target_only_wells,
            name_to_color=name_to_color,
        ),
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
    input_crs = get_input_crs()
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
            input_crs,
            auto_convert=auto_convert,
        )
        if (
            auto_convert
            and selected_crs != input_crs
            and survey_export_crs == selected_crs
        ):
            survey_export_stations = transform_stations_to_crs(
                selected.stations,
                selected_crs,
                input_crs,
                rename_columns=False,
            )
        if auto_convert and selected_crs != input_crs:
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
                        parent_successes=list(successes),
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
            show_plotly_panels=False,
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
    reference_wells = reference_state.reference_wells_from_state()
    reference_uncertainty_models_by_name = (
        ptc_anticollision_params.reference_uncertainty_models_from_state(
            reference_wells
        )
    )
    selected_preset = wt.normalize_uncertainty_preset(
        st.session_state.get(
            "wt_anticollision_uncertainty_preset",
            wt.DEFAULT_UNCERTAINTY_PRESET,
        )
    )
    current_anticollision_snapshot = wt._current_anti_collision_cache_snapshot(
        successes=list(successes),
        uncertainty_model=planning_uncertainty_model_for_preset(selected_preset),
        records=list(records),
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
    )
    target_only_wells = wt._failed_target_only_wells(
        records=list(records),
        summary_rows=list(summary_rows),
    )
    has_pending_target_edits = bool(wt._pending_edit_target_names())
    rendered_cached_snapshot = False
    if has_pending_target_edits:
        try:
            rendered_cached_snapshot = _render_cached_anticollision_snapshot_for_pending_edits(
                successes=list(successes),
                records=list(records),
                summary_rows=list(summary_rows),
                focus_pad_well_names=list(focus_pad_well_names),
                show_visualization=True,
                show_report=False,
            )
        except Exception as exc:
            wt._store_anticollision_failure_state(exc)
            st.error(
                "Не удалось отрисовать сохранённую anti-collision визуализацию. "
                "Отчёт и данные кэша ниже остаются доступными."
            )
            st.caption(f"{type(exc).__name__}: {exc}")
            rendered_cached_snapshot = False
    if rendered_cached_snapshot:
        pass
    elif current_anticollision_snapshot is not None:
        analysis, _recommendations, clusters = current_anticollision_snapshot
        visible_clusters = wt._clusters_touching_focus_pad(
            clusters=clusters,
            focus_pad_well_names=focus_pad_well_names,
        )
        focus_anticollision_well_names = wt._anticollision_focus_well_names(
            clusters=visible_clusters,
            focus_pad_well_names=focus_pad_well_names,
        )
        try:
            _render_anticollision_visual_overview(
                analysis=analysis,
                successes=list(successes),
                records=list(records),
                summary_rows=list(summary_rows),
                focus_pad_well_names=list(focus_pad_well_names),
                focus_anticollision_well_names=list(focus_anticollision_well_names),
                title="### Все скважины, конуса и пересечения",
                caption=None,
                target_only_wells=target_only_wells,
            )
        except Exception as exc:
            wt._store_anticollision_failure_state(exc)
            st.error(
                "Не удалось отрисовать anti-collision визуализацию. "
                "Расчётные данные и отчёт ниже остаются доступными."
            )
            st.caption(f"{type(exc).__name__}: {exc}")
    else:
        _render_target_edit_overview(
            successes=list(successes),
            records=records,
            summary_rows=summary_rows,
            title="### Все скважины и точки целей",
            empty_message="Нет данных для 3D-обзора траекторий и целей.",
            focus_pad_well_names=list(focus_pad_well_names),
            show_focus_selector=False,
        )

    _render_anticollision_panel(
        successes=successes,
        records=records,
        summary_rows=summary_rows,
        focus_pad_id=focus_pad_id,
        focus_pad_well_names=focus_pad_well_names,
        show_visualization=False,
    )
