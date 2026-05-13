from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

import streamlit as st
import pandas as pd

from pywp.eclipse_welltrack import WelltrackRecord
from pywp.models import OPTIMIZATION_ANTI_COLLISION_AVOIDANCE, TrajectoryConfig
from pywp.pilot_wells import (
    SidetrackWindowOverride,
    is_pilot_name,
    parent_name_for_pilot,
    pilot_name_key_for_parent,
    sync_pilot_surfaces_to_parents,
    visible_well_names,
    well_name_key,
)
from pywp.planner import TrajectoryPlanner
from pywp.solver_diagnostics import summarize_problem_ru
from pywp.ui_calc_params import kop_min_vertical_function_from_state
from pywp.ui_utils import format_run_log_line
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    normalize_uncertainty_preset,
    planning_uncertainty_model_for_preset,
)
from pywp.well_pad import apply_pad_layout
from pywp.welltrack_batch import (
    DynamicClusterExecutionContext,
    SuccessfulWellPlan,
    WelltrackBatchPlanner,
    merge_batch_results,
    recommended_batch_selection,
)

__all__ = [
    "BatchRunHooks",
    "BatchRunRequest",
    "BatchSelectionStatus",
    "LOG_COMPACT",
    "LOG_LEVEL_OPTIONS",
    "LOG_VERBOSE",
    "batch_selection_status",
    "run_batch_if_clicked",
    "store_merged_batch_results",
    "sync_selection_state",
]

LOG_COMPACT = "Краткий"
LOG_VERBOSE = "Подробный"
LOG_LEVEL_OPTIONS: tuple[str, ...] = (LOG_COMPACT, LOG_VERBOSE)


@dataclass(frozen=True)
class BatchRunRequest:
    selected_names: list[str]
    config: TrajectoryConfig
    run_clicked: bool
    parallel_workers: int = 0
    sidetrack_window_overrides_by_name: Mapping[
        str, SidetrackWindowOverride
    ] | None = None


@dataclass(frozen=True)
class BatchSelectionStatus:
    has_summary_rows: bool
    ok_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    not_run_count: int = 0


@dataclass(frozen=True)
class BatchRunHooks:
    selected_execution_order: Callable[[list[str]], list[str]]
    pending_edit_target_names: Callable[[], list[str]]
    ensure_pad_configs: Callable[..., list[Any]]
    build_pad_plan_map: Callable[[list[Any]], dict[str, Any]]
    build_selected_override_configs: Callable[..., dict[str, TrajectoryConfig]]
    build_selected_optimization_contexts: Callable[..., dict[str, Any]]
    reference_wells_from_state: Callable[[], tuple[Any, ...]]
    reference_uncertainty_models_from_state: Callable[[tuple[Any, ...]], Mapping[str, Any]]
    resolution_snapshot_well_names: Callable[[dict[str, object]], tuple[str, ...]]
    format_prepared_override_scope: Callable[..., list[dict[str, object]]]
    prepared_plan_kind_label: Callable[[Mapping[str, object] | None], str]
    build_last_anticollision_resolution: Callable[..., dict[str, object] | None]
    focus_all_wells_anticollision_results: Callable[[], None]
    focus_all_wells_trajectory_results: Callable[[], None]


def sync_selection_state(
    state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
) -> tuple[list[str], list[str]]:
    all_names = visible_well_names(records)
    recommended_names = _visible_recommended_names(
        records=records,
        summary_rows=state.get("wt_summary_rows"),
    )
    pending_general = state.pop("wt_pending_selected_names", None)
    if pending_general is not None:
        state["wt_selected_names"] = _coerce_visible_selection(
            pending_general,
            all_names=all_names,
        )

    current = _coerce_visible_selection(
        state.get("wt_selected_names", []),
        all_names=all_names,
    )
    if current != state.get("wt_selected_names", []):
        state["wt_selected_names"] = list(current)
    if not current and recommended_names:
        state["wt_selected_names"] = list(recommended_names)
    return all_names, recommended_names


def batch_selection_status(
    *,
    records: list[WelltrackRecord],
    summary_rows: list[dict[str, object]] | None,
) -> BatchSelectionStatus:
    all_names = visible_well_names(records)
    rows_by_key = {
        well_name_key(row.get("Скважина", "")): row for row in (summary_rows or [])
    }
    ok_count = 0
    warning_count = 0
    error_count = 0
    not_run_count = 0
    for name in all_names:
        state = _combined_visible_row_state(name=name, rows_by_key=rows_by_key)
        if state == "ok":
            ok_count += 1
        elif state == "warning":
            warning_count += 1
        elif state == "not_run":
            not_run_count += 1
        else:
            error_count += 1
    return BatchSelectionStatus(
        has_summary_rows=bool(summary_rows),
        ok_count=ok_count,
        warning_count=warning_count,
        error_count=error_count,
        not_run_count=not_run_count,
    )


def _coerce_visible_selection(
    names: object,
    *,
    all_names: list[str],
) -> list[str]:
    visible_by_key = {well_name_key(name): str(name) for name in all_names}
    coerced: list[str] = []
    seen: set[str] = set()
    for raw_name in names if isinstance(names, (list, tuple, set)) else []:
        name = str(raw_name)
        name_key = well_name_key(name)
        if name_key not in visible_by_key and is_pilot_name(name):
            name_key = well_name_key(parent_name_for_pilot(name))
        visible_name = visible_by_key.get(name_key)
        if visible_name is None or visible_name in seen:
            continue
        coerced.append(visible_name)
        seen.add(visible_name)
    return coerced


def _visible_recommended_names(
    *,
    records: list[WelltrackRecord],
    summary_rows: object,
) -> list[str]:
    raw_recommended = recommended_batch_selection(
        records=records,
        summary_rows=summary_rows,
    )
    recommended_keys = {well_name_key(name) for name in raw_recommended}
    result: list[str] = []
    for name in visible_well_names(records):
        if (
            well_name_key(name) in recommended_keys
            or pilot_name_key_for_parent(name) in recommended_keys
        ):
            result.append(str(name))
    return result


def _combined_visible_row_state(
    *,
    name: str,
    rows_by_key: Mapping[str, dict[str, object]],
) -> str:
    rows = [
        row
        for row in (
            rows_by_key.get(well_name_key(name)),
            rows_by_key.get(pilot_name_key_for_parent(name)),
        )
        if row is not None
    ]
    if not rows:
        return "not_run"
    states = [_row_state(row) for row in rows]
    if "error" in states:
        return "error"
    if "warning" in states:
        return "warning"
    if "not_run" in states:
        return "not_run"
    return "ok"


def _row_state(row: Mapping[str, object]) -> str:
    status = str(row.get("Статус", "")).strip()
    if status == "OK":
        return "warning" if _has_problem_text(row.get("Проблема", "")) else "ok"
    if status == "Не рассчитана":
        return "not_run"
    return "error"


def _has_problem_text(value: object) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text and text not in {"—", "ОК", "OK", "nan", "NaN", "None", "<NA>"})


def store_merged_batch_results(
    state: MutableMapping[str, object],
    *,
    records: list[WelltrackRecord],
    new_rows: list[dict[str, object]],
    new_successes: list[SuccessfulWellPlan],
    pending_edit_target_names: Callable[[], list[str]],
) -> None:
    pending_before = set(pending_edit_target_names())
    merged_rows, merged_successes = merge_batch_results(
        records=records,
        existing_rows=state.get("wt_summary_rows"),
        existing_successes=state.get("wt_successes"),
        new_rows=new_rows,
        new_successes=new_successes,
    )
    state["wt_summary_rows"] = merged_rows
    state["wt_successes"] = merged_successes
    successful_names = {str(success.name) for success in new_successes}
    if successful_names:
        state["wt_edit_targets_highlight_names"] = [
            str(name)
            for name in (state.get("wt_edit_targets_highlight_names") or [])
            if str(name) not in successful_names
        ]
        state["wt_edit_targets_pending_names"] = [
            str(name)
            for name in pending_edit_target_names()
            if str(name) not in successful_names
        ]
        if pending_before and not pending_edit_target_names():
            state["wt_anticollision_analysis_cache"] = {}
    state["wt_pending_selected_names"] = list(
        recommended_batch_selection(records=records, summary_rows=merged_rows)
    )


def run_batch_if_clicked(
    *,
    requests: Sequence[BatchRunRequest],
    records: list[WelltrackRecord],
    hooks: BatchRunHooks,
    st_module: Any = st,
    calc_params_prefix: str = "wt_cfg_",
    log_compact_label: str = LOG_COMPACT,
    log_verbose_label: str = LOG_VERBOSE,
) -> None:
    request = next((item for item in requests if item.run_clicked), None)
    if request is None:
        return
    state = st_module.session_state
    selected_names = [str(name) for name in request.selected_names]
    selected_set = set(selected_names)
    if not selected_set:
        st_module.warning("Выберите минимум одну скважину для расчета.")
        return

    selected_execution_order = hooks.selected_execution_order(selected_names)
    pending_edit_names_before_run = set(hooks.pending_edit_target_names())
    records_for_run = list(records)
    pad_layout_active = bool(str(state.get("wt_pad_last_applied_at", "")))
    if pad_layout_active:
        base_records = state.get("wt_records_original")
        if base_records:
            pads = hooks.ensure_pad_configs(base_records=list(base_records))
            plan_map = hooks.build_pad_plan_map(pads)
            records_for_run = sync_pilot_surfaces_to_parents(
                apply_pad_layout(
                    records=list(base_records),
                    pads=pads,
                    plan_by_pad_id=plan_map,
                )
            )
            state["wt_records"] = list(records_for_run)

    batch = WelltrackBatchPlanner(planner=TrajectoryPlanner())
    log_verbosity = str(state.get("wt_log_verbosity", log_compact_label))
    verbose_log_enabled = log_verbosity == log_verbose_label
    records_by_name = {str(record.name): record for record in records_for_run}
    config_by_name = hooks.build_selected_override_configs(
        base_config=request.config,
        selected_names=selected_set,
        records_by_name=records_by_name,
    )
    optimization_context_by_name = hooks.build_selected_optimization_contexts(
        selected_names=selected_set,
        current_successes=list(state.get("wt_successes") or ()),
    )
    current_success_by_name = {
        str(item.name): item for item in (state.get("wt_successes") or ())
    }
    prepared_snapshot = dict(state.get("wt_prepared_recommendation_snapshot") or {})
    prepared_override_names = {
        str(name) for name in (state.get("wt_prepared_well_overrides") or {}).keys()
    }
    previous_anticollision_successes = {
        str(name): current_success_by_name[str(name)]
        for name in sorted(
            prepared_override_names.intersection(current_success_by_name)
        )
    }
    dynamic_cluster_context = None
    if str(prepared_snapshot.get("kind", "")).strip() == "cluster":
        reference_wells = hooks.reference_wells_from_state()
        target_well_names = tuple(
            str(name)
            for name in prepared_snapshot.get("target_well_names", ()) or ()
            if str(name).strip()
        ) or hooks.resolution_snapshot_well_names(prepared_snapshot)
        if target_well_names:
            dynamic_cluster_context = DynamicClusterExecutionContext(
                target_well_names=tuple(target_well_names),
                uncertainty_model=planning_uncertainty_model_for_preset(
                    normalize_uncertainty_preset(
                        state.get(
                            "wt_anticollision_uncertainty_preset",
                            DEFAULT_UNCERTAINTY_PRESET,
                        )
                    )
                ),
                initial_successes=tuple(state.get("wt_successes") or ()),
                reference_wells=reference_wells,
                reference_uncertainty_models_by_name=(
                    hooks.reference_uncertainty_models_from_state(reference_wells)
                ),
            )
    missing_anticollision_context = sorted(
        well_name
        for well_name, cfg in config_by_name.items()
        if str(cfg.optimization_mode) == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE
        and well_name not in optimization_context_by_name
    )
    if missing_anticollision_context and dynamic_cluster_context is None:
        state["wt_last_error"] = (
            "Не удалось запустить anti-collision пересчет: отсутствует контекст "
            "конфликтного окна для скважин "
            + ", ".join(missing_anticollision_context)
            + ". Подготовьте рекомендацию повторно."
        )
        st_module.error(str(state["wt_last_error"]))
        return

    run_started_s = perf_counter()
    log_lines: list[str] = []
    progress = st_module.progress(0, text="Подготовка batch-расчета...")
    phase_placeholder = st_module.empty()
    live_log_placeholder = st_module.empty()

    def append_log(message: str, *, verbose_only: bool = False) -> None:
        if verbose_only and not verbose_log_enabled:
            return
        log_lines.append(format_run_log_line(run_started_s, message))
        live_log_placeholder.code("\n".join(log_lines[-240:]), language="text")

    def set_phase(message: str) -> None:
        phase_placeholder.caption(message)

    prepared_scope_rows = hooks.format_prepared_override_scope(
        selected_names=selected_names,
    )

    try:
        with st_module.spinner(
            "Выполняется расчет WELLTRACK-набора...", show_time=True
        ):
            started = perf_counter()
            append_log(
                f"Старт batch-расчета. Выбрано скважин: {len(selected_set)}. "
                f"Детализация лога: {log_verbosity}."
            )
            if prepared_scope_rows:
                append_log(
                    "Активен prepared anti-collision plan ("
                    + hooks.prepared_plan_kind_label(prepared_snapshot)
                    + "). Локальные overrides будут применены к "
                    + ", ".join(
                        f"{row['Скважина']} ({row['Локальный режим']})"
                        for row in prepared_scope_rows
                    )
                    + "."
                )
            if optimization_context_by_name:
                append_log(
                    "Для части выбранных скважин активирован anti-collision avoidance "
                    "mode на конфликтном окне."
                )
            selected_name_keys = {well_name_key(item) for item in selected_set}
            active_sidetrack_overrides = {
                str(name): override
                for name, override in (
                    request.sidetrack_window_overrides_by_name or {}
                ).items()
                if well_name_key(name) in selected_name_keys
            }
            if active_sidetrack_overrides:
                append_log(
                    "Активны ручные окна зарезки: "
                    + ", ".join(
                        f"{name} ({override.kind.upper()}={override.value_m:.2f} м)"
                        for name, override in active_sidetrack_overrides.items()
                    )
                    + "."
                )
            active_kop_function = kop_min_vertical_function_from_state(
                prefix=calc_params_prefix
            )
            if active_kop_function is not None:
                append_log(
                    "Для выбранных скважин активна функция KOP / TVD: "
                    + str(active_kop_function.note).strip()
                )
            if dynamic_cluster_context is not None:
                append_log(
                    "Включена iterative cluster-aware execution policy: "
                    "порядок шагов и anti-collision overrides будут пересчитываться "
                    "после каждого успешного шага по текущей topology кластера."
                )
            elif (
                len(selected_execution_order) > 1
                and selected_execution_order != selected_names
            ):
                append_log(
                    "Cluster-aware execution order: "
                    + " -> ".join(selected_execution_order)
                    + ". Следующие скважины используют обновленные reference paths "
                    "уже пересчитанных шагов."
                )
            if int(request.parallel_workers) > 1 and dynamic_cluster_context is None:
                append_log(
                    f"Параллельный расчёт: {int(request.parallel_workers)} процессов."
                )
            elif (
                int(request.parallel_workers) > 1
                and dynamic_cluster_context is not None
            ):
                append_log(
                    "Параллельный расчёт отключён: активен iterative cluster-aware "
                    "режим (скважины зависят друг от друга)."
                )
            if pad_layout_active:
                append_log(
                    "Активна раскладка устьев по кустам: перед расчетом применены "
                    "текущие координаты S из блока 'Кусты и расчет устьев'."
                )
            set_phase(f"Старт расчета набора. Выбрано скважин: {len(selected_set)}.")
            progress_state: dict[str, int] = {"value": 0}
            last_stage_by_well: dict[str, str] = {}

            def update_progress(value: int, text: str) -> None:
                clamped = int(max(0, min(99, value)))
                clamped = max(int(progress_state["value"]), clamped)
                progress_state["value"] = clamped
                progress.progress(clamped, text=text)

            def on_progress(index: int, total: int, name: str) -> None:
                start_fraction = (float(index) - 1.0) / max(float(total), 1.0)
                update_progress(
                    int(round(start_fraction * 100.0)),
                    text=f"{index}/{total}: {name} · подготовка",
                )
                set_phase(f"Расчет скважины {index}/{total}: {name}")
                append_log(f"Расчет скважины {index}/{total}: {name}.")

            def on_solver_progress(
                index: int,
                total: int,
                name: str,
                stage_text: str,
                stage_fraction: float,
            ) -> None:
                local_fraction = float(max(0.0, min(1.0, stage_fraction)))
                overall = (float(index) - 1.0 + local_fraction) / max(float(total), 1.0)
                update_progress(
                    int(round(overall * 100.0)),
                    text=f"{index}/{total}: {name} · {stage_text}",
                )
                set_phase(f"Скважина {index}/{total} {name}: {stage_text}")
                stage_key = f"{index}:{name}"
                stage_norm = str(stage_text)
                if last_stage_by_well.get(stage_key) == stage_norm:
                    return
                last_stage_by_well[stage_key] = stage_norm
                append_log(f"{name}: {stage_norm}", verbose_only=True)

            def on_record_done(
                index: int,
                total: int,
                name: str,
                row: dict[str, object],
            ) -> None:
                end_fraction = float(index) / max(float(total), 1.0)
                update_progress(
                    int(round(end_fraction * 100.0)),
                    text=f"{index}/{total}: {name} · завершено",
                )
                status = str(row.get("Статус", "—"))
                raw_problem_text = str(row.get("Проблема", "")).strip()
                problem_text = (
                    summarize_problem_ru(raw_problem_text) if raw_problem_text else ""
                )
                try:
                    restart_count = int(float(row.get("Рестарты решателя", 0)))
                except (TypeError, ValueError):
                    restart_count = 0
                restart_suffix = (
                    f" Использовано рестартов решателя: {restart_count}."
                    if restart_count > 0
                    else ""
                )
                if status == "OK":
                    if problem_text and problem_text != "ОК":
                        append_log(
                            f"{name}: расчет завершен с предупреждением. "
                            f"{problem_text}{restart_suffix}"
                        )
                    else:
                        append_log(f"{name}: расчет завершен успешно.{restart_suffix}")
                    return
                if problem_text and problem_text != "ОК":
                    append_log(f"{name}: {status}. {problem_text}")
                else:
                    append_log(f"{name}: {status}.")

            summary_rows, successes = batch.evaluate(
                records=records_for_run,
                selected_names=selected_set,
                selected_order=selected_execution_order,
                config=request.config,
                config_by_name=config_by_name,
                optimization_context_by_name=optimization_context_by_name,
                sidetrack_window_overrides_by_name=(
                    request.sidetrack_window_overrides_by_name
                ),
                dynamic_cluster_context=dynamic_cluster_context,
                progress_callback=on_progress,
                solver_progress_callback=on_solver_progress,
                record_done_callback=on_record_done,
                parallel_workers=int(request.parallel_workers),
            )
            batch_metadata = batch.last_evaluation_metadata
            skipped_policy_count = int(len(batch_metadata.skipped_selected_names))
            if dynamic_cluster_context is not None:
                skipped_names = tuple(
                    str(name)
                    for name in batch_metadata.skipped_selected_names
                    if str(name).strip()
                )
                if skipped_names:
                    if bool(batch_metadata.cluster_blocked):
                        blocking_reason = (
                            str(batch_metadata.cluster_blocking_reason).strip()
                            if batch_metadata.cluster_blocking_reason
                            else "cluster-level пересчет перешел в advisory-only режим."
                        )
                        append_log(
                            "Iterative cluster-aware execution остановлен: "
                            + blocking_reason
                            + " Без дополнительного пересчета оставлены: "
                            + ", ".join(skipped_names)
                            + "."
                        )
                    elif bool(batch_metadata.cluster_resolved_early):
                        append_log(
                            "Iterative cluster-aware execution завершился досрочно: "
                            "после очередного шага дополнительные пересчеты для "
                            "оставшихся скважин не потребовались. Без повторного "
                            "пересчета оставлены: " + ", ".join(skipped_names) + "."
                        )

            elapsed_s = perf_counter() - started
            progress.progress(100, text="Batch-расчет завершен.")
            store_merged_batch_results(
                state,
                records=records_for_run,
                new_rows=summary_rows,
                new_successes=successes,
                pending_edit_target_names=hooks.pending_edit_target_names,
            )
            pending_edit_names_after_run = set(hooks.pending_edit_target_names())
            edit_target_recalculation_completed = bool(
                pending_edit_names_before_run
                and not pending_edit_names_after_run
                and pending_edit_names_before_run.issubset(selected_set)
            )
            applied_affected_wells = {
                str(name) for name in prepared_snapshot.get("affected_wells", ())
            }
            applied_prepared_plan = bool(
                prepared_snapshot
                and applied_affected_wells
                and applied_affected_wells.issubset(selected_set)
            )
            if applied_prepared_plan:
                preset = normalize_uncertainty_preset(
                    state.get(
                        "wt_anticollision_uncertainty_preset",
                        DEFAULT_UNCERTAINTY_PRESET,
                    )
                )
                resolution = hooks.build_last_anticollision_resolution(
                    snapshot=prepared_snapshot,
                    successes=list(state.get("wt_successes") or ()),
                    uncertainty_model=planning_uncertainty_model_for_preset(preset),
                    uncertainty_preset=preset,
                )
                state["wt_last_anticollision_resolution"] = resolution
                state["wt_last_anticollision_previous_successes"] = (
                    previous_anticollision_successes
                )
                hooks.focus_all_wells_anticollision_results()
            else:
                state["wt_last_anticollision_resolution"] = None
                state["wt_last_anticollision_previous_successes"] = {}
                if edit_target_recalculation_completed:
                    hooks.focus_all_wells_anticollision_results()
                else:
                    hooks.focus_all_wells_trajectory_results()
            state["wt_last_error"] = ""
            state["wt_last_run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state["wt_last_runtime_s"] = float(elapsed_s)
            state["wt_prepared_well_overrides"] = {}
            state["wt_prepared_override_message"] = ""
            state["wt_prepared_recommendation_id"] = ""
            state["wt_anticollision_prepared_cluster_id"] = ""
            state["wt_prepared_recommendation_snapshot"] = None
            append_log(
                f"Batch-расчет завершен. Успешно: {len(successes)}, "
                f"ошибок: {len(summary_rows) - len(successes)}"
                + (
                    f", без дополнительного пересчета оставлено: {skipped_policy_count}"
                    if skipped_policy_count > 0
                    else ""
                )
                + ". "
                f"Затраченное время: {elapsed_s:.2f} с.",
            )
            if successes:
                phase_placeholder.success(
                    f"Расчет завершен за {elapsed_s:.2f} с. "
                    f"Успешно: {len(successes)}"
                    + (
                        f", без дополнительного пересчета оставлено: {skipped_policy_count}"
                        if skipped_policy_count > 0
                        else ""
                    )
                )
            else:
                phase_placeholder.error(
                    f"Расчет завершен за {elapsed_s:.2f} с, " "но без успешных скважин."
                )
    except Exception as exc:  # noqa: BLE001
        state["wt_last_error"] = str(exc)
        append_log(f"Ошибка batch-расчета: {summarize_problem_ru(str(exc))}")
        phase_placeholder.error("Batch-расчет завершился ошибкой")
    finally:
        state["wt_last_run_log_lines"] = log_lines
        progress.empty()
        live_log_placeholder.empty()
