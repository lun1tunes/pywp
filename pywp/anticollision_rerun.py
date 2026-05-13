from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from pywp.anticollision import (
    AntiCollisionAnalysis,
    DEFINITIVE_SCAN_STEP_M,
    analyze_anti_collision,
    build_anti_collision_well,
)
from pywp.anticollision_optimization import (
    AntiCollisionOptimizationContext,
    build_anti_collision_reference_path,
)
from pywp.anticollision_recommendations import (
    AntiCollisionRecommendation,
    AntiCollisionRecommendationCluster,
    AntiCollisionWellContext,
    MANEUVER_BUILD2_ENTRY,
    MANEUVER_KOP_AND_TRAJECTORY,
    RECOMMENDATION_REDUCE_KOP,
    RECOMMENDATION_TRAJECTORY_REVIEW,
    build_anti_collision_recommendation_clusters,
    build_anti_collision_recommendations,
    recommendation_display_label,
)
from pywp.models import OPTIMIZATION_ANTI_COLLISION_AVOIDANCE, OPTIMIZATION_MINIMIZE_KOP
from pywp.pilot_wells import paired_pilot_parent_names
from pywp.reference_trajectories import (
    ImportedTrajectoryWell,
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_KIND_COLORS,
    reference_well_collision_name,
    reference_well_duplicate_name_keys,
)
from pywp.uncertainty import PlanningUncertaintyModel, fast_proxy_uncertainty_model

if TYPE_CHECKING:
    from pywp.welltrack_batch import SuccessfulWellPlan


DYNAMIC_CLUSTER_PLAN_ACTIVE = "active"
DYNAMIC_CLUSTER_PLAN_RESOLVED = "resolved"
DYNAMIC_CLUSTER_PLAN_BLOCKED = "blocked"
_CLUSTER_REFERENCE_WINDOW_MARGIN_M = 100.0
_ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M = 75.0
_EARLY_ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M = 100.0


@dataclass(frozen=True)
class DynamicClusterExecutionPlan:
    cluster: AntiCollisionRecommendationCluster | None
    ordered_well_names: tuple[str, ...]
    prepared_by_well: dict[str, dict[str, object]]
    skipped_wells: tuple[str, ...]
    resolution_state: str = DYNAMIC_CLUSTER_PLAN_ACTIVE
    blocking_reason: str | None = None


def _reference_wells_by_collision_name(
    reference_wells: tuple[ImportedTrajectoryWell, ...],
    *,
    planned_names: tuple[str, ...],
) -> dict[str, ImportedTrajectoryWell]:
    planned_name_keys = {
        str(name).strip().casefold()
        for name in planned_names
        if str(name).strip()
    }
    duplicate_name_keys = reference_well_duplicate_name_keys(reference_wells)
    result: dict[str, ImportedTrajectoryWell] = {}
    for item in reference_wells:
        collision_name = reference_well_collision_name(
            item,
            planned_names=planned_names,
            duplicate_name_keys=duplicate_name_keys,
        )
        result[str(collision_name)] = item
        item_key = str(item.name).strip().casefold()
        if (
            str(item.name) not in result
            and item_key not in planned_name_keys
            and item_key not in duplicate_name_keys
        ):
            result[str(item.name)] = item
    return result


def build_anti_collision_analysis_for_successes(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: Mapping[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
    include_display_geometry: bool = True,
    build_overlap_geometry: bool = True,
    analysis_sample_step_m: float | None = None,
) -> AntiCollisionAnalysis:
    effective_sample_step_m = (
        float(analysis_sample_step_m)
        if analysis_sample_step_m is not None
        else (
            float(DEFINITIVE_SCAN_STEP_M)
            if include_display_geometry and build_overlap_geometry
            else None
        )
    )
    planned_names = tuple(str(item.name) for item in successes)
    duplicate_reference_name_keys = reference_well_duplicate_name_keys(
        reference_wells
    )
    wells = [
        build_anti_collision_well(
            name=item.name,
            color=(name_to_color or {}).get(str(item.name), "#A0A0A0"),
            stations=item.stations,
            surface=item.surface,
            t1=item.t1,
            t3=item.t3,
            azimuth_deg=float(item.azimuth_deg),
            md_t1_m=float(item.md_t1_m),
            model=model,
            include_display_geometry=include_display_geometry,
            well_kind="project",
            is_reference_only=False,
            analysis_sample_step_m=effective_sample_step_m,
        )
        for item in successes
    ]
    for item in reference_wells:
        collision_name = reference_well_collision_name(
            item,
            planned_names=planned_names,
            duplicate_name_keys=duplicate_reference_name_keys,
        )
        reference_model = _reference_uncertainty_model(
            reference_well=item,
            collision_name=collision_name,
            default_model=model,
            reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        )
        wells.append(
            build_anti_collision_well(
                name=collision_name,
                color=(name_to_color or {}).get(
                    str(collision_name),
                    REFERENCE_WELL_KIND_COLORS.get(str(item.kind), "#A0A0A0"),
                ),
                stations=item.stations,
                surface=item.surface,
                t1=None,
                t3=None,
                azimuth_deg=float(item.azimuth_deg),
                md_t1_m=None,
                md_t3_m=None,
                model=reference_model,
                include_display_geometry=include_display_geometry,
                well_kind=str(item.kind),
                is_reference_only=True,
                analysis_sample_step_m=effective_sample_step_m,
            )
        )
    return analyze_anti_collision(
        wells,
        build_overlap_geometry=build_overlap_geometry,
        pair_filter=lambda left, right: not paired_pilot_parent_names(
            left.name,
            right.name,
        ),
    )


def _reference_uncertainty_model(
    *,
    reference_well: ImportedTrajectoryWell,
    collision_name: str,
    default_model: PlanningUncertaintyModel,
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None,
) -> PlanningUncertaintyModel:
    if not reference_uncertainty_models_by_name:
        return default_model
    if str(getattr(reference_well, "kind", "")) != REFERENCE_WELL_ACTUAL:
        return default_model
    return (
        reference_uncertainty_models_by_name.get(str(collision_name))
        or reference_uncertainty_models_by_name.get(str(reference_well.name))
        or default_model
    )


def _fast_proxy_reference_uncertainty_models(
    models_by_name: Mapping[str, PlanningUncertaintyModel] | None,
) -> dict[str, PlanningUncertaintyModel] | None:
    if not models_by_name:
        return None
    return {
        str(name): fast_proxy_uncertainty_model(model)
        for name, model in models_by_name.items()
    }


def build_anticollision_well_contexts(
    successes: list[SuccessfulWellPlan],
) -> dict[str, AntiCollisionWellContext]:
    contexts: dict[str, AntiCollisionWellContext] = {}
    for success in successes:
        summary = dict(success.summary)
        kop_md_raw = summary.get("kop_md_m")
        kop_md_m = None
        if kop_md_raw is not None:
            try:
                kop_md_m = float(kop_md_raw)
            except (TypeError, ValueError):
                kop_md_m = None
        contexts[str(success.name)] = AntiCollisionWellContext(
            well_name=str(success.name),
            kop_md_m=kop_md_m,
            kop_min_vertical_m=float(success.config.kop_min_vertical_m),
            build1_dls_deg_per_30m=float(
                summary.get("build1_dls_selected_deg_per_30m", 0.0)
            ),
            build_dls_max_deg_per_30m=float(success.config.dls_build_max_deg_per_30m),
            md_t1_m=float(success.md_t1_m),
            md_total_m=float(summary.get("md_total_m", 0.0)),
            optimization_mode=str(success.config.optimization_mode),
            anti_collision_stage=(
                str(summary.get("anti_collision_stage")).strip()
                if summary.get("anti_collision_stage") is not None
                and str(summary.get("anti_collision_stage")).strip()
                else None
            ),
            anti_collision_attempted_stages=tuple(
                str(item).strip()
                for item in (
                    str(summary.get("anti_collision_attempted_stages")).split("|")
                    if isinstance(
                        summary.get("anti_collision_attempted_stages"),
                        str,
                    )
                    else []
                )
                if str(item).strip()
            ),
        )
    return contexts


def build_dynamic_cluster_execution_plan(
    *,
    successes: list[SuccessfulWellPlan],
    selected_names: set[str],
    target_well_names: tuple[str, ...],
    uncertainty_model: PlanningUncertaintyModel,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
    prefer_trajectory_stage: bool = False,
) -> DynamicClusterExecutionPlan | None:
    if not selected_names:
        return None
    if len(successes) + len(reference_wells) < 2:
        return DynamicClusterExecutionPlan(
            cluster=None,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state=DYNAMIC_CLUSTER_PLAN_RESOLVED,
        )
    analysis = build_anti_collision_analysis_for_successes(
        successes,
        model=fast_proxy_uncertainty_model(uncertainty_model),
        reference_wells=reference_wells,
        reference_uncertainty_models_by_name=(
            _fast_proxy_reference_uncertainty_models(
                reference_uncertainty_models_by_name
            )
        ),
        include_display_geometry=False,
        build_overlap_geometry=False,
    )
    recommendations = build_anti_collision_recommendations(
        analysis,
        well_context_by_name=build_anticollision_well_contexts(successes),
    )
    clusters = build_anti_collision_recommendation_clusters(recommendations)
    target_set = {str(name) for name in target_well_names if str(name).strip()}
    if not target_set:
        target_set = {str(name) for name in selected_names}
    relevant_clusters = [
        cluster
        for cluster in clusters
        if target_set.intersection(str(name) for name in cluster.well_names)
        and any(str(name) in selected_names for name in cluster.well_names)
    ]
    if not relevant_clusters:
        return DynamicClusterExecutionPlan(
            cluster=None,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state=DYNAMIC_CLUSTER_PLAN_RESOLVED,
        )
    blocking_clusters = [
        cluster for cluster in relevant_clusters if not bool(cluster.can_prepare_rerun)
    ]
    if blocking_clusters:
        cluster = blocking_clusters[0]
        return DynamicClusterExecutionPlan(
            cluster=cluster,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state=DYNAMIC_CLUSTER_PLAN_BLOCKED,
            blocking_reason=(
                str(cluster.blocking_advisory).strip()
                if cluster.blocking_advisory is not None
                else (
                    "Для текущего anti-collision кластера не осталось автоматических "
                    "шагов пересчета."
                )
            ),
        )
    cluster = relevant_clusters[0]
    prepared, skipped_wells = build_cluster_prepared_overrides(
        cluster,
        successes=successes,
        uncertainty_model=uncertainty_model,
        prefer_trajectory_stage=bool(
            prefer_trajectory_stage and int(cluster.trajectory_conflict_count) > 0
        ),
    )
    if not prepared:
        return DynamicClusterExecutionPlan(
            cluster=cluster,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=tuple(sorted(set(str(name) for name in skipped_wells))),
            resolution_state=DYNAMIC_CLUSTER_PLAN_BLOCKED,
            blocking_reason=(
                "Не удалось подготовить актуальные anti-collision overrides "
                "для оставшихся шагов кластера."
            ),
        )
    staged_frontier = _cluster_staged_frontier_well_names(
        cluster=cluster,
        selected_names=selected_names,
        prepared_by_well=prepared,
        prefer_trajectory_stage=bool(prefer_trajectory_stage),
    )
    active_prepared = (
        {
            str(name): prepared[str(name)]
            for name in staged_frontier
            if str(name) in prepared
        }
        if staged_frontier
        else dict(prepared)
    )
    ordered_well_names = tuple(
        str(name)
        for name in staged_frontier
        if str(name) in selected_names and str(name) in active_prepared
    )
    fallback_names = tuple(
        sorted(
            set(str(name) for name in active_prepared.keys())
            .intersection(selected_names)
            .difference(ordered_well_names)
        )
    )
    final_order = tuple([*ordered_well_names, *fallback_names])
    if not final_order:
        next_step = cluster.action_steps[0] if cluster.action_steps else None
        if next_step is not None:
            blocking_reason = (
                "Следующий рекомендуемый шаг снова относится к уже пересчитанной "
                f"скважине {str(next_step.well_name)} "
                f"({str(next_step.expected_maneuver)}). Текущий cluster-level прогон "
                "остановлен, чтобы не перетирать остальные скважины базовыми "
                "настройками."
            )
        else:
            blocking_reason = (
                "Для оставшихся скважин не осталось актуальных automatic "
                "cluster-level шагов пересчета."
            )
        return DynamicClusterExecutionPlan(
            cluster=cluster,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=tuple(sorted(set(str(name) for name in selected_names))),
            resolution_state=DYNAMIC_CLUSTER_PLAN_BLOCKED,
            blocking_reason=blocking_reason,
        )
    return DynamicClusterExecutionPlan(
        cluster=cluster,
        ordered_well_names=final_order,
        prepared_by_well=active_prepared,
        skipped_wells=tuple(sorted(set(str(name) for name in skipped_wells))),
        resolution_state=DYNAMIC_CLUSTER_PLAN_ACTIVE,
    )


def _cluster_staged_frontier_well_names(
    *,
    cluster: AntiCollisionRecommendationCluster,
    selected_names: set[str],
    prepared_by_well: Mapping[str, dict[str, object]],
    prefer_trajectory_stage: bool,
) -> tuple[str, ...]:
    selected_set = {str(name) for name in selected_names}
    prepared_set = {str(name) for name in prepared_by_well}
    step_by_well = {
        str(step.well_name): step
        for step in cluster.action_steps
        if str(step.well_name) in selected_set and str(step.well_name) in prepared_set
    }
    early_frontier = tuple(
        str(step.well_name)
        for step in cluster.action_steps
        if str(step.well_name) in selected_set
        and str(step.well_name) in prepared_set
        and str(step.category) in {"mixed", RECOMMENDATION_REDUCE_KOP}
    )
    early_frontier = tuple(
        sorted(
            early_frontier,
            key=lambda name: (
                1 if str(step_by_well[str(name)].category) == "mixed" else 0,
                int(step_by_well[str(name)].order_rank),
                str(name),
            ),
        )
    )
    trajectory_frontier = tuple(
        str(step.well_name)
        for step in cluster.action_steps
        if str(step.well_name) in selected_set
        and str(step.well_name) in prepared_set
        and str(step.category) in {"mixed", RECOMMENDATION_TRAJECTORY_REVIEW}
    )
    trajectory_frontier = tuple(
        sorted(
            trajectory_frontier,
            key=lambda name: (
                1
                if str(step_by_well[str(name)].category) == "mixed"
                else 0,
                int(step_by_well[str(name)].order_rank),
                str(name),
            ),
        )
    )
    if prefer_trajectory_stage and trajectory_frontier:
        return trajectory_frontier
    if not prefer_trajectory_stage and early_frontier:
        return early_frontier
    if trajectory_frontier:
        return trajectory_frontier
    if early_frontier:
        return early_frontier
    return tuple(
        str(step.well_name)
        for step in cluster.action_steps
        if str(step.well_name) in selected_set and str(step.well_name) in prepared_set
    )


def recommendation_intervals_for_moving_well(
    *,
    recommendation: AntiCollisionRecommendation,
    moving_well_name: str,
) -> tuple[str, float, float, float, float] | None:
    moving_name = str(moving_well_name)
    if moving_name == str(recommendation.well_a):
        return (
            str(recommendation.well_b),
            float(recommendation.md_a_start_m),
            float(recommendation.md_a_end_m),
            float(recommendation.md_b_start_m),
            float(recommendation.md_b_end_m),
        )
    if moving_name == str(recommendation.well_b):
        return (
            str(recommendation.well_a),
            float(recommendation.md_b_start_m),
            float(recommendation.md_b_end_m),
            float(recommendation.md_a_start_m),
            float(recommendation.md_a_end_m),
        )
    return None


def build_prepared_optimization_context(
    *,
    recommendation: AntiCollisionRecommendation,
    moving_success: SuccessfulWellPlan | None,
    reference_success: SuccessfulWellPlan | None,
    uncertainty_model: PlanningUncertaintyModel,
    all_successes: list[SuccessfulWellPlan] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
) -> AntiCollisionOptimizationContext | None:
    if moving_success is None:
        return None
    intervals = recommendation_intervals_for_moving_well(
        recommendation=recommendation,
        moving_well_name=str(moving_success.name),
    )
    if intervals is None:
        return None
    (
        _reference_name,
        candidate_md_start_m,
        candidate_md_end_m,
        reference_md_start_m,
        reference_md_end_m,
    ) = intervals
    runtime_sample_step_m = (
        _EARLY_ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M
        if str(recommendation.category) == RECOMMENDATION_REDUCE_KOP
        else _ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M
    )
    expected_maneuver = _expected_trajectory_maneuver_for_prepared_step(
        recommendation=recommendation,
        moving_well_name=str(moving_success.name),
    )
    reference_paths = _build_reference_paths_for_candidate_window(
        moving_well_name=str(moving_success.name),
        candidate_md_start_m=float(candidate_md_start_m),
        candidate_md_end_m=float(candidate_md_end_m),
        primary_reference_name=str(_reference_name),
        primary_reference_window=(float(reference_md_start_m), float(reference_md_end_m)),
        successes=list(all_successes or [item for item in (moving_success, reference_success) if item is not None]),
        reference_wells=reference_wells,
        sample_step_m=runtime_sample_step_m,
        uncertainty_model=uncertainty_model,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
    )
    if not reference_paths:
        return None
    moving_summary = dict(moving_success.summary)
    baseline_kop_vertical_m = moving_summary.get("kop_md_m")
    baseline_build1_dls_deg_per_30m = moving_summary.get(
        "build1_dls_selected_deg_per_30m",
        moving_summary.get("build_dls_selected_deg_per_30m"),
    )
    return AntiCollisionOptimizationContext(
        candidate_md_start_m=float(candidate_md_start_m),
        candidate_md_end_m=float(candidate_md_end_m),
        sf_target=1.0,
        sample_step_m=runtime_sample_step_m,
        uncertainty_model=uncertainty_model,
        references=tuple(reference_paths),
        prefer_lower_kop=bool(
            str(recommendation.category) == RECOMMENDATION_REDUCE_KOP
        ),
        prefer_higher_build1=bool(
            str(recommendation.category) == RECOMMENDATION_REDUCE_KOP
        ),
        prefer_keep_kop=bool(
            str(recommendation.category) == RECOMMENDATION_TRAJECTORY_REVIEW
        ),
        prefer_keep_build1=bool(
            str(recommendation.category) == RECOMMENDATION_TRAJECTORY_REVIEW
        ),
        prefer_adjust_build2=bool(
            str(recommendation.category) == RECOMMENDATION_TRAJECTORY_REVIEW
            and str(expected_maneuver) == MANEUVER_BUILD2_ENTRY
        ),
        baseline_kop_vertical_m=(
            float(baseline_kop_vertical_m)
            if baseline_kop_vertical_m is not None
            else None
        ),
        baseline_build1_dls_deg_per_30m=(
            float(baseline_build1_dls_deg_per_30m)
            if baseline_build1_dls_deg_per_30m is not None
            else None
        ),
        interpolation_method=str(getattr(moving_success.config, "interpolation_method", "rodrigues")),
    )


def _build_reference_paths_for_candidate_window(
    *,
    moving_well_name: str,
    candidate_md_start_m: float,
    candidate_md_end_m: float,
    primary_reference_name: str,
    primary_reference_window: tuple[float, float],
    successes: list[SuccessfulWellPlan],
    reference_wells: tuple[ImportedTrajectoryWell, ...],
    sample_step_m: float,
    uncertainty_model: PlanningUncertaintyModel,
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
) -> list[object]:
    success_by_name = {str(item.name): item for item in successes}
    reference_by_name = _reference_wells_by_collision_name(
        reference_wells,
        planned_names=tuple(success_by_name.keys()),
    )
    reference_windows: dict[str, tuple[float, float]] = {
        str(primary_reference_name): (
            float(primary_reference_window[0]),
            float(primary_reference_window[1]),
        )
    }
    for reference_name, reference_success in success_by_name.items():
        if str(reference_name) == str(moving_well_name):
            continue
        if str(reference_name) in reference_windows:
            continue
        if reference_success.stations.empty:
            continue
        md_values = reference_success.stations["MD_m"].to_numpy(dtype=float)
        if len(md_values) == 0:
            continue
        reference_start_m = max(
            float(md_values[0]),
            float(candidate_md_start_m) - _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
        )
        reference_end_m = min(
            float(md_values[-1]),
            float(candidate_md_end_m) + _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
        )
        if reference_end_m <= reference_start_m + 1e-6:
            reference_start_m = float(md_values[0])
            reference_end_m = float(md_values[-1])
        reference_windows[str(reference_name)] = (
            float(reference_start_m),
            float(reference_end_m),
        )
    for reference_name, reference_well in reference_by_name.items():
        if str(reference_name) == str(moving_well_name) or str(reference_name) in reference_windows:
            continue
        if reference_well.stations.empty:
            continue
        md_values = reference_well.stations["MD_m"].to_numpy(dtype=float)
        if len(md_values) == 0:
            continue
        reference_start_m = max(
            float(md_values[0]),
            float(candidate_md_start_m) - _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
        )
        reference_end_m = min(
            float(md_values[-1]),
            float(candidate_md_end_m) + _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
        )
        if reference_end_m <= reference_start_m + 1e-6:
            reference_start_m = float(md_values[0])
            reference_end_m = float(md_values[-1])
        reference_windows[str(reference_name)] = (
            float(reference_start_m),
            float(reference_end_m),
        )
    references: list[object] = []
    for reference_name, window in sorted(reference_windows.items()):
        reference_success = success_by_name.get(str(reference_name))
        if reference_success is not None:
            references.append(
                build_anti_collision_reference_path(
                    well_name=str(reference_success.name),
                    stations=reference_success.stations,
                    md_start_m=float(window[0]),
                    md_end_m=float(window[1]),
                    sample_step_m=float(sample_step_m),
                    model=uncertainty_model,
                )
            )
            continue
        reference_well = reference_by_name.get(str(reference_name))
        if reference_well is None:
            continue
        references.append(
            build_anti_collision_reference_path(
                well_name=str(reference_well.name),
                stations=reference_well.stations,
                md_start_m=float(window[0]),
                md_end_m=float(window[1]),
                sample_step_m=float(sample_step_m),
                model=_reference_uncertainty_model(
                    reference_well=reference_well,
                    collision_name=str(reference_name),
                    default_model=uncertainty_model,
                    reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
                ),
            )
        )
    return references


def build_recommendation_prepared_overrides(
    recommendation: AntiCollisionRecommendation,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
) -> tuple[dict[str, dict[str, object]], list[str], tuple[dict[str, object], ...]]:
    prepared: dict[str, dict[str, object]] = {}
    skipped_wells: list[str] = []
    success_by_name = {str(item.name): item for item in successes}
    steps: list[dict[str, object]] = []
    expected_by_well: dict[str, str] = {}
    for well_name in (str(recommendation.well_a), str(recommendation.well_b)):
        expected_by_well[well_name] = _expected_trajectory_maneuver_for_prepared_step(
            recommendation=recommendation,
            moving_well_name=well_name,
        )
    for order_rank, suggestion in enumerate(recommendation.override_suggestions, start=1):
        update_fields = dict(suggestion.config_updates)
        optimization_mode = str(update_fields.get("optimization_mode", "")).strip()
        optimization_context = None
        if optimization_mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE:
            reference_name = (
                str(recommendation.well_b)
                if str(suggestion.well_name) == str(recommendation.well_a)
                else str(recommendation.well_a)
            )
            optimization_context = build_prepared_optimization_context(
                recommendation=recommendation,
                moving_success=success_by_name.get(str(suggestion.well_name)),
                reference_success=success_by_name.get(reference_name),
                uncertainty_model=uncertainty_model,
                all_successes=successes,
                reference_wells=reference_wells,
                reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
            )
            if optimization_context is None:
                skipped_wells.append(str(suggestion.well_name))
                continue
        prepared[str(suggestion.well_name)] = {
            "update_fields": update_fields,
            "source": recommendation_display_label(recommendation),
            "reason": str(suggestion.reason),
            "optimization_context": optimization_context,
        }
        steps.append(
            {
                "order_rank": int(order_rank),
                "well_name": str(suggestion.well_name),
                "expected_maneuver": str(
                    expected_by_well.get(str(suggestion.well_name), recommendation.expected_maneuver)
                ),
            }
        )
    return prepared, sorted(set(str(name) for name in skipped_wells)), tuple(steps)


def _expected_trajectory_maneuver_for_prepared_step(
    *,
    recommendation: AntiCollisionRecommendation,
    moving_well_name: str,
) -> str:
    if str(recommendation.category) != RECOMMENDATION_TRAJECTORY_REVIEW:
        return str(recommendation.expected_maneuver)
    if str(moving_well_name) == str(recommendation.well_a):
        md_start = float(recommendation.md_a_start_m)
        md_end = float(recommendation.md_a_end_m)
    else:
        md_start = float(recommendation.md_b_start_m)
        md_end = float(recommendation.md_b_end_m)
    if md_end <= 0.0:
        return str(recommendation.expected_maneuver)
    if max(md_start, md_end) >= 3500.0:
        return MANEUVER_BUILD2_ENTRY
    return str(recommendation.expected_maneuver)


def build_cluster_prepared_overrides(
    cluster: AntiCollisionRecommendationCluster,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: Mapping[
        str, PlanningUncertaintyModel
    ] | None = None,
    prefer_trajectory_stage: bool = False,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    success_by_name = {str(item.name): item for item in successes}
    reference_by_name = _reference_wells_by_collision_name(
        reference_wells,
        planned_names=tuple(success_by_name.keys()),
    )
    step_by_well = {
        str(step.well_name): step
        for step in cluster.action_steps
        if str(step.well_name).strip()
    }
    trajectory_specs: dict[str, dict[str, object]] = {}
    vertical_specs: dict[str, list[str]] = {}
    skipped_wells: list[str] = []

    for recommendation in cluster.recommendations:
        for suggestion in recommendation.override_suggestions:
            well_name = str(suggestion.well_name)
            step = step_by_well.get(well_name)
            if step is not None:
                if (
                    str(step.category) == "mixed"
                    and str(step.expected_maneuver) == MANEUVER_KOP_AND_TRAJECTORY
                ):
                    if prefer_trajectory_stage:
                        if str(recommendation.category) != RECOMMENDATION_TRAJECTORY_REVIEW:
                            continue
                    elif str(recommendation.category) != RECOMMENDATION_REDUCE_KOP:
                        continue
                if (
                    str(step.category) == RECOMMENDATION_REDUCE_KOP
                    and str(recommendation.category) != RECOMMENDATION_REDUCE_KOP
                ):
                    continue
                if (
                    str(step.category) == RECOMMENDATION_TRAJECTORY_REVIEW
                    and str(recommendation.category) != RECOMMENDATION_TRAJECTORY_REVIEW
                ):
                    continue
            update_fields = dict(suggestion.config_updates)
            optimization_mode = str(update_fields.get("optimization_mode", "")).strip()
            if optimization_mode == OPTIMIZATION_ANTI_COLLISION_AVOIDANCE:
                intervals = recommendation_intervals_for_moving_well(
                    recommendation=recommendation,
                    moving_well_name=well_name,
                )
                moving_success = success_by_name.get(well_name)
                if intervals is None or moving_success is None:
                    skipped_wells.append(well_name)
                    continue
                (
                    reference_name,
                    candidate_md_start_m,
                    candidate_md_end_m,
                    reference_md_start_m,
                    reference_md_end_m,
                ) = intervals
                reference_success = success_by_name.get(reference_name)
                if reference_success is None and reference_by_name.get(reference_name) is None:
                    skipped_wells.append(well_name)
                    continue
                entry = trajectory_specs.setdefault(
                    well_name,
                    {
                        "candidate_md_start_m": float(candidate_md_start_m),
                        "candidate_md_end_m": float(candidate_md_end_m),
                        "reference_windows": {},
                        "reasons": [],
                        "prefer_lower_kop": False,
                        "prefer_higher_build1": False,
                        "prefer_keep_kop": False,
                        "prefer_keep_build1": False,
                        "prefer_adjust_build2": False,
                    },
                )
                entry["candidate_md_start_m"] = min(
                    float(entry["candidate_md_start_m"]),
                    float(candidate_md_start_m),
                )
                entry["candidate_md_end_m"] = max(
                    float(entry["candidate_md_end_m"]),
                    float(candidate_md_end_m),
                )
                reference_windows = dict(entry["reference_windows"])
                current_window = reference_windows.get(str(reference_name))
                if current_window is None:
                    reference_windows[str(reference_name)] = (
                        float(reference_md_start_m),
                        float(reference_md_end_m),
                    )
                else:
                    reference_windows[str(reference_name)] = (
                        min(float(current_window[0]), float(reference_md_start_m)),
                        max(float(current_window[1]), float(reference_md_end_m)),
                    )
                entry["reference_windows"] = reference_windows
                reasons = list(entry["reasons"])
                reasons.append(str(suggestion.reason))
                entry["reasons"] = reasons
                if str(recommendation.category) == "reduce_kop":
                    entry["prefer_lower_kop"] = True
                    entry["prefer_higher_build1"] = True
                if str(recommendation.category) == RECOMMENDATION_TRAJECTORY_REVIEW:
                    entry["prefer_keep_kop"] = True
                    entry["prefer_keep_build1"] = True
                    if (
                        _expected_trajectory_maneuver_for_prepared_step(
                            recommendation=recommendation,
                            moving_well_name=well_name,
                        )
                        == MANEUVER_BUILD2_ENTRY
                    ):
                        entry["prefer_adjust_build2"] = True
                continue

            if optimization_mode == OPTIMIZATION_MINIMIZE_KOP:
                vertical_specs.setdefault(well_name, []).append(str(suggestion.reason))

    prepared: dict[str, dict[str, object]] = {}
    source_label = (
        f"{cluster.cluster_id} · {', '.join(cluster.well_names)} · "
        f"событий {int(cluster.recommendation_count)} · SF {float(cluster.worst_separation_factor):.2f}"
    )
    ordered_wells = [
        str(step.well_name)
        for step in cluster.action_steps
        if str(step.well_name) in trajectory_specs or str(step.well_name) in vertical_specs
    ]
    fallback_wells = sorted(
        set(trajectory_specs).union(vertical_specs).difference(ordered_wells)
    )
    for well_name in [*ordered_wells, *fallback_wells]:
        if well_name in trajectory_specs:
            spec = dict(trajectory_specs[well_name])
            runtime_sample_step_m = (
                _EARLY_ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M
                if bool(spec.get("prefer_lower_kop", False))
                and bool(spec.get("prefer_higher_build1", False))
                else _ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M
            )
            references: list[object] = []
            reference_windows = {
                str(reference_name): (
                    float(window[0]),
                    float(window[1]),
                )
                for reference_name, window in dict(spec.get("reference_windows", {})).items()
            }
            for reference_name, reference_success in success_by_name.items():
                if str(reference_name) == well_name or str(reference_name) in reference_windows:
                    continue
                if reference_success.stations.empty:
                    continue
                md_values = reference_success.stations["MD_m"].to_numpy(dtype=float)
                if len(md_values) == 0:
                    continue
                candidate_start_m = float(spec["candidate_md_start_m"])
                candidate_end_m = float(spec["candidate_md_end_m"])
                reference_start_m = max(
                    float(md_values[0]),
                    candidate_start_m - _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
                )
                reference_end_m = min(
                    float(md_values[-1]),
                    candidate_end_m + _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
                )
                if reference_end_m <= reference_start_m + 1e-6:
                    reference_start_m = float(md_values[0])
                    reference_end_m = float(md_values[-1])
                reference_windows[str(reference_name)] = (
                    float(reference_start_m),
                    float(reference_end_m),
                )
            for reference_name, reference_well in reference_by_name.items():
                if str(reference_name) == well_name or str(reference_name) in reference_windows:
                    continue
                if reference_well.stations.empty:
                    continue
                md_values = reference_well.stations["MD_m"].to_numpy(dtype=float)
                if len(md_values) == 0:
                    continue
                candidate_start_m = float(spec["candidate_md_start_m"])
                candidate_end_m = float(spec["candidate_md_end_m"])
                reference_start_m = max(
                    float(md_values[0]),
                    candidate_start_m - _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
                )
                reference_end_m = min(
                    float(md_values[-1]),
                    candidate_end_m + _CLUSTER_REFERENCE_WINDOW_MARGIN_M,
                )
                if reference_end_m <= reference_start_m + 1e-6:
                    reference_start_m = float(md_values[0])
                    reference_end_m = float(md_values[-1])
                reference_windows[str(reference_name)] = (
                    float(reference_start_m),
                    float(reference_end_m),
                )
            for reference_name, window in sorted(reference_windows.items()):
                reference_success = success_by_name.get(str(reference_name))
                if reference_success is not None:
                    references.append(
                        build_anti_collision_reference_path(
                            well_name=str(reference_success.name),
                            stations=reference_success.stations,
                            md_start_m=float(window[0]),
                            md_end_m=float(window[1]),
                            sample_step_m=runtime_sample_step_m,
                            model=uncertainty_model,
                        )
                    )
                    continue
                reference_well = reference_by_name.get(str(reference_name))
                if reference_well is None:
                    skipped_wells.append(well_name)
                    references = []
                    break
                references.append(
                    build_anti_collision_reference_path(
                        well_name=str(reference_well.name),
                        stations=reference_well.stations,
                        md_start_m=float(window[0]),
                        md_end_m=float(window[1]),
                        sample_step_m=runtime_sample_step_m,
                        model=_reference_uncertainty_model(
                            reference_well=reference_well,
                            collision_name=str(reference_name),
                            default_model=uncertainty_model,
                            reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
                        ),
                    )
                )
            if not references:
                continue
            combined_reasons = list(spec["reasons"])
            combined_reasons.extend(vertical_specs.get(well_name, []))
            context = AntiCollisionOptimizationContext(
                candidate_md_start_m=float(spec["candidate_md_start_m"]),
                candidate_md_end_m=float(spec["candidate_md_end_m"]),
                sf_target=1.0,
                sample_step_m=runtime_sample_step_m,
                uncertainty_model=uncertainty_model,
                references=tuple(references),
                prefer_lower_kop=bool(spec.get("prefer_lower_kop", False)),
                prefer_higher_build1=bool(spec.get("prefer_higher_build1", False)),
                prefer_keep_kop=bool(spec.get("prefer_keep_kop", False)),
                prefer_keep_build1=bool(spec.get("prefer_keep_build1", False)),
                prefer_adjust_build2=bool(spec.get("prefer_adjust_build2", False)),
                baseline_kop_vertical_m=(
                    float(success_by_name[well_name].summary.get("kop_md_m"))
                    if success_by_name.get(well_name) is not None
                    and success_by_name[well_name].summary.get("kop_md_m") is not None
                    else None
                ),
                baseline_build1_dls_deg_per_30m=(
                    float(
                        success_by_name[well_name].summary.get(
                            "build1_dls_selected_deg_per_30m",
                            success_by_name[well_name].summary.get(
                                "build_dls_selected_deg_per_30m"
                            ),
                        )
                    )
                    if success_by_name.get(well_name) is not None
                    and (
                        success_by_name[well_name].summary.get(
                            "build1_dls_selected_deg_per_30m"
                        )
                        is not None
                        or success_by_name[well_name].summary.get(
                            "build_dls_selected_deg_per_30m"
                        )
                        is not None
                    )
                    else None
                ),
                interpolation_method=(
                    str(getattr(success_by_name[well_name].config, "interpolation_method", "rodrigues"))
                    if success_by_name.get(well_name) is not None
                    else "rodrigues"
                ),
            )
            prepared[well_name] = {
                "update_fields": {
                    "optimization_mode": OPTIMIZATION_ANTI_COLLISION_AVOIDANCE,
                },
                "source": source_label,
                "reason": " | ".join(dict.fromkeys(str(item) for item in combined_reasons)),
                "optimization_context": context,
            }
            continue

        reasons = vertical_specs.get(well_name, [])
        prepared[well_name] = {
            "update_fields": {"optimization_mode": OPTIMIZATION_MINIMIZE_KOP},
            "source": source_label,
            "reason": " | ".join(dict.fromkeys(str(item) for item in reasons)),
            "optimization_context": None,
        }
    return prepared, sorted(set(str(name) for name in skipped_wells))
