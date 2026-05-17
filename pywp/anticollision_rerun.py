from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

from pywp.anticollision import (
    AntiCollisionAnalysis,
    AntiCollisionIncrementalStats,
    AntiCollisionPairCacheEntry,
    AntiCollisionProgress,
    AntiCollisionWell,
    DEFINITIVE_SCAN_STEP_M,
    analyze_anti_collision,
    analyze_anti_collision_incremental,
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
from pywp.anticollision_rerun_models import (
    PreparedOverride,
    ReferenceWindow,
    TrajectoryOverrideSpec,
)
from pywp.models import OPTIMIZATION_ANTI_COLLISION_AVOIDANCE, OPTIMIZATION_MINIMIZE_KOP
from pywp.pilot_wells import paired_pilot_parent_names, well_name_key
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
        str(name).strip().casefold() for name in planned_names if str(name).strip()
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


def _excluded_source_parent_pair_keys(
    *,
    successes: list[SuccessfulWellPlan],
    reference_wells: tuple[ImportedTrajectoryWell, ...],
) -> set[tuple[str, str]]:
    """Pairs that represent the same physical bore and should not be scored."""

    planned_names = tuple(str(item.name) for item in successes)
    duplicate_reference_name_keys = reference_well_duplicate_name_keys(reference_wells)
    excluded: set[tuple[str, str]] = set()
    for success in successes:
        summary = dict(getattr(success, "summary", {}) or {})
        if str(summary.get("trajectory_type", "")).strip() != "FACT_SIDETRACK":
            continue
        parent_name = str(summary.get("actual_parent_well_name", "")).strip()
        if not parent_name:
            parent_name = str(summary.get("sidetrack_parent_well_name", "")).strip()
        if not parent_name:
            continue
        parent_kind = str(summary.get("sidetrack_parent_kind", REFERENCE_WELL_ACTUAL))
        for reference_well in reference_wells:
            if str(reference_well.kind) != parent_kind:
                continue
            if well_name_key(reference_well.name) != well_name_key(parent_name):
                continue
            collision_name = reference_well_collision_name(
                reference_well,
                planned_names=planned_names,
                duplicate_name_keys=duplicate_reference_name_keys,
            )
            excluded.add(tuple(sorted((str(success.name), str(collision_name)))))
    return excluded


def _should_score_anti_collision_pair(
    left: AntiCollisionWell,
    right: AntiCollisionWell,
    *,
    excluded_pair_keys: set[tuple[str, str]],
) -> bool:
    left_name = str(left.name)
    right_name = str(right.name)
    if paired_pilot_parent_names(left_name, right_name):
        return False
    if tuple(sorted((left_name, right_name))) in excluded_pair_keys:
        return False
    return True


def build_anti_collision_analysis_for_successes(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: Mapping[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    include_display_geometry: bool = True,
    build_overlap_geometry: bool = True,
    analysis_sample_step_m: float | None = None,
    progress_callback: Callable[[AntiCollisionProgress], None] | None = None,
    parallel_workers: int = 0,
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
    duplicate_reference_name_keys = reference_well_duplicate_name_keys(reference_wells)
    excluded_pair_keys = _excluded_source_parent_pair_keys(
        successes=successes,
        reference_wells=reference_wells,
    )
    wells = [
        build_anti_collision_well(
            name=item.name,
            color=(name_to_color or {}).get(str(item.name), "#A0A0A0"),
            stations=item.stations,
            surface=item.surface,
            t1=item.t1,
            t3=item.t3,
            target_pairs=tuple(getattr(item, "target_pairs", ()) or ()),
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
        progress_callback=progress_callback,
        parallel_workers=int(parallel_workers),
        pair_filter=lambda left, right: _should_score_anti_collision_pair(
            left,
            right,
            excluded_pair_keys=excluded_pair_keys,
        ),
    )


def build_incremental_anti_collision_analysis_for_successes(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: Mapping[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    include_display_geometry: bool = True,
    build_overlap_geometry: bool = True,
    analysis_sample_step_m: float | None = None,
    well_signature_by_name: Mapping[str, str] | None = None,
    previous_well_cache: Mapping[str, tuple[str, AntiCollisionWell]] | None = None,
    previous_pair_cache: (
        Mapping[tuple[str, str], AntiCollisionPairCacheEntry] | None
    ) = None,
    progress_callback: Callable[[AntiCollisionProgress], None] | None = None,
    parallel_workers: int = 0,
) -> tuple[
    AntiCollisionAnalysis,
    dict[str, tuple[str, AntiCollisionWell]],
    dict[tuple[str, str], AntiCollisionPairCacheEntry],
    AntiCollisionIncrementalStats,
]:
    wells, well_cache, reused_wells, rebuilt_wells = (
        build_anti_collision_wells_for_successes(
            successes,
            model=model,
            name_to_color=name_to_color,
            reference_wells=reference_wells,
            reference_uncertainty_models_by_name=(reference_uncertainty_models_by_name),
            include_display_geometry=include_display_geometry,
            build_overlap_geometry=build_overlap_geometry,
            analysis_sample_step_m=analysis_sample_step_m,
            well_signature_by_name=well_signature_by_name,
            previous_well_cache=previous_well_cache,
        )
    )
    excluded_pair_keys = _excluded_source_parent_pair_keys(
        successes=successes,
        reference_wells=reference_wells,
    )
    analysis, pair_cache, stats = analyze_anti_collision_incremental(
        wells,
        build_overlap_geometry=build_overlap_geometry,
        well_signature_by_name=well_signature_by_name,
        previous_pair_cache=previous_pair_cache,
        reused_well_count=reused_wells,
        rebuilt_well_count=rebuilt_wells,
        progress_callback=progress_callback,
        parallel_workers=int(parallel_workers),
        pair_filter=lambda left, right: _should_score_anti_collision_pair(
            left,
            right,
            excluded_pair_keys=excluded_pair_keys,
        ),
    )
    return analysis, well_cache, pair_cache, stats


def build_anti_collision_wells_for_successes(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: Mapping[str, str] | None = None,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    include_display_geometry: bool = True,
    build_overlap_geometry: bool = True,
    analysis_sample_step_m: float | None = None,
    well_signature_by_name: Mapping[str, str] | None = None,
    previous_well_cache: Mapping[str, tuple[str, AntiCollisionWell]] | None = None,
) -> tuple[
    tuple[AntiCollisionWell, ...],
    dict[str, tuple[str, AntiCollisionWell]],
    int,
    int,
]:
    effective_sample_step_m = (
        float(analysis_sample_step_m)
        if analysis_sample_step_m is not None
        else (
            float(DEFINITIVE_SCAN_STEP_M)
            if include_display_geometry and build_overlap_geometry
            else None
        )
    )
    signatures = {
        str(name): str(signature)
        for name, signature in (well_signature_by_name or {}).items()
    }
    previous_cache = dict(previous_well_cache or {})
    planned_names = tuple(str(item.name) for item in successes)
    duplicate_reference_name_keys = reference_well_duplicate_name_keys(reference_wells)
    wells: list[AntiCollisionWell] = []
    next_cache: dict[str, tuple[str, AntiCollisionWell]] = {}
    reused_count = 0
    rebuilt_count = 0

    def _reuse_or_build(
        *,
        name: str,
        builder: Callable[[], AntiCollisionWell],
    ) -> AntiCollisionWell:
        nonlocal reused_count, rebuilt_count
        signature = signatures.get(str(name), "")
        previous = previous_cache.get(str(name))
        if (
            previous is not None
            and len(previous) == 2
            and str(previous[0]) == signature
            and isinstance(previous[1], AntiCollisionWell)
        ):
            reused_count += 1
            well = previous[1]
        else:
            rebuilt_count += 1
            well = builder()
        next_cache[str(name)] = (signature, well)
        return well

    for item in successes:
        name = str(item.name)
        wells.append(
            _reuse_or_build(
                name=name,
                builder=lambda item=item, name=name: build_anti_collision_well(
                    name=name,
                    color=(name_to_color or {}).get(name, "#A0A0A0"),
                    stations=item.stations,
                    surface=item.surface,
                    t1=item.t1,
                    t3=item.t3,
                    target_pairs=tuple(getattr(item, "target_pairs", ()) or ()),
                    azimuth_deg=float(item.azimuth_deg),
                    md_t1_m=float(item.md_t1_m),
                    model=model,
                    include_display_geometry=include_display_geometry,
                    well_kind="project",
                    is_reference_only=False,
                    analysis_sample_step_m=effective_sample_step_m,
                ),
            )
        )

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
            _reuse_or_build(
                name=str(collision_name),
                builder=(
                    lambda item=item, collision_name=str(
                        collision_name
                    ), reference_model=reference_model: build_anti_collision_well(
                        name=collision_name,
                        color=(name_to_color or {}).get(
                            collision_name,
                            REFERENCE_WELL_KIND_COLORS.get(
                                str(item.kind),
                                "#A0A0A0",
                            ),
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
                ),
            )
        )

    return tuple(wells), next_cache, int(reused_count), int(rebuilt_count)


def _reference_uncertainty_model(
    *,
    reference_well: ImportedTrajectoryWell,
    collision_name: str,
    default_model: PlanningUncertaintyModel,
    reference_uncertainty_models_by_name: Mapping[str, PlanningUncertaintyModel] | None,
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
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
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
                1 if str(step_by_well[str(name)].category) == "mixed" else 0,
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
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
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
        primary_reference_window=(
            float(reference_md_start_m),
            float(reference_md_end_m),
        ),
        successes=list(
            all_successes
            or [
                item for item in (moving_success, reference_success) if item is not None
            ]
        ),
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
        interpolation_method=str(
            getattr(moving_success.config, "interpolation_method", "rodrigues")
        ),
    )


def _reference_window_for_candidate_window(
    *,
    reference_name: str,
    md_values: Any,
    candidate_md_start_m: float,
    candidate_md_end_m: float,
) -> ReferenceWindow | None:
    if len(md_values) == 0:
        return None
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
    return ReferenceWindow(
        well_name=str(reference_name),
        md_start_m=float(reference_start_m),
        md_end_m=float(reference_end_m),
    )


def _collect_reference_windows_for_candidate(
    *,
    moving_well_name: str,
    candidate_md_start_m: float,
    candidate_md_end_m: float,
    initial_windows: Mapping[str, ReferenceWindow],
    success_by_name: Mapping[str, SuccessfulWellPlan],
    reference_by_name: Mapping[str, ImportedTrajectoryWell],
) -> dict[str, ReferenceWindow]:
    reference_windows = {
        str(reference_name): window
        for reference_name, window in initial_windows.items()
    }
    for reference_name, reference_success in success_by_name.items():
        if (
            str(reference_name) == str(moving_well_name)
            or str(reference_name) in reference_windows
            or reference_success.stations.empty
        ):
            continue
        window = _reference_window_for_candidate_window(
            reference_name=str(reference_name),
            md_values=reference_success.stations["MD_m"].to_numpy(dtype=float),
            candidate_md_start_m=float(candidate_md_start_m),
            candidate_md_end_m=float(candidate_md_end_m),
        )
        if window is not None:
            reference_windows[str(reference_name)] = window
    for reference_name, reference_well in reference_by_name.items():
        if (
            str(reference_name) == str(moving_well_name)
            or str(reference_name) in reference_windows
            or reference_well.stations.empty
        ):
            continue
        window = _reference_window_for_candidate_window(
            reference_name=str(reference_name),
            md_values=reference_well.stations["MD_m"].to_numpy(dtype=float),
            candidate_md_start_m=float(candidate_md_start_m),
            candidate_md_end_m=float(candidate_md_end_m),
        )
        if window is not None:
            reference_windows[str(reference_name)] = window
    return reference_windows


def _build_reference_paths_from_windows(
    *,
    reference_windows: Mapping[str, ReferenceWindow],
    success_by_name: Mapping[str, SuccessfulWellPlan],
    reference_by_name: Mapping[str, ImportedTrajectoryWell],
    sample_step_m: float,
    uncertainty_model: PlanningUncertaintyModel,
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    fail_on_missing_reference: bool = False,
) -> tuple[list[object], bool]:
    references: list[object] = []
    for reference_name, window in sorted(reference_windows.items()):
        reference_success = success_by_name.get(str(reference_name))
        if reference_success is not None:
            references.append(
                build_anti_collision_reference_path(
                    well_name=str(reference_success.name),
                    stations=reference_success.stations,
                    md_start_m=float(window.md_start_m),
                    md_end_m=float(window.md_end_m),
                    sample_step_m=float(sample_step_m),
                    model=uncertainty_model,
                )
            )
            continue
        reference_well = reference_by_name.get(str(reference_name))
        if reference_well is None:
            if fail_on_missing_reference:
                return [], True
            continue
        references.append(
            build_anti_collision_reference_path(
                well_name=str(reference_well.name),
                stations=reference_well.stations,
                md_start_m=float(window.md_start_m),
                md_end_m=float(window.md_end_m),
                sample_step_m=float(sample_step_m),
                model=_reference_uncertainty_model(
                    reference_well=reference_well,
                    collision_name=str(reference_name),
                    default_model=uncertainty_model,
                    reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
                ),
            )
        )
    return references, False


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
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
) -> list[object]:
    success_by_name = {str(item.name): item for item in successes}
    reference_by_name = _reference_wells_by_collision_name(
        reference_wells,
        planned_names=tuple(success_by_name.keys()),
    )
    reference_windows = _collect_reference_windows_for_candidate(
        moving_well_name=str(moving_well_name),
        candidate_md_start_m=float(candidate_md_start_m),
        candidate_md_end_m=float(candidate_md_end_m),
        initial_windows={
            str(primary_reference_name): ReferenceWindow(
                well_name=str(primary_reference_name),
                md_start_m=float(primary_reference_window[0]),
                md_end_m=float(primary_reference_window[1]),
            )
        },
        success_by_name=success_by_name,
        reference_by_name=reference_by_name,
    )
    references, _missing_reference = _build_reference_paths_from_windows(
        reference_windows=reference_windows,
        success_by_name=success_by_name,
        reference_by_name=reference_by_name,
        sample_step_m=float(sample_step_m),
        uncertainty_model=uncertainty_model,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        fail_on_missing_reference=False,
    )
    return references


def build_recommendation_prepared_overrides(
    recommendation: AntiCollisionRecommendation,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
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
    for order_rank, suggestion in enumerate(
        recommendation.override_suggestions, start=1
    ):
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
                    expected_by_well.get(
                        str(suggestion.well_name), recommendation.expected_maneuver
                    )
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


def _cluster_source_label(cluster: AntiCollisionRecommendationCluster) -> str:
    return (
        f"{cluster.cluster_id} · {', '.join(cluster.well_names)} · "
        f"событий {int(cluster.recommendation_count)} · "
        f"SF {float(cluster.worst_separation_factor):.2f}"
    )


def _cluster_recommendation_matches_step(
    *,
    recommendation: AntiCollisionRecommendation,
    step: object | None,
    prefer_trajectory_stage: bool,
) -> bool:
    if step is None:
        return True
    step_category = str(getattr(step, "category", ""))
    step_maneuver = str(getattr(step, "expected_maneuver", ""))
    recommendation_category = str(recommendation.category)
    if step_category == "mixed" and step_maneuver == MANEUVER_KOP_AND_TRAJECTORY:
        if prefer_trajectory_stage:
            return recommendation_category == RECOMMENDATION_TRAJECTORY_REVIEW
        return recommendation_category == RECOMMENDATION_REDUCE_KOP
    if (
        step_category == RECOMMENDATION_REDUCE_KOP
        and recommendation_category != RECOMMENDATION_REDUCE_KOP
    ):
        return False
    if (
        step_category == RECOMMENDATION_TRAJECTORY_REVIEW
        and recommendation_category != RECOMMENDATION_TRAJECTORY_REVIEW
    ):
        return False
    return True


def _collect_cluster_override_specs(
    *,
    cluster: AntiCollisionRecommendationCluster,
    success_by_name: Mapping[str, SuccessfulWellPlan],
    reference_by_name: Mapping[str, ImportedTrajectoryWell],
    prefer_trajectory_stage: bool,
) -> tuple[dict[str, TrajectoryOverrideSpec], dict[str, list[str]], list[str]]:
    step_by_well = {
        str(step.well_name): step
        for step in cluster.action_steps
        if str(step.well_name).strip()
    }
    trajectory_specs: dict[str, TrajectoryOverrideSpec] = {}
    vertical_specs: dict[str, list[str]] = {}
    skipped_wells: list[str] = []

    for recommendation in cluster.recommendations:
        for suggestion in recommendation.override_suggestions:
            well_name = str(suggestion.well_name)
            if not _cluster_recommendation_matches_step(
                recommendation=recommendation,
                step=step_by_well.get(well_name),
                prefer_trajectory_stage=bool(prefer_trajectory_stage),
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
                if (
                    success_by_name.get(reference_name) is None
                    and reference_by_name.get(reference_name) is None
                ):
                    skipped_wells.append(well_name)
                    continue
                spec = trajectory_specs.get(well_name)
                if spec is None:
                    spec = TrajectoryOverrideSpec(
                        well_name=well_name,
                        candidate_md_start_m=float(candidate_md_start_m),
                        candidate_md_end_m=float(candidate_md_end_m),
                    )
                    trajectory_specs[well_name] = spec
                else:
                    spec.expand_candidate_window(
                        float(candidate_md_start_m),
                        float(candidate_md_end_m),
                    )
                spec.add_reference_window(
                    reference_name=str(reference_name),
                    md_start_m=float(reference_md_start_m),
                    md_end_m=float(reference_md_end_m),
                )
                spec.add_reason(str(suggestion.reason))
                if str(recommendation.category) == RECOMMENDATION_REDUCE_KOP:
                    spec.prefer_lower_kop = True
                    spec.prefer_higher_build1 = True
                if str(recommendation.category) == RECOMMENDATION_TRAJECTORY_REVIEW:
                    spec.prefer_keep_kop = True
                    spec.prefer_keep_build1 = True
                    expected_maneuver = _expected_trajectory_maneuver_for_prepared_step(
                        recommendation=recommendation,
                        moving_well_name=well_name,
                    )
                    if expected_maneuver == MANEUVER_BUILD2_ENTRY:
                        spec.prefer_adjust_build2 = True
                continue

            if optimization_mode == OPTIMIZATION_MINIMIZE_KOP:
                vertical_specs.setdefault(well_name, []).append(str(suggestion.reason))
    return trajectory_specs, vertical_specs, skipped_wells


def _cluster_override_well_order(
    *,
    cluster: AntiCollisionRecommendationCluster,
    trajectory_specs: Mapping[str, TrajectoryOverrideSpec],
    vertical_specs: Mapping[str, list[str]],
) -> list[str]:
    ordered_wells = [
        str(step.well_name)
        for step in cluster.action_steps
        if str(step.well_name) in trajectory_specs
        or str(step.well_name) in vertical_specs
    ]
    fallback_wells = sorted(
        set(trajectory_specs).union(vertical_specs).difference(ordered_wells)
    )
    return [*ordered_wells, *fallback_wells]


def _runtime_sample_step_for_trajectory_spec(spec: TrajectoryOverrideSpec) -> float:
    if bool(spec.prefer_lower_kop) and bool(spec.prefer_higher_build1):
        return float(_EARLY_ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M)
    return float(_ANTI_COLLISION_RUNTIME_SAMPLE_STEP_M)


def _deduplicated_reason_text(reasons: list[str]) -> str:
    return " | ".join(dict.fromkeys(str(item) for item in reasons))


def _build_trajectory_prepared_override(
    *,
    well_name: str,
    spec: TrajectoryOverrideSpec,
    vertical_reasons: list[str],
    source_label: str,
    success_by_name: Mapping[str, SuccessfulWellPlan],
    reference_by_name: Mapping[str, ImportedTrajectoryWell],
    uncertainty_model: PlanningUncertaintyModel,
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
) -> tuple[PreparedOverride | None, bool]:
    moving_success = success_by_name.get(str(well_name))
    if moving_success is None:
        return None, True
    runtime_sample_step_m = _runtime_sample_step_for_trajectory_spec(spec)
    reference_windows = _collect_reference_windows_for_candidate(
        moving_well_name=str(well_name),
        candidate_md_start_m=float(spec.candidate_md_start_m),
        candidate_md_end_m=float(spec.candidate_md_end_m),
        initial_windows=spec.reference_windows,
        success_by_name=success_by_name,
        reference_by_name=reference_by_name,
    )
    references, missing_reference = _build_reference_paths_from_windows(
        reference_windows=reference_windows,
        success_by_name=success_by_name,
        reference_by_name=reference_by_name,
        sample_step_m=runtime_sample_step_m,
        uncertainty_model=uncertainty_model,
        reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
        fail_on_missing_reference=True,
    )
    if missing_reference:
        return None, True
    if not references:
        return None, False
    moving_summary = dict(moving_success.summary)
    baseline_build1_dls_deg_per_30m = moving_summary.get(
        "build1_dls_selected_deg_per_30m",
        moving_summary.get("build_dls_selected_deg_per_30m"),
    )
    combined_reasons = [*spec.reasons, *vertical_reasons]
    context = AntiCollisionOptimizationContext(
        candidate_md_start_m=float(spec.candidate_md_start_m),
        candidate_md_end_m=float(spec.candidate_md_end_m),
        sf_target=1.0,
        sample_step_m=runtime_sample_step_m,
        uncertainty_model=uncertainty_model,
        references=tuple(references),
        prefer_lower_kop=bool(spec.prefer_lower_kop),
        prefer_higher_build1=bool(spec.prefer_higher_build1),
        prefer_keep_kop=bool(spec.prefer_keep_kop),
        prefer_keep_build1=bool(spec.prefer_keep_build1),
        prefer_adjust_build2=bool(spec.prefer_adjust_build2),
        baseline_kop_vertical_m=(
            float(moving_summary.get("kop_md_m"))
            if moving_summary.get("kop_md_m") is not None
            else None
        ),
        baseline_build1_dls_deg_per_30m=(
            float(baseline_build1_dls_deg_per_30m)
            if baseline_build1_dls_deg_per_30m is not None
            else None
        ),
        interpolation_method=str(
            getattr(moving_success.config, "interpolation_method", "rodrigues")
        ),
    )
    return (
        PreparedOverride(
            update_fields={"optimization_mode": OPTIMIZATION_ANTI_COLLISION_AVOIDANCE},
            source=source_label,
            reason=_deduplicated_reason_text(combined_reasons),
            optimization_context=context,
        ),
        False,
    )


def _prepared_overrides_as_payloads(
    overrides: Mapping[str, PreparedOverride],
) -> dict[str, dict[str, object]]:
    return {str(name): override.as_payload() for name, override in overrides.items()}


def build_cluster_prepared_overrides(
    cluster: AntiCollisionRecommendationCluster,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
    reference_wells: tuple[ImportedTrajectoryWell, ...] = (),
    reference_uncertainty_models_by_name: (
        Mapping[str, PlanningUncertaintyModel] | None
    ) = None,
    prefer_trajectory_stage: bool = False,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    success_by_name = {str(item.name): item for item in successes}
    reference_by_name = _reference_wells_by_collision_name(
        reference_wells,
        planned_names=tuple(success_by_name.keys()),
    )
    trajectory_specs, vertical_specs, skipped_wells = _collect_cluster_override_specs(
        cluster=cluster,
        success_by_name=success_by_name,
        reference_by_name=reference_by_name,
        prefer_trajectory_stage=bool(prefer_trajectory_stage),
    )
    source_label = _cluster_source_label(cluster)
    prepared: dict[str, PreparedOverride] = {}
    for well_name in _cluster_override_well_order(
        cluster=cluster,
        trajectory_specs=trajectory_specs,
        vertical_specs=vertical_specs,
    ):
        if well_name in trajectory_specs:
            override, skipped = _build_trajectory_prepared_override(
                well_name=well_name,
                spec=trajectory_specs[well_name],
                vertical_reasons=vertical_specs.get(well_name, []),
                source_label=source_label,
                success_by_name=success_by_name,
                reference_by_name=reference_by_name,
                uncertainty_model=uncertainty_model,
                reference_uncertainty_models_by_name=reference_uncertainty_models_by_name,
            )
            if skipped:
                skipped_wells.append(well_name)
            if override is not None:
                prepared[well_name] = override
            continue

        reasons = vertical_specs.get(well_name, [])
        prepared[well_name] = PreparedOverride(
            update_fields={"optimization_mode": OPTIMIZATION_MINIMIZE_KOP},
            source=source_label,
            reason=_deduplicated_reason_text(reasons),
            optimization_context=None,
        )
    return _prepared_overrides_as_payloads(prepared), sorted(
        set(str(name) for name in skipped_wells)
    )
