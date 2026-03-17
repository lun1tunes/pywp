from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from pywp.anticollision import (
    AntiCollisionAnalysis,
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
    build_anti_collision_recommendation_clusters,
    build_anti_collision_recommendations,
)
from pywp.models import OPTIMIZATION_ANTI_COLLISION_AVOIDANCE, OPTIMIZATION_MINIMIZE_KOP
from pywp.uncertainty import PlanningUncertaintyModel

if TYPE_CHECKING:
    from pywp.welltrack_batch import SuccessfulWellPlan


DYNAMIC_CLUSTER_PLAN_ACTIVE = "active"
DYNAMIC_CLUSTER_PLAN_RESOLVED = "resolved"
DYNAMIC_CLUSTER_PLAN_BLOCKED = "blocked"


@dataclass(frozen=True)
class DynamicClusterExecutionPlan:
    cluster: AntiCollisionRecommendationCluster | None
    ordered_well_names: tuple[str, ...]
    prepared_by_well: dict[str, dict[str, object]]
    skipped_wells: tuple[str, ...]
    resolution_state: str = DYNAMIC_CLUSTER_PLAN_ACTIVE
    blocking_reason: str | None = None


def build_anti_collision_analysis_for_successes(
    successes: list[SuccessfulWellPlan],
    *,
    model: PlanningUncertaintyModel,
    name_to_color: Mapping[str, str] | None = None,
    include_display_geometry: bool = True,
    build_overlap_geometry: bool = True,
) -> AntiCollisionAnalysis:
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
        )
        for item in successes
    ]
    return analyze_anti_collision(
        wells,
        build_overlap_geometry=build_overlap_geometry,
    )


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
            md_t1_m=float(success.md_t1_m),
            md_total_m=float(summary.get("md_total_m", 0.0)),
            optimization_mode=str(success.config.optimization_mode),
        )
    return contexts


def build_dynamic_cluster_execution_plan(
    *,
    successes: list[SuccessfulWellPlan],
    selected_names: set[str],
    target_well_names: tuple[str, ...],
    uncertainty_model: PlanningUncertaintyModel,
) -> DynamicClusterExecutionPlan | None:
    if not selected_names:
        return None
    if len(successes) < 2:
        return DynamicClusterExecutionPlan(
            cluster=None,
            ordered_well_names=(),
            prepared_by_well={},
            skipped_wells=(),
            resolution_state=DYNAMIC_CLUSTER_PLAN_RESOLVED,
        )
    analysis = build_anti_collision_analysis_for_successes(
        successes,
        model=uncertainty_model,
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
    ordered_well_names = tuple(
        str(step.well_name)
        for step in cluster.action_steps
        if str(step.well_name) in selected_names and str(step.well_name) in prepared
    )
    fallback_names = tuple(
        sorted(
            set(str(name) for name in prepared.keys())
            .intersection(selected_names)
            .difference(ordered_well_names)
        )
    )
    final_order = tuple([*ordered_well_names, *fallback_names])
    if not final_order:
        return None
    return DynamicClusterExecutionPlan(
        cluster=cluster,
        ordered_well_names=final_order,
        prepared_by_well=prepared,
        skipped_wells=tuple(sorted(set(str(name) for name in skipped_wells))),
        resolution_state=DYNAMIC_CLUSTER_PLAN_ACTIVE,
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
) -> AntiCollisionOptimizationContext | None:
    if moving_success is None or reference_success is None:
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
    reference_path = build_anti_collision_reference_path(
        well_name=str(reference_success.name),
        stations=reference_success.stations,
        md_start_m=reference_md_start_m,
        md_end_m=reference_md_end_m,
        sample_step_m=50.0,
        model=uncertainty_model,
    )
    return AntiCollisionOptimizationContext(
        candidate_md_start_m=float(candidate_md_start_m),
        candidate_md_end_m=float(candidate_md_end_m),
        sf_target=1.0,
        sample_step_m=50.0,
        uncertainty_model=uncertainty_model,
        references=(reference_path,),
    )


def build_cluster_prepared_overrides(
    cluster: AntiCollisionRecommendationCluster,
    *,
    successes: list[SuccessfulWellPlan],
    uncertainty_model: PlanningUncertaintyModel,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    success_by_name = {str(item.name): item for item in successes}
    trajectory_specs: dict[str, dict[str, object]] = {}
    vertical_specs: dict[str, list[str]] = {}
    skipped_wells: list[str] = []

    for recommendation in cluster.recommendations:
        for suggestion in recommendation.override_suggestions:
            well_name = str(suggestion.well_name)
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
                if reference_success is None:
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
                continue

            if optimization_mode == OPTIMIZATION_MINIMIZE_KOP:
                vertical_specs.setdefault(well_name, []).append(str(suggestion.reason))
                entry = trajectory_specs.get(well_name)
                if entry is not None:
                    entry["prefer_lower_kop"] = True

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
            references: list[object] = []
            for reference_name, window in sorted(
                dict(spec.get("reference_windows", {})).items()
            ):
                reference_success = success_by_name.get(str(reference_name))
                if reference_success is None:
                    skipped_wells.append(well_name)
                    references = []
                    break
                references.append(
                    build_anti_collision_reference_path(
                        well_name=str(reference_success.name),
                        stations=reference_success.stations,
                        md_start_m=float(window[0]),
                        md_end_m=float(window[1]),
                        sample_step_m=50.0,
                        model=uncertainty_model,
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
                sample_step_m=50.0,
                uncertainty_model=uncertainty_model,
                references=tuple(references),
                prefer_lower_kop=bool(spec.get("prefer_lower_kop")) or bool(vertical_specs.get(well_name)),
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
