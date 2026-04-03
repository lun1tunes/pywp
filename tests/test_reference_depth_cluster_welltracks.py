from __future__ import annotations

from pathlib import Path

from pywp.actual_fund_analysis import (
    build_actual_fund_kop_depth_function,
    build_actual_fund_well_metrics,
    summarize_actual_fund_by_depth,
)
from pywp.reference_trajectories import parse_reference_trajectory_welltrack_text


def test_generated_actual_fund_depth_cluster_welltracks_are_clusterable() -> None:
    wells = parse_reference_trajectory_welltrack_text(
        Path("tests/test_data/WELLTRACKS_FACT_DEPTH_CLUSTERS.INC").read_text(encoding="utf-8"),
        kind="actual",
    )

    metrics = build_actual_fund_well_metrics(wells)
    clusters = summarize_actual_fund_by_depth(metrics)
    kop_function = build_actual_fund_kop_depth_function(metrics)

    assert len(wells) == 60
    assert len(clusters) == 3
    assert min(cluster.well_count for cluster in clusters) >= 18
    assert kop_function is not None
    assert kop_function.mode == "piecewise_linear"
    assert tuple(round(value, 1) for value in kop_function.anchor_kop_md_m) == (
        round(float(clusters[0].anchor_kop_md_m), 1),
        round(float(clusters[1].anchor_kop_md_m), 1),
        round(float(clusters[2].anchor_kop_md_m), 1),
    )
