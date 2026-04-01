from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from pywp import TrajectoryConfig, TrajectoryPlanner
from pywp.eclipse_welltrack import parse_welltrack_text, welltrack_points_to_targets

BASE_SOURCE = Path("tests/test_data/WELLTRACKS3.INC")
DEFAULT_TARGET = Path("tests/test_data/WELLTRACKS_FACT_DEPTH_CLUSTERS.INC")
WELL_COUNT = 60
GRID_COLUMNS = 10
GRID_X_STEP_M = 700.0
GRID_Y_STEP_M = 820.0
OUTPUT_MD_STEP_M = 30.0
DEPTH_CLUSTER_CENTERS_TVD_M = (1650.0, 2550.0, 3450.0)
DEPTH_CLUSTER_SIZE = 20
DEPTH_OFFSETS_TVD_M = (
    -120.0,
    -90.0,
    -60.0,
    -30.0,
    0.0,
    30.0,
    60.0,
    90.0,
    120.0,
    -75.0,
    -45.0,
    -15.0,
    15.0,
    45.0,
    75.0,
    105.0,
    -105.0,
    55.0,
    -55.0,
    0.0,
)
HORIZONTAL_TAIL_DELTA_Z_PATTERN_M = (20.0, 40.0, 60.0, 80.0)


def _grid_offsets(count: int) -> list[tuple[float, float]]:
    rows = int(math.ceil(float(count) / float(GRID_COLUMNS)))
    x_positions = [float(col * GRID_X_STEP_M) for col in range(GRID_COLUMNS)]
    y_positions = [float(row * GRID_Y_STEP_M) for row in range(rows)]
    x_center = 0.5 * (x_positions[0] + x_positions[-1])
    y_center = 0.5 * (y_positions[0] + y_positions[-1])
    offsets: list[tuple[float, float]] = []
    for index in range(count):
        row = index // GRID_COLUMNS
        col = index % GRID_COLUMNS
        offsets.append(
            (
                float(x_positions[col] - x_center),
                float(y_positions[row] - y_center),
            )
        )
    return offsets


def _config() -> TrajectoryConfig:
    return TrajectoryConfig(
        optimization_mode="none",
        md_step_m=OUTPUT_MD_STEP_M,
        md_step_control_m=5.0,
        dls_build_max_deg_per_30m=6.0,
        max_total_md_postcheck_m=20000.0,
    )


def _cluster_depth_tvd(index: int) -> float:
    cluster_index = min(index // DEPTH_CLUSTER_SIZE, len(DEPTH_CLUSTER_CENTERS_TVD_M) - 1)
    depth_center = float(DEPTH_CLUSTER_CENTERS_TVD_M[cluster_index])
    local_index = index % DEPTH_CLUSTER_SIZE
    return float(depth_center + DEPTH_OFFSETS_TVD_M[local_index])


def _translated_targets(
    *,
    surface,
    t1,
    t3,
    dx_m: float,
    dy_m: float,
    horizontal_entry_tvd_m: float,
    tail_delta_z_m: float,
):
    return (
        surface.model_copy(
            update={"x": float(surface.x + dx_m), "y": float(surface.y + dy_m)}
        ),
        t1.model_copy(
            update={
                "x": float(t1.x + dx_m),
                "y": float(t1.y + dy_m),
                "z": float(horizontal_entry_tvd_m),
            }
        ),
        t3.model_copy(
            update={
                "x": float(t3.x + dx_m),
                "y": float(t3.y + dy_m),
                "z": float(horizontal_entry_tvd_m + tail_delta_z_m),
            }
        ),
    )


def _format_welltrack_text(wells: list[tuple[str, pd.DataFrame]]) -> str:
    blocks: list[str] = []
    for well_name, stations in wells:
        lines = [f"WELLTRACK '{well_name}'"]
        for row in stations[["X_m", "Y_m", "Z_m", "MD_m"]].itertuples(index=False):
            lines.append(
                f"{float(row.X_m):.3f} {float(row.Y_m):.3f} "
                f"{float(row.Z_m):.3f} {float(row.MD_m):.3f}"
            )
        lines.append("/")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def generate_actual_fund_depth_clusters(*, target: Path) -> None:
    base_records = parse_welltrack_text(BASE_SOURCE.read_text(encoding="utf-8"))
    templates = [welltrack_points_to_targets(record.points) for record in base_records]
    offsets = _grid_offsets(WELL_COUNT)
    planner = TrajectoryPlanner()
    config = _config()

    wells: list[tuple[str, pd.DataFrame]] = []
    for index in range(WELL_COUNT):
        surface, t1, t3 = templates[index % len(templates)]
        dx_m, dy_m = offsets[index]
        horizontal_entry_tvd_m = _cluster_depth_tvd(index)
        tail_delta_z_m = float(
            HORIZONTAL_TAIL_DELTA_Z_PATTERN_M[index % len(HORIZONTAL_TAIL_DELTA_Z_PATTERN_M)]
        )
        shifted_surface, shifted_t1, shifted_t3 = _translated_targets(
            surface=surface,
            t1=t1,
            t3=t3,
            dx_m=dx_m,
            dy_m=dy_m,
            horizontal_entry_tvd_m=horizontal_entry_tvd_m,
            tail_delta_z_m=tail_delta_z_m,
        )
        result = planner.plan(
            surface=shifted_surface,
            t1=shifted_t1,
            t3=shifted_t3,
            config=config,
        )
        wells.append((f"FACT_DEPTH_{index + 1:03d}", result.stations.copy()))

    target.write_text(_format_welltrack_text(wells), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clustered actual-fund WELLTRACK data across several depth bands."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Output path (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args()
    generate_actual_fund_depth_clusters(target=Path(args.target))


if __name__ == "__main__":
    main()
