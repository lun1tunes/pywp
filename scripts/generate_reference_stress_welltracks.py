from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from pywp import TrajectoryConfig, TrajectoryPlanner
from pywp.eclipse_welltrack import parse_welltrack_text, welltrack_points_to_targets

BASE_SOURCE = Path("tests/test_data/WELLTRACKS3.INC")
DEFAULT_FACT_TARGET = Path("tests/test_data/WELLTRACKS_FACT.INC")
DEFAULT_PROJECT_TARGET = Path("tests/test_data/WELLTRACKS_PROJECT.INC")

FACT_COUNT = 130
PROJECT_COUNT = 90
GRID_COLUMNS = 10
X_STEP_PATTERN_M = (1050.0, 1500.0, 1950.0, 2400.0, 3000.0, 3600.0)
Y_STEP_PATTERN_M = (1200.0, 1650.0, 2100.0, 2550.0, 3150.0)
PROJECT_GLOBAL_OFFSET_X_M = 32000.0
PROJECT_GLOBAL_OFFSET_Y_M = 6000.0
OUTPUT_MD_STEP_M = 30.0


def _cumulative_positions(count: int, steps: tuple[float, ...]) -> list[float]:
    values = [0.0]
    while len(values) < count:
        previous = values[-1]
        step = float(steps[(len(values) - 1) % len(steps)])
        values.append(previous + step)
    return values


def _centered_grid_offsets(
    *,
    count: int,
    columns: int,
    global_x_offset_m: float = 0.0,
    global_y_offset_m: float = 0.0,
) -> list[tuple[float, float]]:
    rows = int(math.ceil(float(count) / float(columns)))
    x_positions = _cumulative_positions(columns, X_STEP_PATTERN_M)
    y_positions = _cumulative_positions(rows, Y_STEP_PATTERN_M)
    x_center = 0.5 * (x_positions[0] + x_positions[-1])
    y_center = 0.5 * (y_positions[0] + y_positions[-1])
    offsets: list[tuple[float, float]] = []
    for index in range(count):
        row = index // columns
        col = index % columns
        offsets.append(
            (
                float(x_positions[col] - x_center + global_x_offset_m),
                float(y_positions[row] - y_center + global_y_offset_m),
            )
        )
    return offsets


def _stress_config() -> TrajectoryConfig:
    return TrajectoryConfig(
        optimization_mode="none",
        md_step_m=OUTPUT_MD_STEP_M,
        md_step_control_m=5.0,
        max_total_md_postcheck_m=20000.0,
    )


def _translate_targets(surface, t1, t3, dx_m: float, dy_m: float):
    return (
        surface.model_copy(
            update={"x": float(surface.x + dx_m), "y": float(surface.y + dy_m)}
        ),
        t1.model_copy(update={"x": float(t1.x + dx_m), "y": float(t1.y + dy_m)}),
        t3.model_copy(update={"x": float(t3.x + dx_m), "y": float(t3.y + dy_m)}),
    )


def _plan_translated_wells(
    *,
    prefix: str,
    count: int,
    global_x_offset_m: float = 0.0,
    global_y_offset_m: float = 0.0,
) -> list[tuple[str, pd.DataFrame]]:
    base_records = parse_welltrack_text(BASE_SOURCE.read_text(encoding="utf-8"))
    templates = [welltrack_points_to_targets(record.points) for record in base_records]
    offsets = _centered_grid_offsets(
        count=count,
        columns=GRID_COLUMNS,
        global_x_offset_m=global_x_offset_m,
        global_y_offset_m=global_y_offset_m,
    )
    planner = TrajectoryPlanner()
    config = _stress_config()
    planned: list[tuple[str, pd.DataFrame]] = []
    for index in range(count):
        surface, t1, t3 = templates[index % len(templates)]
        dx_m, dy_m = offsets[index]
        shifted_surface, shifted_t1, shifted_t3 = _translate_targets(
            surface,
            t1,
            t3,
            dx_m,
            dy_m,
        )
        result = planner.plan(
            surface=shifted_surface,
            t1=shifted_t1,
            t3=shifted_t3,
            config=config,
        )
        planned.append((f"{prefix}_{index + 1:03d}", result.stations.copy()))
    return planned


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


def generate_stress_welltracks(
    *,
    fact_target: Path,
    project_target: Path,
) -> None:
    fact_wells = _plan_translated_wells(prefix="FACT", count=FACT_COUNT)
    project_wells = _plan_translated_wells(
        prefix="PROJECT",
        count=PROJECT_COUNT,
        global_x_offset_m=PROJECT_GLOBAL_OFFSET_X_M,
        global_y_offset_m=PROJECT_GLOBAL_OFFSET_Y_M,
    )
    fact_target.write_text(_format_welltrack_text(fact_wells), encoding="utf-8")
    project_target.write_text(_format_welltrack_text(project_wells), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate large reference WELLTRACK datasets for custom 3D "
            "performance testing."
        )
    )
    parser.add_argument(
        "--fact-target",
        type=Path,
        default=DEFAULT_FACT_TARGET,
        help=f"Output path for factual trajectories (default: {DEFAULT_FACT_TARGET})",
    )
    parser.add_argument(
        "--project-target",
        type=Path,
        default=DEFAULT_PROJECT_TARGET,
        help=(
            "Output path for approved project trajectories "
            f"(default: {DEFAULT_PROJECT_TARGET})"
        ),
    )
    args = parser.parse_args()
    generate_stress_welltracks(
        fact_target=Path(args.fact_target),
        project_target=Path(args.project_target),
    )


if __name__ == "__main__":
    main()
