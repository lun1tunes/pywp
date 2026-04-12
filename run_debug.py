import pytest
from pywp.planner import TrajectoryPlanner
from pywp.models import TrajectoryConfig, OPTIMIZATION_MINIMIZE_MD, OPTIMIZATION_MINIMIZE_KOP
from pywp.eclipse_welltrack import parse_welltrack_text
from pywp.eclipse_welltrack import welltrack_points_to_targets
from pathlib import Path

planner = TrajectoryPlanner()
records = parse_welltrack_text(Path("tests/test_data/WELLTRACKS3.INC").read_text(encoding="utf-8"))
target_record = next(record for record in records if str(record.name) == "well_02")
surface, t1, t3 = welltrack_points_to_targets(target_record.points)

print("Planning MD")
try:
    result_md = planner.plan(surface=surface, t1=t1, t3=t3, config=TrajectoryConfig(optimization_mode=OPTIMIZATION_MINIMIZE_MD, turn_solver_max_restarts=2))
    print("MD success")
except Exception as e:
    print(f"MD failed: {e}")

print("Planning KOP")
try:
    result_kop = planner.plan(surface=surface, t1=t1, t3=t3, config=TrajectoryConfig(optimization_mode=OPTIMIZATION_MINIMIZE_KOP, turn_solver_max_restarts=2))
    print("KOP success")
except Exception as e:
    print(f"KOP failed: {e}")

