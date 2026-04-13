from pywp.eclipse_welltrack import WelltrackRecord, WelltrackPoint
from pywp.models import TrajectoryConfig, Point3D
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL
from pywp.welltrack_batch import SuccessfulWellPlan
from pywp.pad_optimization import optimize_pad_order
from pywp.anticollision import AntiCollisionAnalysis, AntiCollisionZone
import pandas as pd

r1 = WelltrackRecord(
    name="WELL-A", points=(WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0), WelltrackPoint(x=100.0, y=100.0, z=1000.0, md=1000.0), WelltrackPoint(x=200.0, y=200.0, z=2000.0, md=2000.0))
)
r2 = WelltrackRecord(
    name="WELL-B", points=(WelltrackPoint(x=10.0, y=10.0, z=0.0, md=0.0), WelltrackPoint(x=110.0, y=110.0, z=1000.0, md=1000.0), WelltrackPoint(x=210.0, y=210.0, z=2000.0, md=2000.0))
)
r3 = WelltrackRecord(
    name="WELL-FAILED", points=(WelltrackPoint(x=20.0, y=20.0, z=0.0, md=0.0), WelltrackPoint(x=120.0, y=120.0, z=1000.0, md=1000.0), WelltrackPoint(x=220.0, y=220.0, z=2000.0, md=2000.0))
)

records = [r1, r2, r3]
config = TrajectoryConfig()
s1 = SuccessfulWellPlan(
    name="WELL-A", surface=Point3D(0.0, 0.0, 0.0), t1=Point3D(100.0, 100.0, 1000.0), t3=Point3D(200.0, 200.0, 2000.0),
    stations=pd.DataFrame({"MD_m": [0.0, 100.0], "INC_deg": [0.0, 0.0], "AZI_deg": [0.0, 0.0], "X_m": [0.0, 0.0], "Y_m": [0.0, 0.0], "Z_m": [0.0, 100.0]}),
    summary={}, azimuth_deg=45.0, md_t1_m=1000.0, config=config,
)
s2 = SuccessfulWellPlan(
    name="WELL-B", surface=Point3D(10.0, 10.0, 0.0), t1=Point3D(110.0, 110.0, 1000.0), t3=Point3D(210.0, 210.0, 2000.0),
    stations=pd.DataFrame({"MD_m": [0.0, 100.0], "INC_deg": [0.0, 0.0], "AZI_deg": [0.0, 0.0], "X_m": [10.0, 10.0], "Y_m": [10.0, 10.0], "Z_m": [0.0, 100.0]}),
    summary={}, azimuth_deg=45.0, md_t1_m=1000.0, config=config,
)
successes = {"WELL-A": s1, "WELL-B": s2}

# Force worst zone between A and B
class MockModel: pass
import pywp.pad_optimization as po
def mock_ac(*args, **kwargs):
    return AntiCollisionAnalysis(wells=(), corridors=(), well_segments=(), zones=[AntiCollisionZone(well_a="WELL-A", well_b="WELL-B", separation_factor=0.5, classification="severe", priority_rank=1, hotspot_xyz=(0,0,0), label_a="", label_b="", md_a_m=0.0, md_b_m=0.0, combined_radius_m=1.0, overlap_depth_m=1.0, display_radius_m=1.0, center_distance_m=1.0)], pair_count=1, overlapping_pair_count=1, target_overlap_pair_count=0, worst_separation_factor=0.5)

po.build_anti_collision_analysis_for_successes = mock_ac

try:
    po.optimize_pad_order(records, successes, {"WELL-A", "WELL-B", "WELL-FAILED"}, DEFAULT_PLANNING_UNCERTAINTY_MODEL, [], {"WELL-A": config, "WELL-B": config}, lambda *args: None)
    print("OK")
except Exception as e:
    import traceback
    traceback.print_exc()

