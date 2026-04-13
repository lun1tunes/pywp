import pytest
from pywp.eclipse_welltrack import WelltrackRecord, WelltrackPoint
from pywp.models import TrajectoryConfig, Point3D
from pywp.uncertainty import DEFAULT_PLANNING_UNCERTAINTY_MODEL
from pywp.welltrack_batch import SuccessfulWellPlan
from pywp.pad_optimization import optimize_pad_order
from pywp.anticollision import AntiCollisionAnalysis, AntiCollisionZone
import pandas as pd

def test_optimize_pad_order_no_records():
    records = []
    successes = {}
    model = DEFAULT_PLANNING_UNCERTAINTY_MODEL
    def cb(pct, msg): pass
    res_rec, res_suc, improved = optimize_pad_order(records, successes, set(), model, [], {}, cb)
    assert not improved

