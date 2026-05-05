"""Tests for analytical_precheck module."""

from pywp.analytical_precheck import (
    AnalyticalPreCheckResult,
    analytical_precheck,
    format_precheck_summary_ru,
    _calculate_min_kop_for_t1_reach,
)
from pywp.models import Point3D, TrajectoryConfig


def test_precheck_feasible_trajectory() -> None:
    """Standard trajectory should be feasible."""
    surface = Point3D(x=0, y=0, z=0)
    t1 = Point3D(x=100, y=0, z=1000)  # 100m offset, 1000m TVD
    t3 = Point3D(x=500, y=0, z=1500)  # 500m offset, 1500m TVD
    
    config = TrajectoryConfig(
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=18.0,  # 6 deg/10m - generous limit
        entry_inc_target_deg=85.0,
        max_inc_deg=90.0,
    )
    
    result = analytical_precheck(surface=surface, t1=t1, t3=t3, config=config)
    
    assert result.is_feasible
    assert result.primary_issue == "none"
    assert result.required_pi_build_deg_per_10m < 15.0  # Physical
    assert result.is_pi_physical
    assert result.is_geometry_compatible
    assert result.kop_margin_m > 0


def test_precheck_kop_too_deep() -> None:
    """KOP deeper than t1 should be infeasible."""
    surface = Point3D(x=0, y=0, z=0)
    t1 = Point3D(x=100, y=0, z=500)
    t3 = Point3D(x=500, y=0, z=1000)
    
    config = TrajectoryConfig(
        kop_min_vertical_m=600.0,  # Deeper than t1!
    )
    
    result = analytical_precheck(surface=surface, t1=t1, t3=t3, config=config)
    
    assert not result.is_feasible
    assert result.primary_issue == "kop_too_deep"
    assert "Уменьшите параметр" in result.recommendation_ru


def test_precheck_unphysical_pi() -> None:
    """Very tight geometry should require unphysical PI."""
    surface = Point3D(x=0, y=0, z=0)
    # Extreme: 800m offset with very shallow t1 and deep KOP
    # Only 20m vertical for BUILD (600-580=20) with 800m horizontal!
    t1 = Point3D(x=800, y=0, z=600)
    t3 = Point3D(x=1200, y=0, z=1000)
    
    config = TrajectoryConfig(
        kop_min_vertical_m=580.0,  # Only 20m for BUILD!
        dls_build_max_deg_per_30m=3.0,  # 1 deg/10m
        entry_inc_target_deg=85.0,
    )
    
    result = analytical_precheck(surface=surface, t1=t1, t3=t3, config=config)
    
    assert not result.is_feasible
    assert result.primary_issue == "pi_insufficient"
    assert not result.is_pi_physical
    assert result.required_pi_build_deg_per_10m > 15.0
    assert "нефизично" in result.recommendation_ru.lower()


def test_precheck_t1_t3_infeasible() -> None:
    """Steep t1->t3 without enough max INC."""
    surface = Point3D(x=0, y=0, z=0)
    t1 = Point3D(x=100, y=0, z=1000)
    # t3 requires very steep angle
    t3 = Point3D(x=600, y=0, z=1100)  # 500m offset in only 100m depth
    
    config = TrajectoryConfig(
        max_inc_deg=60.0,  # Not enough for this geometry
        entry_inc_target_deg=55.0,  # Must be <= max_inc_deg
    )
    
    result = analytical_precheck(surface=surface, t1=t1, t3=t3, config=config)
    
    assert not result.is_feasible
    assert result.primary_issue == "t1_t3_infeasible"
    assert not result.is_geometry_compatible


def test_format_precheck_summary() -> None:
    """Summary formatting should work."""
    result = AnalyticalPreCheckResult(
        is_feasible=True,
        required_pi_build_deg_per_10m=2.5,
        min_kop_depth_m=200.0,
        kop_margin_m=800.0,
        hold_length_required_m=150.0,
        primary_issue="none",
        recommendation_ru="Траектория достижима.",
        is_pi_physical=True,
        is_geometry_compatible=True,
    )
    
    summary = format_precheck_summary_ru(result)
    assert "2.50" in summary
    assert "физично" in summary
    assert "✅" in summary


def test_min_kop_calculation() -> None:
    """Min KOP calculation should be reasonable."""
    surface = Point3D(x=0, y=0, z=0)
    t1 = Point3D(x=200, y=0, z=1000)
    
    # With low PI (gentle curve), need more vertical space
    min_kop_gentle = _calculate_min_kop_for_t1_reach(
        surface=surface,
        t1=t1,
        max_pi_deg_per_10m=1.0,  # Very gentle
        inc_entry_deg=85.0,
    )
    
    # With high PI (tight curve), can start deeper
    min_kop_tight = _calculate_min_kop_for_t1_reach(
        surface=surface,
        t1=t1,
        max_pi_deg_per_10m=5.0,  # Tighter
        inc_entry_deg=85.0,
    )
    
    # Tighter PI allows shallower KOP (deeper start)
    assert min_kop_tight > min_kop_gentle
