"""Fast analytical pre-check for trajectory feasibility.

This module provides millisecond-fast analytical calculations to diagnose
trajectory feasibility BEFORE running the full numerical solver.

Key capabilities:
- Minimum required PI (DLS) for pure BUILD arc
- Minimum KOP depth for given geometry
- Geometric compatibility checks for t1→t3
- Quick feasibility verdict with specific recommendations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from pywp.constants import DEG2RAD, RAD2DEG, SMALL
from pywp.models import Point3D, TrajectoryConfig
from pywp.planner_geometry import (
    _azimuth_deg_from_pair,
    _azimuth_deg_from_points,
    _build_section_geometry,
    _dls_from_radius,
    _horizontal_offset,
    _inclination_from_displacement,
    _project_to_section_axis,
    _required_dls_for_t1_reach,
)


@dataclass(frozen=True)
class AnalyticalPreCheckResult:
    """Result of fast analytical pre-check."""

    is_feasible: bool
    required_pi_build_deg_per_10m: float
    min_kop_depth_m: float
    kop_margin_m: float  # available_vertical - kop_min
    hold_length_required_m: float  # linear distance from KOP to t1
    
    # Diagnostic messages
    primary_issue: str  # "none", "kop_too_deep", "pi_insufficient", "t1_t3_infeasible"
    recommendation_ru: str
    
    # Physical limits check
    is_pi_physical: bool  # < 15 deg/10m is considered physically achievable
    is_geometry_compatible: bool  # t1→t3 can be connected without overbend


def _calculate_min_kop_for_t1_reach(
    surface: Point3D,
    t1: Point3D,
    max_pi_deg_per_10m: float,
    inc_entry_deg: float,
) -> float:
    """Calculate minimum KOP depth required to reach t1 with given max PI.
    
    Uses pure circular arc geometry: given max curvature (PI), 
    what's the minimum vertical span needed?
    
    Returns minimum KOP TVD (vertical depth from surface).
    """
    if max_pi_deg_per_10m <= SMALL:
        return float("inf")
    
    # Convert PI to radius
    # PI [deg/10m] -> DLS [deg/30m] = PI * 3
    dls_deg_per_30m = max_pi_deg_per_10m * 3.0
    radius_m = 30.0 * RAD2DEG / dls_deg_per_30m if dls_deg_per_30m > SMALL else float("inf")
    
    # Horizontal offset to t1
    horizontal_m = _horizontal_offset(surface, t1)
    t1_tvd_m = t1.z - surface.z
    
    if radius_m <= SMALL or horizontal_m <= SMALL:
        return 0.0
    
    # For a circular arc from vertical to inc_entry at t1:
    # Vertical span needed = radius * sin(inc_entry)
    # Horizontal span needed = radius * (1 - cos(inc_entry))
    inc_rad = inc_entry_deg * DEG2RAD
    
    # Check if arc can reach the horizontal offset at inc_entry
    max_horizontal_from_arc = radius_m * (1.0 - np.cos(inc_rad))
    max_vertical_from_arc = radius_m * np.sin(inc_rad)
    
    if max_horizontal_from_arc < horizontal_m - SMALL:
        # Arc alone cannot reach t1 horizontally at target inc.
        # Need arc+HOLD: arc to inc_entry, then hold to cover remaining horizontal
        # This requires the arc to reach inc_entry fully, then hold section
        hold_horizontal = horizontal_m - max_horizontal_from_arc
        hold_vertical = hold_horizontal / np.tan(inc_rad) if np.tan(inc_rad) > SMALL else float("inf")
        total_vertical = max_vertical_from_arc + hold_vertical
        if not np.isfinite(total_vertical):
            return float("inf")
        min_kop_depth = t1_tvd_m - total_vertical
        return max(0.0, min_kop_depth)
    
    # Pure arc is sufficient (horizontal offset <= max arc horizontal)
    # Solve for actual inc needed to reach horizontal_m
    # horizontal = radius * (1 - cos(inc)) => cos(inc) = 1 - horizontal/radius
    cos_inc = 1.0 - horizontal_m / radius_m
    if cos_inc < -1.0 or cos_inc > 1.0:
        return float("inf")
    actual_inc = np.arccos(cos_inc)
    total_vertical = radius_m * np.sin(actual_inc)
    
    min_kop_depth = t1_tvd_m - total_vertical
    return max(0.0, min_kop_depth)


def analytical_precheck(
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    config: TrajectoryConfig,
) -> AnalyticalPreCheckResult:
    """Perform fast analytical pre-check of trajectory feasibility.
    
    This runs in O(1) time and provides immediate feedback on:
    - Whether current config can theoretically reach targets
    - Minimum KOP depth required
    - Required PI for pure BUILD
    - Specific recommendations
    
    Returns AnalyticalPreCheckResult with diagnostic information.
    """
    
    # Extract basic geometry
    t1_tvd_m = t1.z - surface.z
    t1_horizontal_m = _horizontal_offset(surface, t1)
    kop_min_m = float(config.kop_min_vertical_m)
    max_pi = float(config.dls_build_max_deg_per_30m) / 3.0  # DLS->PI
    inc_entry_deg = float(config.entry_inc_target_deg)
    
    # Calculate available BUILD vertical
    build_vertical_available_m = max(t1_tvd_m - kop_min_m, 0.0)
    
    # Check 1: KOP depth feasibility
    if kop_min_m >= t1_tvd_m:
        return AnalyticalPreCheckResult(
            is_feasible=False,
            required_pi_build_deg_per_10m=float("inf"),
            min_kop_depth_m=kop_min_m,
            kop_margin_m=0.0,
            hold_length_required_m=float("nan"),
            primary_issue="kop_too_deep",
            recommendation_ru=(
                f"Мин VERTICAL до KOP ({kop_min_m:.1f} м) больше глубины t1 ({t1_tvd_m:.1f} м). "
                f"Уменьшите параметр до максимум {t1_tvd_m*0.75:.1f} м."
            ),
            is_pi_physical=False,
            is_geometry_compatible=False,
        )
    
    # Check 2: Required PI for pure BUILD arc
    # This assumes a pure circular arc from KOP to t1
    if build_vertical_available_m > SMALL and t1_horizontal_m > SMALL:
        required_pi = _required_dls_for_t1_reach(
            s1_m=t1_horizontal_m,
            z1_m=build_vertical_available_m,
            inc_entry_deg=inc_entry_deg,
        ) / 3.0  # DLS->PI
        required_pi = required_pi if np.isfinite(required_pi) else float("inf")
    else:
        required_pi = float("inf")
    
    # Check 3: t1->t3 geometry compatibility
    try:
        geometry = _build_section_geometry(surface=surface, t1=t1, t3=t3, config=config)
        inc_required_t1_t3 = geometry.inc_required_t1_t3_deg
        is_t1_t3_compatible = inc_required_t1_t3 <= config.max_inc_deg + SMALL
    except Exception:
        inc_required_t1_t3 = float("inf")
        is_t1_t3_compatible = False
    
    # Calculate minimum KOP depth for given max PI
    min_kop_for_max_pi = _calculate_min_kop_for_t1_reach(
        surface=surface,
        t1=t1,
        max_pi_deg_per_10m=max_pi,
        inc_entry_deg=inc_entry_deg,
    )
    
    # Calculate hold length (straight section from KOP end to t1)
    if np.isfinite(required_pi) and required_pi > SMALL:
        # Arc from KOP reaches certain inc, then hold to t1
        dls = required_pi * 3.0  # PI->DLS
        radius = 30.0 * RAD2DEG / dls if dls > SMALL else float("inf")
        inc_rad = inc_entry_deg * DEG2RAD
        
        # Horizontal from pure arc
        horizontal_from_arc = radius * (1.0 - np.cos(inc_rad))
        
        if horizontal_from_arc < t1_horizontal_m:
            hold_horizontal = t1_horizontal_m - horizontal_from_arc
            hold_length = hold_horizontal / np.sin(inc_rad) if np.sin(inc_rad) > SMALL else float("inf")
        else:
            hold_length = 0.0
    else:
        hold_length = float("nan")
    
    # Determine feasibility and primary issue
    is_pi_sufficient = required_pi <= max_pi + SMALL
    is_pi_physical = required_pi <= 15.0  # >15 deg/10m is unphysical
    
    if not is_t1_t3_compatible:
        primary_issue = "t1_t3_infeasible"
        is_feasible = False
        recommendation = (
            f"Геометрия t1→t3 несовместима: требуется INC {inc_required_t1_t3:.1f}° "
            f"при max {config.max_inc_deg}°. "
            f"Углубите t3 и/или сократите горизонтальное смещение t1→t3."
        )
    elif not is_pi_sufficient:
        primary_issue = "pi_insufficient"
        is_feasible = False
        if not is_pi_physical:
            recommendation = (
                f"Требуется нефизично высокий ПИ (>15°/10м). "
                f"Уменьшите Мин VERTICAL до KOP (сейчас {kop_min_m:.1f} м) "
                f"или увеличьте max ПИ BUILD."
            )
        else:
            recommendation = (
                f"Требуемый ПИ ({required_pi:.2f}°/10м) выше max ({max_pi:.2f}°/10м). "
                f"Увеличьте max ПИ BUILD или уменьшите Мин VERTICAL до KOP."
            )
    elif kop_min_m > min_kop_for_max_pi + 10.0:  # 10m tolerance
        # KOP is deeper than necessary - will work but suboptimal
        primary_issue = "kop_suboptimal"
        is_feasible = True
        recommendation = (
            f"Траектория достижима, но Мин VERTICAL до KOP ({kop_min_m:.1f} м) "
            f"глубже оптимума ({min_kop_for_max_pi:.1f} м). "
            f"Рекомендуется уменьшить для более мягкого профиля."
        )
    else:
        primary_issue = "none"
        is_feasible = True
        recommendation = "Геометрия совместима с текущими ограничениями."
    
    return AnalyticalPreCheckResult(
        is_feasible=is_feasible,
        required_pi_build_deg_per_10m=round(required_pi, 2) if np.isfinite(required_pi) else 999.99,
        min_kop_depth_m=round(min_kop_for_max_pi, 1) if np.isfinite(min_kop_for_max_pi) else 9999.0,
        kop_margin_m=round(build_vertical_available_m, 1),
        hold_length_required_m=round(hold_length, 1) if np.isfinite(hold_length) else 0.0,
        primary_issue=primary_issue,
        recommendation_ru=recommendation,
        is_pi_physical=is_pi_physical,
        is_geometry_compatible=is_t1_t3_compatible,
    )


def format_precheck_summary_ru(result: AnalyticalPreCheckResult) -> str:
    """Format pre-check result as human-readable Russian summary."""
    lines = [
        f"**Аналитический анализ (мгновенный):**",
        f"",
        f"- **Требуемый ПИ:** {result.required_pi_build_deg_per_10m:.2f}°/10м "
        f"({'физично' if result.is_pi_physical else 'НЕФИЗИЧНО'})",
        f"- **Минимальный KOP:** {result.min_kop_depth_m:.1f} м (доступно: {result.kop_margin_m:.1f} м)",
        f"- **Длина HOLD:** {result.hold_length_required_m:.1f} м",
        f"- **Статус:** {'✅ Достижимо' if result.is_feasible else '❌ Недостижимо'}",
        f"",
        f"**Рекомендация:** {result.recommendation_ru}",
    ]
    return "\n".join(lines)
