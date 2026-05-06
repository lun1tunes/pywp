"""Integration of coordinate systems with trajectory planning UI.

Provides sidebar controls and coordinate transformation for well trajectory results.
Default CRS: PNO16 (as per user requirements).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from pywp.coordinate_systems import (
    CoordinateSystem,
    CoordinateSystemError,
    CoordinateTransformer,
    GeodeticCoord,
    HAS_PYPROJ,
    LocalCoordinateSystem,
    LocalCoordinateTransformer,
    ProjectedCoord,
    disambiguate_pno16,
    define_pno_16_system,
    define_pno_13_system,
)
from pywp.models import Point3D

if TYPE_CHECKING:
    from pywp.ui_well_result import SingleWellResultView

logger = logging.getLogger(__name__)

# Session state keys
CRS_SELECTED_KEY = "trajectory_crs_selected"
CRS_SELECTBOX_KEY = "trajectory_crs_selectbox"  # Separate key for widget
CRS_AUTO_CONVERT_KEY = "trajectory_crs_auto_convert"

# Default CRS as per user requirement
DEFAULT_CRS = CoordinateSystem.PNO_16_ZONE

# Available CRS options for the UI
CRS_OPTIONS: list[tuple[str, CoordinateSystem]] = [
    ("PNO-16 (Зона)", CoordinateSystem.PNO_16_ZONE),
    ("PNO-16 (CM)", CoordinateSystem.PNO_16_CM),
    ("PNO-13 (Зона)", CoordinateSystem.PNO_13_ZONE),
    ("PNO-13 (CM)", CoordinateSystem.PNO_13_CM),
    ("СК-42 (Градусные)", CoordinateSystem.PULKOVO_1942),
    ("СК-42 Зона 6", CoordinateSystem.PULKOVO_1942_ZONE_6),
    ("СК-42 Зона 7", CoordinateSystem.PULKOVO_1942_ZONE_7),
    ("СК-42 Зона 8", CoordinateSystem.PULKOVO_1942_ZONE_8),
    ("СК-42 Зона 9", CoordinateSystem.PULKOVO_1942_ZONE_9),
    ("СК-42 Зона 10", CoordinateSystem.PULKOVO_1942_ZONE_10),
    ("СК-42 Зона 11", CoordinateSystem.PULKOVO_1942_ZONE_11),
    ("СК-42 Зона 12", CoordinateSystem.PULKOVO_1942_ZONE_12),
    ("СК-42 Зона 13", CoordinateSystem.PULKOVO_1942_ZONE_13),
    ("СК-42 Зона 14", CoordinateSystem.PULKOVO_1942_ZONE_14),
    ("СК-42 Зона 15", CoordinateSystem.PULKOVO_1942_ZONE_15),
    ("СК-42 Зона 16", CoordinateSystem.PULKOVO_1942_ZONE_16),
    ("СК-42 Зона 17", CoordinateSystem.PULKOVO_1942_ZONE_17),
    ("СК-42 Зона 18", CoordinateSystem.PULKOVO_1942_ZONE_18),
    ("СК-42 Зона 19", CoordinateSystem.PULKOVO_1942_ZONE_19),
    ("СК-42 Зона 20", CoordinateSystem.PULKOVO_1942_ZONE_20),
    ("Пулково 1995 (Градусные)", CoordinateSystem.PULKOVO_1995),
    ("Пулково 1995 Зона 13", CoordinateSystem.PULKOVO_1995_ZONE_13),
    ("Пулково 1995 Зона 18", CoordinateSystem.PULKOVO_1995_ZONE_18),
    ("ГСК-2011", CoordinateSystem.GSK_2011),
    ("WGS84", CoordinateSystem.WGS84),
]

CRS_LABEL_BY_VALUE: dict[CoordinateSystem, str] = {
    crs: label for label, crs in CRS_OPTIONS
}

# Pre-compute default index to avoid hardcoded magic number
_DEFAULT_CRS_INDEX = next(
    (i for i, (_, c) in enumerate(CRS_OPTIONS) if c == DEFAULT_CRS),
    0,
)


def _get_crs_index(crs: CoordinateSystem) -> int:
    """Get index of CRS in options list."""
    for i, (_, c) in enumerate(CRS_OPTIONS):
        if c == crs:
            return i
    return _DEFAULT_CRS_INDEX


def render_crs_sidebar() -> CoordinateSystem:
    """Render coordinate system selection in sidebar.

    Returns:
        Selected coordinate system
    """
    with st.sidebar:
        st.divider()
        st.markdown("### Система координат")

        # Initialize with default if not set
        if CRS_SELECTED_KEY not in st.session_state:
            st.session_state[CRS_SELECTED_KEY] = DEFAULT_CRS

        # Get current selection label from stored enum
        current_crs = st.session_state.get(CRS_SELECTED_KEY, DEFAULT_CRS)
        current_label = next(
            (label for label, crs in CRS_OPTIONS if crs == current_crs),
            CRS_OPTIONS[_DEFAULT_CRS_INDEX][0],  # Default fallback label
        )

        # Pre-set widget state as string label (not enum)
        if CRS_SELECTBOX_KEY not in st.session_state:
            st.session_state[CRS_SELECTBOX_KEY] = current_label

        selected_label = st.selectbox(
            "Выберите CRS",
            options=[label for label, _ in CRS_OPTIONS],
            index=_get_crs_index(current_crs),
            key=CRS_SELECTBOX_KEY,
            help=(
                "Система координат применяется только к CSV-выгрузкам "
                "инклинометрии. Экранные точки, кусты, графики и таблицы "
                "остаются в расчётной системе. По умолчанию: PNO-16 (Зона)."
            ),
        )

        # Find the CRS enum for selected label and store separately
        selected_crs = next(
            (crs for label, crs in CRS_OPTIONS if label == selected_label),
            DEFAULT_CRS,  # Default fallback
        )
        st.session_state[CRS_SELECTED_KEY] = selected_crs

        # Auto-convert toggle
        if CRS_AUTO_CONVERT_KEY not in st.session_state:
            st.session_state[CRS_AUTO_CONVERT_KEY] = True

        st.toggle(
            "Пересчитывать координаты в CSV",
            key=CRS_AUTO_CONVERT_KEY,
            help="Автоматически пересчитывать только CSV инклинометрии при смене CRS",
        )

        # Show selected CRS info
        st.caption(f"**Текущая:** {CRS_LABEL_BY_VALUE.get(selected_crs, selected_label)}")

        return selected_crs


def get_selected_crs() -> CoordinateSystem:
    """Get currently selected coordinate system from session state.

    Returns:
        Selected coordinate system (defaults to PNO_16_ZONE)
    """
    return st.session_state.get(CRS_SELECTED_KEY, DEFAULT_CRS)


def should_auto_convert() -> bool:
    """Check if auto-convert is enabled."""
    return st.session_state.get(CRS_AUTO_CONVERT_KEY, True)


def _try_create_transformer() -> CoordinateTransformer | None:
    """Create CoordinateTransformer if pyproj is available.

    Returns:
        CoordinateTransformer instance or None if pyproj unavailable.
    """
    if not HAS_PYPROJ:
        return None
    try:
        return CoordinateTransformer()
    except CoordinateSystemError:
        return None


def _can_transform_directly(from_crs: CoordinateSystem, to_crs: CoordinateSystem) -> bool:
    """Check if direct pyproj transformation is available.

    Direct EPSG-to-EPSG transforms work when both systems have
    EPSG codes (not PNO/LOCAL placeholders).
    """
    if from_crs == to_crs:
        return True

    effective_from = _effective_pyproj_crs(from_crs)
    effective_to = _effective_pyproj_crs(to_crs)
    if effective_from is None or effective_to is None:
        return False
    if effective_from == effective_to:
        return True

    from_val = effective_from.value
    to_val = effective_to.value
    return (
        from_val.startswith("EPSG:")
        and to_val.startswith("EPSG:")
    )


def _effective_pyproj_crs(crs: CoordinateSystem) -> CoordinateSystem | None:
    """Return the EPSG-backed CRS used for transformation.

    PNO systems are labels over known base projected systems in this app. The
    field-specific false easting/rotation parameters still need project setup,
    but the zero-offset base CRS allows the UI to consistently transform display
    coordinates instead of silently leaving every view unchanged.
    """
    if crs == CoordinateSystem.LOCAL:
        return None
    if crs == CoordinateSystem.PNO_16_ZONE:
        return CoordinateSystem.PULKOVO_1942_ZONE_16
    if crs == CoordinateSystem.PNO_16_CM:
        return CoordinateSystem.PULKOVO_1995_CM_39E
    if crs == CoordinateSystem.PNO_13_ZONE:
        return CoordinateSystem.PULKOVO_1942_ZONE_13
    if crs == CoordinateSystem.PNO_13_CM:
        return CoordinateSystem.PULKOVO_1995_CM_39E
    return crs


def _transform_xy(
    x: float,
    y: float,
    from_crs: CoordinateSystem,
    to_crs: CoordinateSystem,
) -> tuple[float, float]:
    """Transform horizontal coordinates (x, y) between CRSs.

    Returns:
        (transformed_x, transformed_y) or original if transformation fails.
    """
    if from_crs == to_crs:
        return x, y

    transformer = _try_create_transformer()
    if transformer is None:
        return x, y

    effective_from = _effective_pyproj_crs(from_crs)
    effective_to = _effective_pyproj_crs(to_crs)
    if effective_from is None or effective_to is None:
        return x, y
    if effective_from == effective_to:
        return x, y

    if not _can_transform_directly(from_crs, to_crs):
        return x, y

    try:
        if effective_from.is_geographic() and effective_to.is_geographic():
            # Both geographic: lon/lat -> lon/lat
            result = transformer.transform(
                GeodeticCoord(lat_deg=y, lon_deg=x),
                effective_from, effective_to,
            )
            if isinstance(result, GeodeticCoord):
                return result.lon_deg, result.lat_deg
        elif not effective_from.is_geographic() and not effective_to.is_geographic():
            # Both projected: easting/northing -> easting/northing
            result = transformer.transform(
                ProjectedCoord(easting_m=x, northing_m=y),
                effective_from, effective_to,
            )
            if isinstance(result, ProjectedCoord):
                return result.easting_m, result.northing_m
        elif effective_from.is_geographic() and not effective_to.is_geographic():
            # Geographic -> projected
            result = transformer.transform(
                GeodeticCoord(lat_deg=y, lon_deg=x),
                effective_from, effective_to,
            )
            if isinstance(result, ProjectedCoord):
                return result.easting_m, result.northing_m
        else:
            # Projected -> geographic
            result = transformer.transform(
                ProjectedCoord(easting_m=x, northing_m=y),
                effective_from, effective_to,
            )
            if isinstance(result, GeodeticCoord):
                return result.lon_deg, result.lat_deg
    except Exception as exc:
        logger.warning(
            f"Coordinate transformation failed ({from_crs.name} -> {to_crs.name}): {exc}"
        )

    return x, y


def transform_point_to_crs(
    point: Point3D,
    from_crs: CoordinateSystem,
    to_crs: CoordinateSystem,
) -> Point3D:
    """Transform a 3D point between coordinate systems.

    Transforms horizontal coordinates (x, y) while preserving z (vertical/TVD).
    Falls back to original coordinates if pyproj is unavailable or
    the transformation path is unsupported.

    Args:
        point: Point with x, y, z coordinates
        from_crs: Source coordinate system
        to_crs: Target coordinate system

    Returns:
        Transformed point (z is always preserved).
    """
    if from_crs == to_crs:
        return point

    tx, ty = _transform_xy(point.x, point.y, from_crs, to_crs)
    return Point3D(x=tx, y=ty, z=point.z)


def transform_stations_to_crs(
    stations: pd.DataFrame,
    to_crs: CoordinateSystem,
    from_crs: CoordinateSystem = DEFAULT_CRS,
    *,
    rename_columns: bool = True,
) -> pd.DataFrame:
    """Transform survey stations to target coordinate system.

    Transforms X_m and Y_m columns while preserving Z_m (TVD).
    Falls back to original values with renamed columns if pyproj
    is unavailable or the transformation path is unsupported.

    Args:
        stations: DataFrame with X_m, Y_m, Z_m columns
        to_crs: Target coordinate system
        from_crs: Source coordinate system (defaults to DEFAULT_CRS)
        rename_columns: Rename X/Y/Z columns for display/export. Keep this
            disabled for internal plotting data, because visualization code
            expects the standard X_m/Y_m/Z_m names.

    Returns:
        Transformed stations DataFrame with renamed columns.
    """
    if to_crs == from_crs:
        return stations.copy()

    result = stations.copy()

    # Determine column suffix based on target CRS
    col_suffix = get_crs_display_suffix(to_crs).strip().replace(" ", "_").strip("()")
    if not col_suffix:
        col_suffix = to_crs.name

    # Check if we can actually transform the values
    can_transform = HAS_PYPROJ and _can_transform_directly(from_crs, to_crs)

    if "X_m" in result.columns and "Y_m" in result.columns and can_transform:
        # Transform each station's x,y coordinates
        transformed = np.zeros((len(result), 2), dtype=float)
        for i in range(len(result)):
            tx, ty = _transform_xy(
                float(result["X_m"].iloc[i]),
                float(result["Y_m"].iloc[i]),
                from_crs,
                to_crs,
            )
            transformed[i, 0] = tx
            transformed[i, 1] = ty
        result["X_m"] = transformed[:, 0]
        result["Y_m"] = transformed[:, 1]

    if rename_columns:
        xy_unit = "deg" if to_crs.is_geographic() else "m"
        if "X_m" in result.columns:
            result = result.rename(columns={"X_m": f"X_{col_suffix}_{xy_unit}"})
        if "Y_m" in result.columns:
            result = result.rename(columns={"Y_m": f"Y_{col_suffix}_{xy_unit}"})
        # Z is vertical depth (TVD) - always preserved
        if "Z_m" in result.columns:
            result = result.rename(columns={"Z_m": "Z_TVD_m"})

    return result


def format_coordinates_for_display(
    x: float,
    y: float,
    z: float,
    crs: CoordinateSystem,
) -> tuple[str, str, str]:
    """Format coordinates for display in UI.

    Uses locale-independent formatting to avoid issues with
    different locale settings (e.g. German uses 1.234,56).

    Args:
        x: X coordinate (easting or longitude)
        y: Y coordinate (northing or latitude)
        z: Z coordinate (TVD or elevation)
        crs: Target coordinate system

    Returns:
        Tuple of formatted (x_str, y_str, z_str)
    """
    if crs.is_geographic():
        # Geographic: degrees with 6 decimal places
        return (
            f"{x:.6f}°",
            f"{y:.6f}°",
            f"{z:.2f} м",
        )
    else:
        # Projected: meters with space as thousands separator (Russian locale)
        def _fmt_meters(v: float) -> str:
            # Use explicit space separator, locale-independent
            s = f"{abs(v):.2f}"
            parts = s.split(".")
            int_part = parts[0]
            # Add spaces every 3 digits from the right
            formatted_int = ""
            for i, ch in enumerate(reversed(int_part)):
                if i > 0 and i % 3 == 0:
                    formatted_int = " " + formatted_int
                formatted_int = ch + formatted_int
            result = formatted_int + "." + parts[1]
            return f"{result} м"

        return (
            _fmt_meters(x),
            _fmt_meters(y),
            f"{z:.2f} м",
        )


def _build_transform_message(
    target_crs: CoordinateSystem,
    has_pyproj: bool,
) -> str:
    """Build user-facing message about transform status."""
    if not has_pyproj:
        return (
            f"Координаты отображены в исходной системе. "
            f"Для пересчёта в {get_crs_display_suffix(target_crs).strip('()')} "
            f"требуется установка pyproj: pip install pyproj"
        )
    if target_crs in {
        CoordinateSystem.PNO_13_ZONE,
        CoordinateSystem.PNO_13_CM,
        CoordinateSystem.PNO_16_ZONE,
        CoordinateSystem.PNO_16_CM,
        CoordinateSystem.LOCAL,
    }:
        return (
            f"Локальная система {get_crs_display_suffix(target_crs)}: "
            f"требуется настройка параметров (false easting, meridian) для пересчёта."
        )
    return ""


def apply_crs_to_well_view(
    view: SingleWellResultView,
    target_crs: CoordinateSystem,
    source_crs: CoordinateSystem = DEFAULT_CRS,
) -> SingleWellResultView:
    """Apply coordinate system transformation to well result view.

    Transforms surface, t1, t3 coordinates and survey stations from
    the source CRS to the target CRS. Preserves z (TVD) values.
    Falls back to original coordinates with a warning if pyproj is
    unavailable or the transformation path is unsupported.

    Args:
        view: Original well result view
        target_crs: Target coordinate system for display
        source_crs: Source coordinate system of the data (defaults to DEFAULT_CRS)

    Returns:
        Well view with transformed coordinates (or original with warning).
    """
    if target_crs == source_crs:
        return view

    # Determine if we can actually transform
    can_transform = HAS_PYPROJ and _can_transform_directly(source_crs, target_crs)
    transform_msg = _build_transform_message(target_crs, HAS_PYPROJ)

    # Transform surface, t1, t3 points
    surface_tx = transform_point_to_crs(view.surface, source_crs, target_crs)
    t1_tx = transform_point_to_crs(view.t1, source_crs, target_crs)
    t3_tx = transform_point_to_crs(view.t3, source_crs, target_crs)

    # Transform stations
    stations_tx = transform_stations_to_crs(
        view.stations,
        target_crs,
        source_crs,
        rename_columns=False,
    )

    # Build new issue messages
    new_messages = list(view.issue_messages)
    if transform_msg and not can_transform:
        new_messages.append(transform_msg)

    # Update summary to indicate CRS used
    summary = dict(view.summary)
    summary["display_crs"] = get_crs_display_suffix(target_crs).strip("()")
    summary["display_crs_xy_unit"] = "deg" if target_crs.is_geographic() else "м"

    # Build and return updated view
    return view.model_copy(
        update={
            "surface": surface_tx,
            "t1": t1_tx,
            "t3": t3_tx,
            "stations": stations_tx,
            "issue_messages": tuple(new_messages),
            "summary": summary,
        }
    )


def get_crs_display_suffix(crs: CoordinateSystem) -> str:
    """Get display suffix for coordinate system.

    Args:
        crs: Coordinate system

    Returns:
        Short suffix for display in tables/headers
    """
    suffix_map = {
        CoordinateSystem.PNO_16_ZONE: " (PNO16)",
        CoordinateSystem.PNO_16_CM: " (PNO16-CM)",
        CoordinateSystem.PNO_13_ZONE: " (PNO13)",
        CoordinateSystem.PNO_13_CM: " (PNO13-CM)",
        CoordinateSystem.PULKOVO_1942: " (СК-42)",
        CoordinateSystem.PULKOVO_1995: " (П95)",
        CoordinateSystem.PULKOVO_1995_CM_39E: " (П95/CM39)",
        CoordinateSystem.GSK_2011: " (ГСК)",
        CoordinateSystem.WGS84: " (WGS)",
    }

    # Add Pulkovo 1942 zone suffixes
    for zone_num in range(6, 21):
        zone_crs = getattr(CoordinateSystem, f"PULKOVO_1942_ZONE_{zone_num}", None)
        if zone_crs:
            suffix_map[zone_crs] = f" (СК-42/З{zone_num})"

    # Add Pulkovo 1995 zone suffixes
    for zone_num in (13, 18):
        zone_crs = getattr(CoordinateSystem, f"PULKOVO_1995_ZONE_{zone_num}", None)
        if zone_crs:
            suffix_map[zone_crs] = f" (П95/З{zone_num})"

    return suffix_map.get(crs, f" ({crs.name})")


__all__ = [
    "render_crs_sidebar",
    "get_selected_crs",
    "should_auto_convert",
    "transform_point_to_crs",
    "transform_stations_to_crs",
    "format_coordinates_for_display",
    "apply_crs_to_well_view",
    "get_crs_display_suffix",
    "DEFAULT_CRS",
    "CRS_OPTIONS",
    "_can_transform_directly",
    "_transform_xy",
    "_effective_pyproj_crs",
]
