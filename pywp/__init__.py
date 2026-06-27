from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner

_COORDINATE_SYSTEM_EXPORTS = {
    "CoordinateSystem": ("pywp.coordinate_systems", "CoordinateSystem"),
    "CoordinateTransformer": ("pywp.coordinate_systems", "CoordinateTransformer"),
    "LocalCoordinateSystem": ("pywp.coordinate_systems", "LocalCoordinateSystem"),
    "GeoPoint": ("pywp.geo_point", "GeoPoint"),
    "KnownCRSs": ("pywp.geo_point", "KnownCRSs"),
    "disambiguate_pno13": ("pywp.coordinate_systems", "disambiguate_pno13"),
    "disambiguate_pno16": ("pywp.coordinate_systems", "disambiguate_pno16"),
    "define_pno_13_system": ("pywp.coordinate_systems", "define_pno_13_system"),
    "define_pno_16_system": ("pywp.coordinate_systems", "define_pno_16_system"),
    "get_pulkovo_zone": ("pywp.coordinate_systems", "get_pulkovo_zone"),
}

_COORDINATE_INTEGRATION_EXPORTS = {
    "DEFAULT_CRS": ("pywp.coordinate_integration", "DEFAULT_CRS"),
    "render_crs_sidebar": ("pywp.coordinate_integration", "render_crs_sidebar"),
    "get_input_crs": ("pywp.coordinate_integration", "get_input_crs"),
    "get_selected_crs": ("pywp.coordinate_integration", "get_selected_crs"),
    "should_auto_convert": ("pywp.coordinate_integration", "should_auto_convert"),
    "apply_crs_to_well_view": ("pywp.coordinate_integration", "apply_crs_to_well_view"),
    "get_crs_display_suffix": (
        "pywp.coordinate_integration",
        "get_crs_display_suffix",
    ),
}

HAS_COORDINATE_SYSTEMS = find_spec("pyproj") is not None
HAS_COORDINATE_INTEGRATION = bool(
    HAS_COORDINATE_SYSTEMS and find_spec("streamlit") is not None
)


def _mark_optional_exports_unavailable(export_names: set[str], flag_name: str) -> None:
    globals()[flag_name] = False
    for export_name in export_names:
        globals()[export_name] = None


def _load_optional_export(
    name: str,
    *,
    exports: dict[str, tuple[str, str]],
    flag_name: str,
) -> Any:
    module_name, attr_name = exports[name]
    try:
        module = import_module(module_name)
    except ImportError:
        _mark_optional_exports_unavailable(set(exports), flag_name)
        return None
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if not HAS_COORDINATE_SYSTEMS:
    _mark_optional_exports_unavailable(
        set(_COORDINATE_SYSTEM_EXPORTS), "HAS_COORDINATE_SYSTEMS"
    )

if not HAS_COORDINATE_INTEGRATION:
    _mark_optional_exports_unavailable(
        set(_COORDINATE_INTEGRATION_EXPORTS), "HAS_COORDINATE_INTEGRATION"
    )

__all__ = [
    "PlannerResult",
    "Point3D",
    "TrajectoryConfig",
    "PlanningError",
    "TrajectoryPlanner",
    # Coordinate systems (conditional)
    "CoordinateSystem",
    "CoordinateTransformer",
    "GeoPoint",
    "LocalCoordinateSystem",
    "KnownCRSs",
    "disambiguate_pno13",
    "disambiguate_pno16",
    "define_pno_13_system",
    "define_pno_16_system",
    "get_pulkovo_zone",
    # Coordinate integration (conditional)
    "DEFAULT_CRS",
    "render_crs_sidebar",
    "get_input_crs",
    "get_selected_crs",
    "should_auto_convert",
    "apply_crs_to_well_view",
    "get_crs_display_suffix",
    # Feature flags
    "HAS_COORDINATE_SYSTEMS",
    "HAS_COORDINATE_INTEGRATION",
]


def __getattr__(name: str) -> Any:
    if name in _COORDINATE_SYSTEM_EXPORTS:
        if not bool(HAS_COORDINATE_SYSTEMS):
            return None
        return _load_optional_export(
            name,
            exports=_COORDINATE_SYSTEM_EXPORTS,
            flag_name="HAS_COORDINATE_SYSTEMS",
        )
    if name in _COORDINATE_INTEGRATION_EXPORTS:
        if not bool(HAS_COORDINATE_INTEGRATION):
            return None
        return _load_optional_export(
            name,
            exports=_COORDINATE_INTEGRATION_EXPORTS,
            flag_name="HAS_COORDINATE_INTEGRATION",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
