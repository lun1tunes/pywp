from pywp.models import PlannerResult, Point3D, TrajectoryConfig
from pywp.planner import PlanningError, TrajectoryPlanner

# Core coordinate systems (only needs pyproj)
try:
    from pywp.coordinate_systems import (
        CoordinateSystem,
        CoordinateTransformer,
        LocalCoordinateSystem,
        disambiguate_pno13,
        disambiguate_pno16,
        define_pno_13_system,
        define_pno_16_system,
        get_pulkovo_zone,
    )
    from pywp.geo_point import GeoPoint, KnownCRSs
    HAS_COORDINATE_SYSTEMS = True
except ImportError:
    HAS_COORDINATE_SYSTEMS = False
    CoordinateSystem = None  # type: ignore[assignment,misc]
    CoordinateTransformer = None  # type: ignore[assignment,misc]
    LocalCoordinateSystem = None  # type: ignore[assignment,misc]
    GeoPoint = None  # type: ignore[assignment,misc]
    KnownCRSs = None  # type: ignore[assignment,misc]
    disambiguate_pno13 = None  # type: ignore[assignment,misc]
    disambiguate_pno16 = None  # type: ignore[assignment,misc]
    define_pno_13_system = None  # type: ignore[assignment,misc]
    define_pno_16_system = None  # type: ignore[assignment,misc]
    get_pulkovo_zone = None  # type: ignore[assignment,misc]

# Coordinate integration for Streamlit UI (needs both pyproj and streamlit)
try:
    from pywp.coordinate_integration import (
        DEFAULT_CRS,
        apply_crs_to_well_view,
        get_crs_display_suffix,
        get_input_crs,
        get_selected_crs,
        render_crs_sidebar,
        should_auto_convert,
    )
    HAS_COORDINATE_INTEGRATION = True
except ImportError:
    HAS_COORDINATE_INTEGRATION = False
    DEFAULT_CRS = None  # type: ignore[assignment,misc]
    apply_crs_to_well_view = None  # type: ignore[assignment,misc]
    get_crs_display_suffix = None  # type: ignore[assignment,misc]
    get_input_crs = None  # type: ignore[assignment,misc]
    get_selected_crs = None  # type: ignore[assignment,misc]
    render_crs_sidebar = None  # type: ignore[assignment,misc]
    should_auto_convert = None  # type: ignore[assignment,misc]

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
