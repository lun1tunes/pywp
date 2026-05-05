"""Tests for coordinate system integration with trajectory planning."""
import pandas as pd
import pytest

from pywp.coordinate_integration import (
    CRS_OPTIONS,
    DEFAULT_CRS,
    _can_transform_directly,
    _transform_xy,
    apply_crs_to_well_view,
    format_coordinates_for_display,
    get_crs_display_suffix,
    transform_point_to_crs,
    transform_stations_to_crs,
)
from pywp.coordinate_systems import CoordinateSystem
from pywp.models import Point3D
from pywp.ui_well_result import SingleWellResultView


class TestCoordinateIntegration:
    """Test coordinate system integration functions."""

    def test_default_crs_is_pno16_zone(self) -> None:
        """Default CRS should be PNO-16 ZONE as per user requirements."""
        assert DEFAULT_CRS == CoordinateSystem.PNO_16_ZONE

    def test_crs_options_contains_pno16(self) -> None:
        """CRS options should include PNO-16 variants."""
        labels = [label for label, _ in CRS_OPTIONS]
        assert "PNO-16 (Зона)" in labels
        assert "PNO-16 (CM)" in labels

    def test_crs_options_contains_pulkovo_zones(self) -> None:
        """CRS options should include Pulkovo 1942 zones."""
        labels = [label for label, _ in CRS_OPTIONS]
        assert "СК-42 Зона 8" in labels
        assert "СК-42 Зона 13" in labels
        assert "СК-42 Зона 16" in labels

    def test_format_coordinates_projected(self) -> None:
        """Format projected coordinates with locale-independent space separator."""
        x, y, z = format_coordinates_for_display(
            1500000.0, 6500000.0, 2500.0, CoordinateSystem.PNO_16_ZONE
        )
        assert x == "1 500 000.00 м"
        assert y == "6 500 000.00 м"
        assert z == "2500.00 м"  # Z uses simple formatting without thousands separator

    def test_format_coordinates_geographic(self) -> None:
        """Format geographic coordinates (degrees)."""
        x, y, z = format_coordinates_for_display(
            37.6173, 55.7558, 150.0, CoordinateSystem.WGS84
        )
        assert "37.617300°" == x
        assert "55.755800°" == y
        assert "150.00 м" == z

    def test_transform_point_same_crs(self) -> None:
        """Transform to same CRS returns point unchanged."""
        point = Point3D(x=1000.0, y=2000.0, z=3000.0)
        result = transform_point_to_crs(
            point,
            CoordinateSystem.LOCAL,
            CoordinateSystem.LOCAL,
        )
        assert result.x == point.x
        assert result.y == point.y
        assert result.z == point.z

    def test_transform_stations_same_crs(self) -> None:
        """Transform stations to same CRS returns copy unchanged."""
        stations = pd.DataFrame({
            "X_m": [1000.0, 2000.0],
            "Y_m": [5000.0, 6000.0],
            "Z_m": [100.0, 200.0],
        })
        result = transform_stations_to_crs(
            stations,
            CoordinateSystem.LOCAL,
            CoordinateSystem.LOCAL,
        )
        assert len(result) == 2
        assert "X_m" in result.columns  # Same CRS: no rename

    def test_transform_stations_different_crs_renames_columns(self) -> None:
        """Transform to different CRS renames columns even without pyproj."""
        stations = pd.DataFrame({
            "X_m": [1000.0, 2000.0],
            "Y_m": [5000.0, 6000.0],
            "Z_m": [100.0, 200.0],
        })
        result = transform_stations_to_crs(
            stations,
            CoordinateSystem.PNO_16_ZONE,
            CoordinateSystem.LOCAL,
        )
        assert len(result) == 2
        # Columns should be renamed to reflect target CRS
        assert "X_PNO16_m" in result.columns
        assert "Y_PNO16_m" in result.columns
        assert "Z_TVD_m" in result.columns

    def test_get_crs_display_suffix_pno16(self) -> None:
        """Display suffix for PNO-16."""
        suffix = get_crs_display_suffix(CoordinateSystem.PNO_16_ZONE)
        assert "PNO16" in suffix

    def test_get_crs_display_suffix_pulkovo1942(self) -> None:
        """Display suffix for Pulkovo 1942."""
        suffix = get_crs_display_suffix(CoordinateSystem.PULKOVO_1942)
        assert "СК-42" in suffix

    def test_get_crs_display_suffix_wgs84(self) -> None:
        """Display suffix for WGS84."""
        suffix = get_crs_display_suffix(CoordinateSystem.WGS84)
        assert "WGS" in suffix

    def test_get_crs_display_suffix_pulkovo1995_cm39(self) -> None:
        """Display suffix for Pulkovo 1995 CM 39E."""
        suffix = get_crs_display_suffix(CoordinateSystem.PULKOVO_1995_CM_39E)
        assert "П95/CM39" in suffix

    def test_can_transform_directly_same_crs(self) -> None:
        """Same CRS is always directly transformable."""
        assert _can_transform_directly(
            CoordinateSystem.WGS84, CoordinateSystem.WGS84
        ) is True

    def test_can_transform_directly_epsg_to_epsg(self) -> None:
        """EPSG-to-EPSG transforms are directly available."""
        assert _can_transform_directly(
            CoordinateSystem.WGS84, CoordinateSystem.PULKOVO_1942
        ) is True

    def test_can_transform_directly_placeholder_blocked(self) -> None:
        """PNO/LOCAL placeholders cannot transform directly."""
        assert _can_transform_directly(
            CoordinateSystem.LOCAL, CoordinateSystem.WGS84
        ) is False
        assert _can_transform_directly(
            CoordinateSystem.PNO_16_ZONE, CoordinateSystem.WGS84
        ) is False

    def test_transform_xy_same_crs(self) -> None:
        """_transform_xy with same CRS returns original values."""
        x, y = _transform_xy(100.0, 200.0, CoordinateSystem.LOCAL, CoordinateSystem.LOCAL)
        assert x == 100.0
        assert y == 200.0

    def test_apply_crs_to_well_view_same_crs(self) -> None:
        """apply_crs_to_well_view with same source/target returns unchanged."""
        stations = pd.DataFrame({
            "X_m": [1000.0],
            "Y_m": [2000.0],
            "Z_m": [100.0],
        })
        view = SingleWellResultView(
            well_name="TEST-01",
            surface=Point3D(x=0.0, y=0.0, z=0.0),
            t1=Point3D(x=1000.0, y=2000.0, z=2500.0),
            t3=Point3D(x=2000.0, y=4000.0, z=2500.0),
            stations=stations,
            summary={"md_m": 5000.0},
            config={"lateral_tolerance_m": 30.0},
            azimuth_deg=45.0,
            md_t1_m=3000.0,
        )
        result = apply_crs_to_well_view(view, CoordinateSystem.LOCAL, CoordinateSystem.LOCAL)
        assert result.surface.x == view.surface.x
        assert result.t1.y == view.t1.y
        assert result.stations is view.stations  # Same CRS: returns original view

    def test_apply_crs_to_well_view_adds_display_crs_to_summary(self) -> None:
        """apply_crs_to_well_view adds display_crs to summary."""
        stations = pd.DataFrame({
            "X_m": [1000.0],
            "Y_m": [2000.0],
            "Z_m": [100.0],
        })
        view = SingleWellResultView(
            well_name="TEST-01",
            surface=Point3D(x=0.0, y=0.0, z=0.0),
            t1=Point3D(x=1000.0, y=2000.0, z=2500.0),
            t3=Point3D(x=2000.0, y=4000.0, z=2500.0),
            stations=stations,
            summary={"md_m": 5000.0},
            config={"lateral_tolerance_m": 30.0},
            azimuth_deg=45.0,
            md_t1_m=3000.0,
        )
        result = apply_crs_to_well_view(
            view, CoordinateSystem.PNO_16_ZONE, CoordinateSystem.LOCAL
        )
        assert "display_crs" in result.summary
        assert "PNO16" in result.summary["display_crs"]

    def test_apply_crs_to_well_view_preserves_z(self) -> None:
        """apply_crs_to_well_view always preserves z (TVD)."""
        stations = pd.DataFrame({
            "X_m": [1000.0],
            "Y_m": [2000.0],
            "Z_m": [100.0],
        })
        view = SingleWellResultView(
            well_name="TEST-01",
            surface=Point3D(x=0.0, y=0.0, z=500.0),
            t1=Point3D(x=1000.0, y=2000.0, z=2500.0),
            t3=Point3D(x=2000.0, y=4000.0, z=2500.0),
            stations=stations,
            summary={"md_m": 5000.0},
            config={"lateral_tolerance_m": 30.0},
            azimuth_deg=45.0,
            md_t1_m=3000.0,
        )
        result = apply_crs_to_well_view(
            view, CoordinateSystem.PNO_16_ZONE, CoordinateSystem.LOCAL
        )
        assert result.surface.z == 500.0
        assert result.t1.z == 2500.0
        assert result.t3.z == 2500.0

