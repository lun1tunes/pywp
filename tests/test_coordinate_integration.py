"""Tests for coordinate system integration with trajectory planning."""
import pandas as pd
import pytest

import pywp.coordinate_integration as ci
from pywp.coordinate_integration import (
    CSV_CRS_OPTIONS,
    CRS_OPTIONS,
    DEFAULT_CRS,
    INPUT_CRS_OPTIONS,
    _can_transform_directly,
    csv_export_crs,
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

    def test_default_crs_is_gk_13n_42(self) -> None:
        """Default CRS should be ГК_13N_42 as per user requirements."""
        assert DEFAULT_CRS == CoordinateSystem.PULKOVO_1942_GK_13N

    def test_input_crs_options_are_projected_meter_systems_only(self) -> None:
        """Input CRS options stay to projected meter systems."""
        labels = [label for label, _ in INPUT_CRS_OPTIONS]
        assert "PNO-16 (Зона)" not in labels
        assert "WGS84" not in labels
        assert "СК-42 (Градусные)" not in labels
        assert all(crs.is_projected() for _label, crs in INPUT_CRS_OPTIONS)
        assert not CoordinateSystem.GSK_2011_GEOCENTRIC.is_projected()

    def test_csv_crs_options_include_geographic_outputs(self) -> None:
        """CSV output CRS options include projected meters and geographic degrees."""
        labels = [label for label, _ in CSV_CRS_OPTIONS]
        assert CRS_OPTIONS == CSV_CRS_OPTIONS
        assert "WGS84 (Градусные)" in labels
        assert "СК-42 (Градусные)" in labels
        assert "WGS84 UTM 43N" in labels
        assert "ГК_13N_42" in labels
        assert CoordinateSystem.WGS84 in {crs for _label, crs in CSV_CRS_OPTIONS}
        assert CoordinateSystem.WGS84_UTM_ZONE_43N in {
            crs for _label, crs in CSV_CRS_OPTIONS
        }

    def test_geographic_csv_outputs_are_really_geographic_crs(self) -> None:
        """CRS options labelled as degree outputs must not point to geocentric EPSG."""
        geographic_labels = {
            label: crs
            for label, crs in CSV_CRS_OPTIONS
            if "Градусные" in label
        }

        assert geographic_labels
        assert all(crs.is_geographic() for crs in geographic_labels.values())
        assert CoordinateSystem.GSK_2011.value == "EPSG:7683"

    def test_crs_options_contains_pulkovo_zones(self) -> None:
        """CRS options should include Pulkovo 1942 zones."""
        labels = [label for label, _ in CRS_OPTIONS]
        assert "СК-42 Зона 8" in labels
        assert "ГК_13N_42" in labels
        assert "СК-42 Зона 13 (13 млн)" in labels
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

    def test_transform_stations_different_crs_labels_source_when_unsupported(self) -> None:
        """Unsupported transforms keep source CRS labels to avoid false units."""
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
        assert "X_LOCAL_m" in result.columns
        assert "Y_LOCAL_m" in result.columns
        assert "Z_TVD_m" in result.columns

    def test_get_crs_display_suffix_pno16(self) -> None:
        """Display suffix for PNO-16."""
        suffix = get_crs_display_suffix(CoordinateSystem.PNO_16_ZONE)
        assert "PNO16" in suffix

    def test_get_crs_display_suffix_pulkovo1942(self) -> None:
        """Display suffix for Pulkovo 1942."""
        suffix = get_crs_display_suffix(CoordinateSystem.PULKOVO_1942)
        assert "СК-42" in suffix

    def test_get_crs_display_suffix_gk_13n_42(self) -> None:
        """Display suffix for the default GK_13N_42 CRS."""
        suffix = get_crs_display_suffix(CoordinateSystem.PULKOVO_1942_GK_13N)
        assert "ГК_13N_42" in suffix

    def test_get_crs_display_suffix_wgs84_utm43(self) -> None:
        """Display suffix for WGS84 UTM zone 43N."""
        suffix = get_crs_display_suffix(CoordinateSystem.WGS84_UTM_ZONE_43N)
        assert "UTM43N" in suffix

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

    def test_can_transform_directly_local_and_pno_placeholders_blocked(self) -> None:
        """LOCAL and PNO placeholders require project-specific parameters."""
        assert _can_transform_directly(
            CoordinateSystem.LOCAL, CoordinateSystem.WGS84
        ) is False
        assert _can_transform_directly(
            CoordinateSystem.PNO_16_ZONE, CoordinateSystem.WGS84
        ) is False

    def test_csv_export_crs_falls_back_to_source_for_pno_placeholder(self) -> None:
        """CSV labels must match actual values when PNO conversion is unavailable."""
        assert (
            csv_export_crs(
                CoordinateSystem.WGS84,
                CoordinateSystem.PNO_16_ZONE,
                auto_convert=True,
            )
            == CoordinateSystem.PNO_16_ZONE
        )
        assert (
            csv_export_crs(
                CoordinateSystem.WGS84,
                CoordinateSystem.PNO_16_ZONE,
                auto_convert=False,
            )
            == CoordinateSystem.PNO_16_ZONE
        )

    def test_transform_xy_same_crs(self) -> None:
        """_transform_xy with same CRS returns original values."""
        x, y = _transform_xy(100.0, 200.0, CoordinateSystem.LOCAL, CoordinateSystem.LOCAL)
        assert x == 100.0
        assert y == 200.0

    @pytest.mark.skipif(not ci.HAS_PYPROJ, reason="pyproj is required")
    def test_transform_gk_13n_42_to_wgs84_utm43_control_point(self) -> None:
        """ГК_13N_42 (EPSG:2503/28473 equivalent) converts to WGS84 UTM43N."""
        x, y = _transform_xy(
            500_053.72885872814,
            6_097_276.68079257,
            CoordinateSystem.PULKOVO_1942_GK_13N,
            CoordinateSystem.WGS84_UTM_ZONE_43N,
        )

        assert x == pytest.approx(499_999.999696543, abs=0.001)
        assert y == pytest.approx(6_094_791.4212895455, abs=0.001)

    @pytest.mark.skipif(not ci.HAS_PYPROJ, reason="pyproj is required")
    def test_transform_gk_13n_42_to_wgs84_degrees_control_point(self) -> None:
        """The same control point resolves back to lon/lat near 75E, 55N."""
        lon, lat = _transform_xy(
            500_053.72885872814,
            6_097_276.68079257,
            CoordinateSystem.PULKOVO_1942_GK_13N,
            CoordinateSystem.WGS84,
        )

        assert lon == pytest.approx(75.0, abs=1e-7)
        assert lat == pytest.approx(55.0, abs=1e-7)

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

    def test_apply_crs_to_well_view_labels_actual_display_crs(self) -> None:
        """Unsupported transforms keep source CRS in display metadata."""
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
        assert "LOCAL" in result.summary["display_crs"]

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

    def test_apply_crs_to_well_view_keeps_internal_station_columns(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Transformed views keep X_m/Y_m so Plotly builders keep working."""
        stations = pd.DataFrame({
            "X_m": [1000.0],
            "Y_m": [2000.0],
            "Z_m": [100.0],
        })
        view = SingleWellResultView(
            well_name="TEST-01",
            surface=Point3D(x=10.0, y=20.0, z=500.0),
            t1=Point3D(x=1000.0, y=2000.0, z=2500.0),
            t3=Point3D(x=2000.0, y=4000.0, z=2500.0),
            stations=stations,
            summary={"md_m": 5000.0},
            config={"lateral_tolerance_m": 30.0},
            azimuth_deg=45.0,
            md_t1_m=3000.0,
        )

        monkeypatch.setattr(ci, "HAS_PYPROJ", True)
        monkeypatch.setattr(
            ci,
            "_transform_xy",
            lambda x, y, _from_crs, _to_crs: (float(x) + 1.0, float(y) + 2.0),
        )

        result = apply_crs_to_well_view(
            view,
            CoordinateSystem.WGS84,
            CoordinateSystem.PULKOVO_1942_ZONE_16,
        )

        assert list(result.stations.columns) == ["X_m", "Y_m", "Z_m"]
        assert result.stations["X_m"].tolist() == pytest.approx([1001.0])
        assert result.stations["Y_m"].tolist() == pytest.approx([2002.0])
        assert result.surface.x == pytest.approx(11.0)
        assert result.summary["display_crs_xy_unit"] == "deg"

    def test_apply_crs_to_well_view_transforms_multi_horizontal_pairs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stations = pd.DataFrame({"X_m": [0.0], "Y_m": [0.0], "Z_m": [0.0]})
        view = SingleWellResultView(
            well_name="MULTI",
            surface=Point3D(x=10.0, y=20.0, z=0.0),
            t1=Point3D(x=100.0, y=200.0, z=1000.0),
            t3=Point3D(x=900.0, y=200.0, z=1040.0),
            target_pairs=(
                (
                    Point3D(x=100.0, y=200.0, z=1000.0),
                    Point3D(x=500.0, y=200.0, z=1000.0),
                ),
                (
                    Point3D(x=600.0, y=200.0, z=1040.0),
                    Point3D(x=900.0, y=200.0, z=1040.0),
                ),
            ),
            stations=stations,
            summary={"md_m": 5000.0},
            config={"lateral_tolerance_m": 30.0},
            azimuth_deg=45.0,
            md_t1_m=3000.0,
        )
        monkeypatch.setattr(ci, "HAS_PYPROJ", True)
        monkeypatch.setattr(
            ci,
            "_transform_xy",
            lambda x, y, _from_crs, _to_crs: (float(x) + 1.0, float(y) + 2.0),
        )

        result = apply_crs_to_well_view(
            view,
            CoordinateSystem.WGS84,
            CoordinateSystem.PULKOVO_1942_ZONE_16,
        )

        assert result.target_pairs[0][0].x == pytest.approx(101.0)
        assert result.target_pairs[0][0].y == pytest.approx(202.0)
        assert result.target_pairs[1][1].x == pytest.approx(901.0)
        assert result.target_pairs[1][1].z == pytest.approx(1040.0)

    def test_transform_stations_can_keep_standard_columns(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stations = pd.DataFrame({
            "X_m": [1000.0],
            "Y_m": [2000.0],
            "Z_m": [100.0],
        })
        monkeypatch.setattr(ci, "HAS_PYPROJ", True)
        monkeypatch.setattr(
            ci,
            "_transform_xy",
            lambda x, y, _from_crs, _to_crs: (float(x) + 1.0, float(y) + 2.0),
        )

        result = transform_stations_to_crs(
            stations,
            CoordinateSystem.WGS84,
            CoordinateSystem.PULKOVO_1942_ZONE_16,
            rename_columns=False,
        )

        assert list(result.columns) == ["X_m", "Y_m", "Z_m"]
        assert result["X_m"].tolist() == pytest.approx([1001.0])
