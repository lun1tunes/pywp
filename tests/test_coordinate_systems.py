"""Tests for coordinate system transformations."""
import numpy as np
import pytest
from pywp.coordinate_systems import (
    CoordinateSystem, GeodeticCoord, HAS_PYPROJ,
    LocalCoordinateSystem, LocalCoordinateTransformer, ProjectedCoord,
    define_pno_13_system, define_pno_16_system,
    disambiguate_pno13, disambiguate_pno16,
    get_pulkovo_zone
)
if HAS_PYPROJ:
    from pywp.coordinate_systems import CoordinateTransformer


class TestCoordinateSystem:
    def test_known_systems(self) -> None:
        assert CoordinateSystem.WGS84.value == "EPSG:4326"
        assert CoordinateSystem.PULKOVO_1942.value == "EPSG:4284"
        # Pulkovo 1995 from PDF (EPSG:4200)
        assert CoordinateSystem.PULKOVO_1995.value == "EPSG:4200"
        # PNO now disambiguated per PDF research
        assert CoordinateSystem.PNO_13_ZONE.value == "PNO-13-ZONE"
        assert CoordinateSystem.PNO_16_CM.value == "PNO-16-CM"

    def test_is_geographic(self) -> None:
        assert CoordinateSystem.WGS84.is_geographic()
        assert CoordinateSystem.PULKOVO_1995.is_geographic()
        assert not CoordinateSystem.PULKOVO_1942_ZONE_10.is_geographic()
        assert not CoordinateSystem.PULKOVO_1995_ZONE_13.is_geographic()


class TestLocalCoordinateSystem:
    def test_basic_creation(self) -> None:
        local = LocalCoordinateSystem(name="Test", false_easting_m=1000.0)
        assert local.name == "Test"

    def test_invalid_scale(self) -> None:
        with pytest.raises(ValueError, match="Scale factor"):
            LocalCoordinateSystem(name="Bad", scale_factor=0)


class TestLocalTransformations:
    def test_identity_transform(self) -> None:
        local_def = LocalCoordinateSystem(name="Identity", rotation_deg=0.0)
        transformer = LocalCoordinateTransformer(local_def)
        coord = ProjectedCoord(1000.0, 2000.0)
        to_local = transformer.to_local(coord)
        back = transformer.from_local(to_local)
        assert np.isclose(back.easting_m, coord.easting_m)

    def test_translation_only(self) -> None:
        local_def = LocalCoordinateSystem(
            name="Shift", false_easting_m=500.0, false_northing_m=-300.0
        )
        transformer = LocalCoordinateTransformer(local_def)
        base = ProjectedCoord(1000.0, 2000.0)
        local = transformer.to_local(base)
        assert np.isclose(local.easting_m, 500.0)


class TestZoneDetection:
    def test_moscow_zone(self) -> None:
        zone = get_pulkovo_zone(37.6)  # Moscow longitude
        assert zone == CoordinateSystem.PULKOVO_1942_ZONE_7

    def test_gk_13n_42_zone(self) -> None:
        assert get_pulkovo_zone(75.0) == CoordinateSystem.PULKOVO_1942_ZONE_13

    def test_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            get_pulkovo_zone(10.0)  # Too far west


class TestGeodeticCoord:
    def test_repr(self) -> None:
        coord = GeodeticCoord(55.7558, 37.6173)
        repr_str = repr(coord)
        assert "55.7558" in repr_str
        assert "37.6173" in repr_str


class TestPNODisambiguation:
    """Tests per PDF: PNO13/PNO16 require disambiguation by easting."""

    def test_pno16_zone_vs_cm(self) -> None:
        """Per PDF: easting > 1,000,000 → ZONE, else CM."""
        # Large easting → zone-based
        zone_sys = disambiguate_pno16(1_500_000.0)
        assert zone_sys == CoordinateSystem.PNO_16_ZONE

        # Small easting → custom meridian
        cm_sys = disambiguate_pno16(500_000.0)
        assert cm_sys == CoordinateSystem.PNO_16_CM

    def test_pno13_zone_vs_cm(self) -> None:
        """Same logic applies to PNO13."""
        zone_sys = disambiguate_pno13(2_000_000.0)
        assert zone_sys == CoordinateSystem.PNO_13_ZONE

        cm_sys = disambiguate_pno13(800_000.0)
        assert cm_sys == CoordinateSystem.PNO_13_CM

    def test_pno16_boundary(self) -> None:
        """Test exactly at 1,000,000 boundary."""
        # At boundary: should be CM (not > 1M)
        cm_sys = disambiguate_pno16(1_000_000.0)
        assert cm_sys == CoordinateSystem.PNO_16_CM

        # Just above boundary: zone
        zone_sys = disambiguate_pno16(1_000_001.0)
        assert zone_sys == CoordinateSystem.PNO_16_ZONE

    def test_define_pno16_with_easting_hint(self) -> None:
        """Test PNO16 system definition with easting disambiguation."""
        # Zone-based (default)
        zone_def = define_pno_16_system()
        assert "PNO-16" in zone_def.name
        assert zone_def.base_system == CoordinateSystem.PULKOVO_1942_ZONE_16
        assert zone_def.central_meridian_deg == 93.0

        # CM-based (with hint)
        cm_def = define_pno_16_system(easting_hint=500_000.0)
        assert cm_def.central_meridian_deg == 39.0  # Per implementation

    def test_define_pno13_zone_metadata_matches_base_crs(self) -> None:
        """PNO13 zone metadata should match its Pulkovo 1942 GK base zone."""
        zone_def = define_pno_13_system()

        assert zone_def.base_system == CoordinateSystem.PULKOVO_1942_ZONE_13
        assert zone_def.central_meridian_deg == 75.0
