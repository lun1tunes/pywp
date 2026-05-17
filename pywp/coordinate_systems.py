"""Coordinate system transformations for well trajectory planning.

Supports Russian petroleum industry coordinate systems:
- Pulkovo 1942 (СК-42) - EPSG:4284 - Legacy Soviet system
- GSK-2011 (ГСК-2011) - EPSG:7683 geographic / EPSG:7681 geocentric
- WGS84 - EPSG:4326 - Global GPS standard
- Custom local systems (PNO-13, PNO-16, etc.)

For projected coordinates (Gauss-Kruger, UTM), use zone-specific EPSG codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import numpy as np

# Optional pyproj support
try:
    from pyproj import Transformer, CRS

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


class CoordinateSystemError(RuntimeError):
    """Raised for coordinate system operations errors."""


class CoordinateSystem(Enum):
    """Supported coordinate systems for Russian petroleum industry.

    Based on research: "Преобразования координат между Pulkovo PNO13,
    PNO16, СК-42 и WGS 84 для нефтегазовых данных"
    """

    # Global standard
    WGS84 = "EPSG:4326"
    WGS84_UTM = "EPSG:32600"  # Base for UTM zones
    WGS84_UTM_ZONE_43N = "EPSG:32643"  # WGS 84 / UTM zone 43N

    # Russian state systems
    PULKOVO_1942 = "EPSG:4284"  # СК-42, legacy Soviet (accuracy ~3m)
    PULKOVO_1995 = "EPSG:4200"  # Pulkovo 1995, modernized (accuracy ~1m)
    GSK_2011_GEOCENTRIC = "EPSG:7681"  # GSK-2011 geocentric X/Y/Z, metres
    GSK_2011 = "EPSG:7683"  # ГСК-2011 geographic 2D, degrees

    # Pulkovo 1942 Gauss-Kruger zones (6-degree, legacy).
    # EPSG:28413 carries zone-prefixed false easting 13,500,000 m.
    # EPSG:2503 is the non-deprecated truncated CM 75E form equivalent
    # to the deprecated EPSG:28473 "Gauss-Kruger 13N" definition.
    PULKOVO_1942_GK_13N = "EPSG:2503"    # CM 75°, false easting 500,000 m
    PULKOVO_1942_ZONE_6 = "EPSG:28406"   # CM 33°
    PULKOVO_1942_ZONE_7 = "EPSG:28407"   # CM 39°
    PULKOVO_1942_ZONE_8 = "EPSG:28408"   # CM 45°
    PULKOVO_1942_ZONE_9 = "EPSG:28409"   # CM 51°
    PULKOVO_1942_ZONE_10 = "EPSG:28410"  # CM 57°
    PULKOVO_1942_ZONE_11 = "EPSG:28411"  # CM 63°
    PULKOVO_1942_ZONE_12 = "EPSG:28412"  # CM 69°
    PULKOVO_1942_ZONE_13 = "EPSG:28413"  # CM 75°
    PULKOVO_1942_ZONE_14 = "EPSG:28414"  # CM 81°
    PULKOVO_1942_ZONE_15 = "EPSG:28415"  # CM 87°
    PULKOVO_1942_ZONE_16 = "EPSG:28416"  # CM 93°
    PULKOVO_1942_ZONE_17 = "EPSG:28417"  # CM 99°
    PULKOVO_1942_ZONE_18 = "EPSG:28418"  # CM 105°
    PULKOVO_1942_ZONE_19 = "EPSG:28419"  # CM 111°
    PULKOVO_1942_ZONE_20 = "EPSG:28420"  # CM 117°

    # Pulkovo 1995 Gauss-Kruger zones (modern, per PDF EPSG:2472, 20073)
    PULKOVO_1995_ZONE_13 = "EPSG:2472"   # Zone 13, CM 75°E
    PULKOVO_1995_ZONE_18 = "EPSG:20073"  # Zone 18, CM 105°E
    PULKOVO_1995_CM_39E = "EPSG:5043"    # Custom meridian 39°E

    # PNO systems - AMBIGUOUS, require disambiguation (per PDF)
    # PNO_16: if easting > 1,000,000 → ZONE, else CM (custom meridian)
    PNO_13_ZONE = "PNO-13-ZONE"    # Zone-based (easting > 1M typically)
    PNO_13_CM = "PNO-13-CM"        # Custom meridian
    PNO_16_ZONE = "PNO-16-ZONE"    # Zone-based
    PNO_16_CM = "PNO-16-CM"        # Custom meridian (< 1M easting)
    LOCAL = "LOCAL"  # Generic local system

    def __repr__(self) -> str:
        return f"{self.name}({self.value})"

    def is_geographic(self) -> bool:
        """True if system uses latitude/longitude."""
        return self in {
            CoordinateSystem.WGS84,
            CoordinateSystem.PULKOVO_1942,
            CoordinateSystem.PULKOVO_1995,
            CoordinateSystem.GSK_2011,
        }

    def is_geocentric(self) -> bool:
        """True if system uses earth-centered XYZ coordinates."""
        return self == CoordinateSystem.GSK_2011_GEOCENTRIC

    def is_projected(self) -> bool:
        """True if system uses projected coordinates (meters)."""
        return (
            not self.is_geographic()
            and not self.is_geocentric()
            and self not in {
                CoordinateSystem.PNO_13_ZONE,
                CoordinateSystem.PNO_13_CM,
                CoordinateSystem.PNO_16_ZONE,
                CoordinateSystem.PNO_16_CM,
                CoordinateSystem.LOCAL,
            }
        )


@dataclass(frozen=True)
class LocalCoordinateSystem:
    """Definition of a custom local coordinate system.

    For oilfield-specific coordinate systems like PNO-13, PNO-16.
    """

    name: str
    # False easting and northing (shift from reference)
    false_easting_m: float = 0.0
    false_northing_m: float = 0.0
    # Rotation angle in degrees (clockwise from north)
    rotation_deg: float = 0.0
    # Scale factor (usually 1.0)
    scale_factor: float = 1.0
    # Reference system this local system is based on
    base_system: CoordinateSystem = CoordinateSystem.PULKOVO_1942
    # Central meridian for projection (if applicable)
    central_meridian_deg: float | None = None

    def __post_init__(self) -> None:
        if self.scale_factor <= 0:
            raise ValueError("Scale factor must be positive")


class GeodeticCoord(NamedTuple):
    """Geographic coordinates: latitude and longitude in degrees."""

    lat_deg: float
    lon_deg: float

    def __repr__(self) -> str:
        ns = "N" if self.lat_deg >= 0 else "S"
        ew = "E" if self.lon_deg >= 0 else "W"
        return f"GeodeticCoord({abs(self.lat_deg):.6f}°{ns}, {abs(self.lon_deg):.6f}°{ew})"


class ProjectedCoord(NamedTuple):
    """Projected coordinates: easting and northing in meters."""

    easting_m: float
    northing_m: float

    def __repr__(self) -> str:
        return f"ProjectedCoord(E={self.easting_m:.2f}m, N={self.northing_m:.2f}m)"


class CoordinateTransformer:
    """Transform between coordinate systems using pyproj.

    Example:
        >>> transformer = CoordinateTransformer()
        >>> wgs84 = GeodeticCoord(55.7558, 37.6173)  # Moscow
        >>> pulkovo = transformer.transform(wgs84, CS.WGS84, CS.PULKOVO_1942)
    """

    def __init__(self) -> None:
        if not HAS_PYPROJ:
            raise CoordinateSystemError(
                "pyproj is required for coordinate transformations. "
                "Install with: pip install pyproj"
            )
        self._cache: dict[tuple[str, str], Transformer] = {}

    def _get_transformer(
        self, from_crs: CoordinateSystem | str, to_crs: CoordinateSystem | str
    ) -> Transformer:
        """Get or create transformer for given coordinate systems."""
        from_str = from_crs.value if isinstance(from_crs, CoordinateSystem) else from_crs
        to_str = to_crs.value if isinstance(to_crs, CoordinateSystem) else to_crs

        key = (from_str, to_str)
        if key not in self._cache:
            try:
                self._cache[key] = Transformer.from_crs(
                    from_str, to_str, always_xy=True
                )
            except Exception as e:
                raise CoordinateSystemError(
                    f"Failed to create transformer from {from_str} to {to_str}: {e}"
                )
        return self._cache[key]

    def transform(
        self,
        coord: GeodeticCoord | ProjectedCoord,
        from_crs: CoordinateSystem,
        to_crs: CoordinateSystem,
    ) -> GeodeticCoord | ProjectedCoord:
        """Transform single coordinate between systems.

        Args:
            coord: Input coordinate
            from_crs: Source coordinate system
            to_crs: Target coordinate system

        Returns:
            Transformed coordinate (same type as input if both systems
            are geographic or both projected, otherwise appropriate type)
        """
        if from_crs == to_crs:
            return coord

        transformer = self._get_transformer(from_crs, to_crs)

        if isinstance(coord, GeodeticCoord):
            x, y = transformer.transform(coord.lon_deg, coord.lat_deg)
        else:
            x, y = transformer.transform(coord.easting_m, coord.northing_m)

        if to_crs.is_geographic():
            return GeodeticCoord(lat_deg=y, lon_deg=x)
        else:
            return ProjectedCoord(easting_m=x, northing_m=y)

    def transform_array(
        self,
        coords: np.ndarray,
        from_crs: CoordinateSystem,
        to_crs: CoordinateSystem,
    ) -> np.ndarray:
        """Transform array of coordinates.

        Args:
            coords: Nx2 array of [lon, lat] or [easting, northing]
            from_crs: Source coordinate system
            to_crs: Target coordinate system

        Returns:
            Nx2 array of transformed coordinates
        """
        if from_crs == to_crs:
            return coords.copy()

        transformer = self._get_transformer(from_crs, to_crs)
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])


class LocalCoordinateTransformer:
    """Transform to/from local oilfield coordinate systems.

    Handles custom local systems like PNO-13, PNO-16 that are typically:
    - Based on a national system (e.g., Pulkovo 1942)
    - Shifted by false easting/northing
    - Possibly rotated
    - Used within a specific oil/gas field
    """

    def __init__(self, local_def: LocalCoordinateSystem) -> None:
        self.local = local_def
        self._rotation_rad = np.radians(local_def.rotation_deg)
        self._cos_r = np.cos(self._rotation_rad)
        self._sin_r = np.sin(self._rotation_rad)

        # Base transformer if base system is not local and pyproj available
        self._base_transformer = None
        if local_def.base_system not in {
            CoordinateSystem.PNO_13_ZONE,
            CoordinateSystem.PNO_13_CM,
            CoordinateSystem.PNO_16_ZONE,
            CoordinateSystem.PNO_16_CM,
            CoordinateSystem.LOCAL,
        }:
            if HAS_PYPROJ:
                self._base_transformer = CoordinateTransformer()

    def to_local(self, coord: ProjectedCoord) -> ProjectedCoord:
        """Transform from base system to local system."""
        # Translate
        x = coord.easting_m - self.local.false_easting_m
        y = coord.northing_m - self.local.false_northing_m

        # Rotate
        if self._rotation_rad != 0:
            x_rot = x * self._cos_r + y * self._sin_r
            y_rot = -x * self._sin_r + y * self._cos_r
            x, y = x_rot, y_rot

        # Scale
        x *= self.local.scale_factor
        y *= self.local.scale_factor

        return ProjectedCoord(easting_m=x, northing_m=y)

    def from_local(self, coord: ProjectedCoord) -> ProjectedCoord:
        """Transform from local system to base system."""
        # Inverse scale
        x = coord.easting_m / self.local.scale_factor
        y = coord.northing_m / self.local.scale_factor

        # Inverse rotate
        if self._rotation_rad != 0:
            x_rot = x * self._cos_r - y * self._sin_r
            y_rot = x * self._sin_r + y * self._cos_r
            x, y = x_rot, y_rot

        # Translate back
        x += self.local.false_easting_m
        y += self.local.false_northing_m

        return ProjectedCoord(easting_m=x, northing_m=y)

    def to_wgs84(
        self, local_coord: ProjectedCoord
    ) -> GeodeticCoord | ProjectedCoord:
        """Transform from local to WGS84.

        Returns:
            WGS84 geodetic coordinate (if path available)
        """
        base_coord = self.from_local(local_coord)

        if self._base_transformer is None:
            return base_coord  # Cannot convert further

        # Assume base is projected, need zone info
        # This is simplified - real implementation needs proper CRS handling
        raise NotImplementedError(
            "Direct local to WGS84 requires zone-specific implementation"
        )


def get_pulkovo_zone(longitude_deg: float) -> CoordinateSystem:
    """Get appropriate Pulkovo 1942 Gauss-Kruger zone for given longitude.

    Args:
        longitude_deg: Longitude in degrees

    Returns:
        Appropriate zone (6-20) or raises ValueError
    """
    if not 30.0 <= longitude_deg <= 120.0:
        raise ValueError(
            f"Longitude {longitude_deg}° outside supported Pulkovo zone range "
            "(30°-120°E)"
        )

    # Gauss-Kruger 6-degree zones: zone N has central meridian 6N - 3.
    zone = int(np.floor(float(longitude_deg) / 6.0)) + 1

    zone_mapping = {
        6: CoordinateSystem.PULKOVO_1942_ZONE_6,
        7: CoordinateSystem.PULKOVO_1942_ZONE_7,
        8: CoordinateSystem.PULKOVO_1942_ZONE_8,
        9: CoordinateSystem.PULKOVO_1942_ZONE_9,
        10: CoordinateSystem.PULKOVO_1942_ZONE_10,
        11: CoordinateSystem.PULKOVO_1942_ZONE_11,
        12: CoordinateSystem.PULKOVO_1942_ZONE_12,
        13: CoordinateSystem.PULKOVO_1942_ZONE_13,
        14: CoordinateSystem.PULKOVO_1942_ZONE_14,
        15: CoordinateSystem.PULKOVO_1942_ZONE_15,
        16: CoordinateSystem.PULKOVO_1942_ZONE_16,
        17: CoordinateSystem.PULKOVO_1942_ZONE_17,
        18: CoordinateSystem.PULKOVO_1942_ZONE_18,
        19: CoordinateSystem.PULKOVO_1942_ZONE_19,
        20: CoordinateSystem.PULKOVO_1942_ZONE_20,
    }

    if zone > 20:
        zone = 20

    return zone_mapping.get(zone, CoordinateSystem.PULKOVO_1942_ZONE_10)


def disambiguate_pno16(easting_m: float) -> CoordinateSystem:
    """Disambiguate PNO16 based on easting value.

    Per PDF research: PNO16 is ambiguous - provide easting or context.
    If easting > 1,000,000 → ZONE-based, else CM (custom meridian).

    Args:
        easting_m: Easting coordinate in meters

    Returns:
        Disambiguated PNO_16_ZONE or PNO_16_CM
    """
    if easting_m > 1_000_000:
        return CoordinateSystem.PNO_16_ZONE
    else:
        return CoordinateSystem.PNO_16_CM


def disambiguate_pno13(easting_m: float) -> CoordinateSystem:
    """Disambiguate PNO13 based on easting value.

    Same logic as PNO16: > 1,000,000 m → ZONE-based.

    Args:
        easting_m: Easting coordinate in meters

    Returns:
        Disambiguated PNO_13_ZONE or PNO_13_CM
    """
    if easting_m > 1_000_000:
        return CoordinateSystem.PNO_13_ZONE
    else:
        return CoordinateSystem.PNO_13_CM


def define_pno_13_system(
    easting_hint: float | None = None,
) -> LocalCoordinateSystem:
    """Define PNO-13 coordinate system with disambiguation.

    Per PDF: PNO13 is NOT an authority - must be normalized to EPSG/GOST.
    If easting_hint provided, uses it for ZONE vs CM selection.

    Args:
        easting_hint: Optional easting value to disambiguate zone vs CM
    """
    # Default: zone-based (modern fields typically use zones)
    base = CoordinateSystem.PULKOVO_1942_ZONE_13
    cm = 75.0

    if easting_hint is not None and easting_hint < 1_000_000:
        # Custom meridian case
        base = CoordinateSystem.PULKOVO_1995_CM_39E  # Example fallback
        cm = 39.0

    return LocalCoordinateSystem(
        name="PNO-13",
        false_easting_m=0.0,  # Real value from field documentation
        false_northing_m=0.0,
        rotation_deg=0.0,
        scale_factor=1.0,
        base_system=base,
        central_meridian_deg=cm,
    )


def define_pno_16_system(
    easting_hint: float | None = None,
) -> LocalCoordinateSystem:
    """Define PNO-16 coordinate system with disambiguation.

    Per PDF: PNO16 is ambiguous - if easting > 1,000,000 → ZONE, else CM.

    Args:
        easting_hint: Optional easting value to disambiguate zone vs CM
    """
    # Default: zone-based
    base = CoordinateSystem.PULKOVO_1942_ZONE_16
    cm = 93.0

    if easting_hint is not None and easting_hint < 1_000_000:
        # Custom meridian case (< 1M easting)
        base = CoordinateSystem.PULKOVO_1995_CM_39E
        cm = 39.0

    return LocalCoordinateSystem(
        name="PNO-16",
        false_easting_m=0.0,
        false_northing_m=0.0,
        rotation_deg=0.0,
        scale_factor=1.0,
        base_system=base,
        central_meridian_deg=cm,
    )


# Backwards compatibility and convenience
CS = CoordinateSystem  # Alias for shorter code

__all__ = [
    # Core classes and enums
    "CoordinateSystem",
    "CoordinateSystemError",
    "LocalCoordinateSystem",
    "GeodeticCoord",
    "ProjectedCoord",
    # Transformers
    "CoordinateTransformer",
    "LocalCoordinateTransformer",
    # Utility functions
    "get_pulkovo_zone",
    "disambiguate_pno13",
    "disambiguate_pno16",
    "define_pno_13_system",
    "define_pno_16_system",
    # Short alias
    "CS",
    # Feature flag
    "HAS_PYPROJ",
]
