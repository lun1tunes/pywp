"""Georeferenced Point3D with coordinate system support.

Extends the basic Point3D to include coordinate system information
for proper transformation between different reference frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pywp.coordinate_systems import CoordinateSystem

from pywp.models import Point3D


@dataclass(frozen=True)
class GeoPoint:
    """Point with explicit coordinate system reference.

    Combines 3D coordinates with coordinate system information for
    proper georeferencing in petroleum applications.

    Attributes:
        x: X-coordinate (easting for projected, longitude for geodetic)
        y: Y-coordinate (northing for projected, latitude for geodetic)
        z: Z-coordinate (elevation or depth, typically meters)
        crs: Coordinate reference system
    """

    x: float
    y: float
    z: float
    crs: CoordinateSystem

    def to_point3d(self) -> Point3D:
        """Convert to basic Point3D (loses CRS info)."""
        return Point3D(x=self.x, y=self.y, z=self.z)

    @classmethod
    def from_point3d(cls, point: Point3D, crs: CoordinateSystem) -> GeoPoint:
        """Create GeoPoint from Point3D with CRS."""
        return cls(x=point.x, y=point.y, z=point.z, crs=crs)

    def with_z(self, new_z: float) -> GeoPoint:
        """Return new point with modified Z coordinate."""
        return GeoPoint(x=self.x, y=self.y, z=new_z, crs=self.crs)

    def __repr__(self) -> str:
        return f"GeoPoint({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, {self.crs.name})"


# Well-known coordinate reference systems for petroleum industry
class KnownCRSs:
    """Commonly used CRS configurations for Russian oil & gas fields."""

    @staticmethod
    def wgs84() -> CoordinateSystem:
        from pywp.coordinate_systems import CoordinateSystem
        return CoordinateSystem.WGS84

    @staticmethod
    def pulkovo1942() -> CoordinateSystem:
        from pywp.coordinate_systems import CoordinateSystem
        return CoordinateSystem.PULKOVO_1942

    @staticmethod
    def gsk2011() -> CoordinateSystem:
        from pywp.coordinate_systems import CoordinateSystem
        return CoordinateSystem.GSK_2011

    @staticmethod
    def pulkovo_zone(longitude_deg: float) -> CoordinateSystem:
        """Get appropriate Pulkovo zone for longitude."""
        from pywp.coordinate_systems import get_pulkovo_zone
        return get_pulkovo_zone(longitude_deg)
