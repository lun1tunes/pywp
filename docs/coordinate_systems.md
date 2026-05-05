# Coordinate Systems for Russian Petroleum Industry

Based on research: *"Преобразования координат между Pulkovo PNO13, PNO16, СК-42 и WGS 84 для нефтегазовых данных"*

## Overview

This module provides comprehensive support for coordinate systems used in Russian oil and gas exploration:

- **WGS84** - Global GPS standard (EPSG:4326)
- **Pulkovo 1942 (СК-42)** - Legacy Soviet datum, ~3m accuracy (EPSG:4284)
- **Pulkovo 1995** - Modernized Soviet datum, ~1m accuracy (EPSG:4200)
- **GSK-2011 (ГСК-2011)** - Current Russian state system (EPSG:7681)
- **Local field systems** - PNO-13, PNO-16 with disambiguation (ZONE vs CM)

## EPSG Codes Reference

### Geographic Systems
| System | EPSG Code | Description | Typical Accuracy |
|--------|-----------|-------------|------------------|
| WGS84 | 4326 | Global geodetic system | GPS standard |
| Pulkovo 1942 | 4284 | СК-42, legacy Soviet datum | ~3 meters |
| Pulkovo 1995 | 4200 | Modernized Pulkovo | ~1 meter |
| GSK-2011 | 7681 | ГСК-2011, current state system | <1 meter |

### Projected Systems - Pulkovo 1995 (Modern)
Per PDF research, Pulkovo 1995 zones for modern fields:
| Zone | EPSG Code | Notes |
|------|-----------|-------|
| Zone 13 | 2472 | CM 75°E (Yamal, Gydan) |
| Zone 18 | 20073 | CM 105°E (Eastern Siberia) |
| CM 39°E | 5043 | Custom meridian (Western fields) |

### Projected Systems (Gauss-Kruger)
| Zone | Central Meridian | EPSG Code |
|------|-------------------|-----------|
| Zone 6 | 33°E | 28406 |
| Zone 7 | 36°E | 28407 |
| Zone 8 | 39°E | 28408 |
| Zone 9 | 42°E | 28409 |
| Zone 10 | 45°E | 28410 |
| Zone 11 | 48°E | 28411 |
| Zone 12 | 51°E | 28412 |
| Zone 13 | 54°E | 28413 |
| Zone 14 | 57°E | 28414 |
| Zone 15 | 60°E | 28415 |
| Zone 16 | 63°E | 28416 |
| Zone 17 | 66°E | 28417 |
| Zone 18 | 69°E | 28418 |
| Zone 19 | 72°E | 28419 |
| Zone 20 | 75°E | 28420 |

## PNO Disambiguation (Critical)

Per PDF research: **PNO13/PNO16 are NOT authorities** - they require disambiguation:
- If `easting > 1,000,000` → ZONE-based system
- If `easting < 1,000,000` → CM (custom meridian)

```python
from pywp import disambiguate_pno16, CoordinateSystem

# Large easting → zone-based
crs = disambiguate_pno16(1_500_000.0)
# Returns: CoordinateSystem.PNO_16_ZONE

# Small easting → custom meridian
crs = disambiguate_pno16(500_000.0)
# Returns: CoordinateSystem.PNO_16_CM
```

## Usage Examples

### Basic Transformations

```python
from pywp import CoordinateSystem, CoordinateTransformer, GeodeticCoord

# Create transformer
transformer = CoordinateTransformer()

# Moscow coordinates in WGS84
wgs84 = GeodeticCoord(lat_deg=55.7558, lon_deg=37.6173)

# Convert to Pulkovo 1942
pulkovo = transformer.transform(
    wgs84,
    from_crs=CoordinateSystem.WGS84,
    to_crs=CoordinateSystem.PULKOVO_1942
)
```

### Local Field Systems

```python
from pywp import LocalCoordinateSystem, LocalCoordinateTransformer

# Define custom oilfield system
field_crs = LocalCoordinateSystem(
    name="MyField-01",
    false_easting_m=12500000.0,  # 12,500 km easting
    false_northing_m=6500000.0,    # 6,500 km northing
    rotation_deg=15.0,             # Grid convergence
    base_system=CoordinateSystem.PULKOVO_1942_ZONE_13
)

# Create transformer
transformer = LocalCoordinateTransformer(field_crs)
```

### Zone Detection

```python
from pywp.coordinate_systems import get_pulkovo_zone

# Automatically select appropriate zone
zone = get_pulkovo_zone(longitude_deg=65.5)  # Returns ZONE_17
```

## Transformation Accuracy

Per PDF datum shift research:

| From | To | Typical Accuracy | Source |
|------|-----|------------------|--------|
| WGS84 | Pulkovo 1942 | ~3 meters | Open baselines |
| WGS84 | Pulkovo 1995 | ~1 meter | Open baselines |
| WGS84 | GSK-2011 | <1 meter | Modern datum |
| Pulkovo 1942 | GSK-2011 | 1-3 meters | Conversion |

**Important**: For engineering projects, use local 7-parameter keys if available from field survey data.

## Dependencies

- `pyproj>=3.0` - Required for datum transformations
- `numpy` - Array operations

## References

1. EPSG Registry: https://epsg.io/
2. PDF Research: *"Преобразования координат между Pulkovo PNO13, PNO16, СК-42 и WGS 84"*
3. Russian Federal Law on State Coordinate Systems
4. GOST 32453-2017 (ГСК-2011 specification)
5. PROJ documentation: https://proj.org/
