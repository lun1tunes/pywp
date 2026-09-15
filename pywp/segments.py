from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pywp.constants import RAD2DEG
from pywp.mcm import dogleg_angle_rad

INTERPOLATION_SLERP = "slerp"
INTERPOLATION_RODRIGUES = "rodrigues"
DEFAULT_INTERPOLATION_METHOD = INTERPOLATION_RODRIGUES
MAX_MD_GRID_STATIONS = 1_000_000


def _finite_float(value: object, *, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _non_negative_length(value: object) -> float:
    normalized = _finite_float(value, name="length_m")
    if normalized < 0.0:
        raise ValueError("length_m must be non-negative")
    return normalized


def _inclination_deg(value: object, *, name: str) -> float:
    normalized = _finite_float(value, name=name)
    if not 0.0 <= normalized <= 180.0:
        raise ValueError(f"{name} must be in the range [0, 180] degrees")
    return normalized


def _make_md_grid(md_start: float, length_m: float, md_step_m: float) -> np.ndarray:
    try:
        md_start = float(md_start)
        length_m = float(length_m)
        md_step_m = float(md_step_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("md_start, length_m and md_step_m must be numbers") from exc
    if not all(np.isfinite(value) for value in (md_start, length_m, md_step_m)):
        raise ValueError("md_start, length_m and md_step_m must be finite")
    if length_m < 0.0:
        raise ValueError("length_m must be non-negative")
    if md_step_m <= 0.0:
        raise ValueError("md_step_m must be greater than zero")
    if length_m == 0.0:
        return np.array([md_start], dtype=float)

    md_end = md_start + float(length_m)
    if not np.isfinite(md_end) or md_end <= md_start:
        raise ValueError(
            "md_start + length_m must be a finite value greater than md_start"
        )
    estimated_intervals = length_m / md_step_m
    if not np.isfinite(estimated_intervals):
        raise ValueError(
            f"MD grid would contain more than {MAX_MD_GRID_STATIONS:,} stations"
        )
    estimated_station_count = int(np.ceil(estimated_intervals)) + 1
    if estimated_station_count > MAX_MD_GRID_STATIONS:
        raise ValueError(
            f"MD grid would contain more than {MAX_MD_GRID_STATIONS:,} stations"
        )
    md = np.arange(md_start, md_end, md_step_m, dtype=float)
    md = md[md < md_end]
    md = np.append(md, md_end)
    if len(md) > 1 and np.any(np.diff(md) <= 0.0):
        raise ValueError("md_step_m is too small to produce distinct MD stations")
    return md


class Segment(ABC):
    name: str

    @property
    @abstractmethod
    def length_m(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def generate(self, md_start: float, md_step_m: float) -> pd.DataFrame:
        raise NotImplementedError


class VerticalSegment(Segment):
    def __init__(self, length_m: float, azi_deg: float = 0.0, name: str = "VERTICAL"):
        self._length_m = _non_negative_length(length_m)
        self.azi_deg = _finite_float(azi_deg, name="azi_deg")
        self.name = name

    @property
    def length_m(self) -> float:
        return self._length_m

    def generate(self, md_start: float, md_step_m: float) -> pd.DataFrame:
        md = _make_md_grid(md_start, self.length_m, md_step_m)
        return pd.DataFrame(
            {
                "MD_m": md,
                "INC_deg": np.zeros_like(md),
                "AZI_deg": np.full_like(md, self.azi_deg, dtype=float),
                "segment": self.name,
            }
        )


class HoldSegment(Segment):
    def __init__(self, length_m: float, inc_deg: float, azi_deg: float, name: str = "HOLD"):
        self._length_m = _non_negative_length(length_m)
        self.inc_deg = _inclination_deg(inc_deg, name="inc_deg")
        self.azi_deg = _finite_float(azi_deg, name="azi_deg")
        self.name = name

    @property
    def length_m(self) -> float:
        return self._length_m

    def generate(self, md_start: float, md_step_m: float) -> pd.DataFrame:
        md = _make_md_grid(md_start, self.length_m, md_step_m)
        return pd.DataFrame(
            {
                "MD_m": md,
                "INC_deg": np.full_like(md, self.inc_deg, dtype=float),
                "AZI_deg": np.full_like(md, self.azi_deg, dtype=float),
                "segment": self.name,
            }
        )


class BuildSegment(Segment):
    def __init__(
        self,
        inc_from_deg: float,
        inc_to_deg: float,
        dls_deg_per_30m: float,
        azi_deg: float,
        azi_to_deg: float | None = None,
        name: str = "BUILD",
        interpolation_method: str = DEFAULT_INTERPOLATION_METHOD,
    ):
        self.inc_from_deg = _inclination_deg(inc_from_deg, name="inc_from_deg")
        self.inc_to_deg = _inclination_deg(inc_to_deg, name="inc_to_deg")
        self.dls_deg_per_30m = _finite_float(
            dls_deg_per_30m,
            name="dls_deg_per_30m",
        )
        if self.dls_deg_per_30m < 0.0:
            raise ValueError("dls_deg_per_30m must be non-negative")
        self.azi_from_deg = _finite_float(azi_deg, name="azi_deg")
        self.azi_to_deg = _finite_float(
            azi_deg if azi_to_deg is None else azi_to_deg,
            name="azi_to_deg",
        )
        self.name = name
        self.interpolation_method = str(interpolation_method)

    @property
    def length_m(self) -> float:
        dogleg_deg = float(
            dogleg_angle_rad(
                self.inc_from_deg,
                self.azi_from_deg,
                self.inc_to_deg,
                self.azi_to_deg,
            )
            * RAD2DEG
        )
        if self.dls_deg_per_30m <= 0.0:
            if dogleg_deg > 1e-12:
                raise ValueError(
                    "dls_deg_per_30m must be positive for a non-zero dogleg"
                )
            return 0.0
        return dogleg_deg / self.dls_deg_per_30m * 30.0

    def generate(self, md_start: float, md_step_m: float) -> pd.DataFrame:
        md = _make_md_grid(md_start, self.length_m, md_step_m)
        if self.length_m <= 1e-9:
            inc = np.array([self.inc_to_deg], dtype=float)
            azi = np.array([self.azi_to_deg], dtype=float)
        else:
            t = (md - md_start) / self.length_m
            t[0] = 0.0
            t[-1] = 1.0
            direction_from = _direction_vector(inc_deg=self.inc_from_deg, azi_deg=self.azi_from_deg)
            direction_to = _direction_vector(inc_deg=self.inc_to_deg, azi_deg=self.azi_to_deg)
            if self.interpolation_method == INTERPOLATION_SLERP:
                directions = _slerp_directions(direction_from=direction_from, direction_to=direction_to, t=t)
            elif self.interpolation_method == INTERPOLATION_RODRIGUES:
                directions = _rodrigues_directions(direction_from=direction_from, direction_to=direction_to, t=t)
            else:
                raise ValueError(
                    f"Unknown interpolation_method: {self.interpolation_method!r}. "
                    f"Expected {INTERPOLATION_SLERP!r} or {INTERPOLATION_RODRIGUES!r}."
                )
            inc, azi = _angles_from_directions(directions=directions)

        return pd.DataFrame(
            {
                "MD_m": md,
                "INC_deg": inc,
                "AZI_deg": azi,
                "segment": self.name,
            }
        )


class HorizontalSegment(HoldSegment):
    def __init__(self, length_m: float, azi_deg: float, inc_deg: float = 90.0, name: str = "HORIZONTAL"):
        super().__init__(length_m=length_m, inc_deg=inc_deg, azi_deg=azi_deg, name=name)


def _direction_vector(inc_deg: float, azi_deg: float) -> np.ndarray:
    inc_rad = np.radians(float(inc_deg))
    azi_rad = np.radians(float(azi_deg))
    return np.array(
        [
            np.sin(inc_rad) * np.cos(azi_rad),
            np.sin(inc_rad) * np.sin(azi_rad),
            np.cos(inc_rad),
        ],
        dtype=float,
    )


def _unit_direction(direction: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError("Direction must be a finite three-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Direction vector must be non-zero")
    return vector / norm


def _deterministic_orthogonal_axis(direction: np.ndarray) -> np.ndarray:
    """Return a stable rotation axis perpendicular to ``direction``.

    An exactly antipodal pair of directions has infinitely many shortest
    rotation planes.  Selecting the least-aligned Cartesian basis vector makes
    that otherwise ambiguous case deterministic and avoids silently returning
    a constant (and therefore wrong) direction.
    """
    unit = _unit_direction(direction)
    basis = np.eye(3, dtype=float)[int(np.argmin(np.abs(unit)))]
    axis = np.cross(unit, basis)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:  # Defensive; the least-aligned basis cannot be parallel.
        raise ValueError("Unable to construct an orthogonal interpolation axis")
    return axis / axis_norm


def _rodrigues_directions(direction_from: np.ndarray, direction_to: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Interpolate directions using Rodrigues' rotation formula.

    Numerically superior to SLERP when the dogleg angle approaches pi,
    because the evaluation never divides by sin(dogleg) per-point — only
    a one-time cross-product normalisation is needed.
    """
    start = _unit_direction(direction_from)
    end = _unit_direction(direction_to)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    theta = float(np.arccos(dot))
    t_vec = np.asarray(t, dtype=float).reshape(-1)
    if not np.all(np.isfinite(t_vec)) or np.any((t_vec < 0.0) | (t_vec > 1.0)):
        raise ValueError("Interpolation parameters must be finite and in [0, 1]")

    if theta <= 1e-12:
        vectors = np.repeat(start.reshape(1, 3), len(t_vec), axis=0)
    else:
        cross = np.cross(start, end)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-10:
            if dot > 0.0:
                vectors = np.repeat(start.reshape(1, 3), len(t_vec), axis=0)
            else:
                # Exactly antipodal directions need a deterministic choice of
                # rotation plane.  A constant fallback violates the endpoint.
                rot_axis = _deterministic_orthogonal_axis(start)
                in_plane = np.cross(rot_axis, start)
                angles = (t_vec * theta).reshape(-1, 1)
                vectors = (
                    np.cos(angles) * start[None, :]
                    + np.sin(angles) * in_plane[None, :]
                )
        else:
            rot_axis = cross / cross_norm
            in_plane = np.cross(rot_axis, start)
            angles = (t_vec * theta).reshape(-1, 1)
            vectors = np.cos(angles) * start[None, :] + np.sin(angles) * in_plane[None, :]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    vectors = vectors / norms
    # Preserve exact endpoints even when a nearly-antipodal pair is routed
    # through the deterministic fallback axis.
    vectors[t_vec == 0.0] = start
    vectors[t_vec == 1.0] = end
    return vectors


def _slerp_directions(direction_from: np.ndarray, direction_to: np.ndarray, t: np.ndarray) -> np.ndarray:
    start = _unit_direction(direction_from)
    end = _unit_direction(direction_to)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    theta = float(np.arccos(dot))
    t_vec = np.asarray(t, dtype=float).reshape(-1)
    if not np.all(np.isfinite(t_vec)) or np.any((t_vec < 0.0) | (t_vec > 1.0)):
        raise ValueError("Interpolation parameters must be finite and in [0, 1]")

    if theta <= 1e-12:
        vectors = np.repeat(start.reshape(1, 3), len(t_vec), axis=0)
    else:
        sin_theta = np.sin(theta)
        if abs(float(sin_theta)) <= 1e-10:
            if dot > 0.0:
                vectors = np.repeat(start.reshape(1, 3), len(t_vec), axis=0)
            else:
                rot_axis = _deterministic_orthogonal_axis(start)
                in_plane = np.cross(rot_axis, start)
                angles = (t_vec * theta).reshape(-1, 1)
                vectors = (
                    np.cos(angles) * start[None, :]
                    + np.sin(angles) * in_plane[None, :]
                )
        else:
            w0 = np.sin((1.0 - t_vec) * theta) / sin_theta
            w1 = np.sin(t_vec * theta) / sin_theta
            vectors = w0[:, None] * start[None, :] + w1[:, None] * end[None, :]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    vectors = vectors / norms
    vectors[t_vec == 0.0] = start
    vectors[t_vec == 1.0] = end
    return vectors


def _angles_from_directions(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dirs = np.asarray(directions, dtype=float)
    horizontal = np.hypot(dirs[:, 0], dirs[:, 1])
    inc_deg = np.degrees(np.arctan2(horizontal, dirs[:, 2]))
    azi_deg = np.mod(np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])), 360.0)
    return inc_deg.astype(float), azi_deg.astype(float)
