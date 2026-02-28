from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pywp.mcm import dogleg_angle_rad

RAD2DEG = 180.0 / np.pi


def _make_md_grid(md_start: float, length_m: float, md_step_m: float) -> np.ndarray:
    if length_m <= 1e-9:
        return np.array([md_start], dtype=float)

    md_end = md_start + float(length_m)
    md = np.arange(md_start, md_end, md_step_m, dtype=float)
    if len(md) == 0 or not np.isclose(md[-1], md_end):
        md = np.append(md, md_end)
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
        self._length_m = float(max(length_m, 0.0))
        self.azi_deg = float(azi_deg)
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
        self._length_m = float(max(length_m, 0.0))
        self.inc_deg = float(inc_deg)
        self.azi_deg = float(azi_deg)
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
    ):
        self.inc_from_deg = float(inc_from_deg)
        self.inc_to_deg = float(inc_to_deg)
        self.dls_deg_per_30m = float(max(dls_deg_per_30m, 0.0))
        self.azi_from_deg = float(azi_deg)
        self.azi_to_deg = float(azi_deg if azi_to_deg is None else azi_to_deg)
        self.name = name

    @property
    def length_m(self) -> float:
        if self.dls_deg_per_30m <= 0.0:
            return 0.0
        dogleg_deg = float(
            dogleg_angle_rad(
                self.inc_from_deg,
                self.azi_from_deg,
                self.inc_to_deg,
                self.azi_to_deg,
            )
            * RAD2DEG
        )
        return dogleg_deg / self.dls_deg_per_30m * 30.0

    def generate(self, md_start: float, md_step_m: float) -> pd.DataFrame:
        md = _make_md_grid(md_start, self.length_m, md_step_m)
        if self.length_m <= 1e-9:
            inc = np.array([self.inc_to_deg], dtype=float)
            azi = np.array([self.azi_to_deg], dtype=float)
        else:
            t = (md - md_start) / self.length_m
            direction_from = _direction_vector(inc_deg=self.inc_from_deg, azi_deg=self.azi_from_deg)
            direction_to = _direction_vector(inc_deg=self.inc_to_deg, azi_deg=self.azi_to_deg)
            directions = _slerp_directions(direction_from=direction_from, direction_to=direction_to, t=t)
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


def _slerp_directions(direction_from: np.ndarray, direction_to: np.ndarray, t: np.ndarray) -> np.ndarray:
    start = direction_from / np.linalg.norm(direction_from)
    end = direction_to / np.linalg.norm(direction_to)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    theta = float(np.arccos(dot))
    t_vec = np.asarray(t, dtype=float).reshape(-1)

    if theta <= 1e-12:
        vectors = np.repeat(start.reshape(1, 3), len(t_vec), axis=0)
    else:
        sin_theta = np.sin(theta)
        w0 = np.sin((1.0 - t_vec) * theta) / sin_theta
        w1 = np.sin(t_vec * theta) / sin_theta
        vectors = w0[:, None] * start[None, :] + w1[:, None] * end[None, :]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return vectors / norms


def _angles_from_directions(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dirs = np.asarray(directions, dtype=float)
    horizontal = np.hypot(dirs[:, 0], dirs[:, 1])
    inc_deg = np.degrees(np.arctan2(horizontal, dirs[:, 2]))
    azi_deg = np.mod(np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])), 360.0)
    return inc_deg.astype(float), azi_deg.astype(float)
