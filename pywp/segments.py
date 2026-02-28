from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


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
        name: str = "BUILD",
    ):
        self.inc_from_deg = float(inc_from_deg)
        self.inc_to_deg = float(inc_to_deg)
        self.dls_deg_per_30m = float(max(dls_deg_per_30m, 0.0))
        self.azi_deg = float(azi_deg)
        self.name = name

    @property
    def length_m(self) -> float:
        if self.dls_deg_per_30m <= 0.0:
            return 0.0
        return abs(self.inc_to_deg - self.inc_from_deg) / self.dls_deg_per_30m * 30.0

    def generate(self, md_start: float, md_step_m: float) -> pd.DataFrame:
        md = _make_md_grid(md_start, self.length_m, md_step_m)
        if self.length_m <= 1e-9:
            inc = np.array([self.inc_to_deg], dtype=float)
        else:
            t = (md - md_start) / self.length_m
            inc = self.inc_from_deg + (self.inc_to_deg - self.inc_from_deg) * t

        return pd.DataFrame(
            {
                "MD_m": md,
                "INC_deg": inc,
                "AZI_deg": np.full_like(md, self.azi_deg, dtype=float),
                "segment": self.name,
            }
        )


class HorizontalSegment(HoldSegment):
    def __init__(self, length_m: float, azi_deg: float, inc_deg: float = 90.0, name: str = "HORIZONTAL"):
        super().__init__(length_m=length_m, inc_deg=inc_deg, azi_deg=azi_deg, name=name)
