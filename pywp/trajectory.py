from __future__ import annotations

import pandas as pd

from pywp.segments import Segment


class WellTrajectory:
    def __init__(self, segments: list[Segment]):
        self.segments = segments

    def stations(self, md_step_m: float) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        md_start = 0.0
        for segment in self.segments:
            block = segment.generate(md_start=md_start, md_step_m=md_step_m)
            if parts:
                block = block.iloc[1:].copy()
            if not block.empty:
                parts.append(block)
                md_start = float(block["MD_m"].iloc[-1])

        if not parts:
            raise ValueError("trajectory contains no stations")

        stations = pd.concat(parts, ignore_index=True)
        stations = stations.drop_duplicates(subset=["MD_m"], keep="last").reset_index(drop=True)
        return stations
