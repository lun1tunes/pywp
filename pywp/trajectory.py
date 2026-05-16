from __future__ import annotations

import pandas as pd

from pywp.segments import Segment


MIN_STATION_MD_INTERVAL_M = 1e-3


class WellTrajectory:
    def __init__(self, segments: list[Segment]):
        self.segments = segments

    def stations(self, md_step_m: float) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        md_start = 0.0
        last_output_md: float | None = None
        for segment in self.segments:
            generated = segment.generate(md_start=md_start, md_step_m=md_step_m)
            generated = _collapse_short_internal_stations(generated)
            if not generated.empty:
                md_start = float(generated["MD_m"].iloc[-1])
            block = generated
            if parts:
                block = block.iloc[1:].copy()
            if last_output_md is not None:
                if block.empty:
                    last_output_md = _collapse_short_boundary_station(
                        parts=parts,
                        generated=generated,
                        last_output_md=last_output_md,
                    )
                else:
                    md_values = block["MD_m"].to_numpy(dtype=float)
                    block = block.loc[
                        md_values > last_output_md + MIN_STATION_MD_INTERVAL_M
                    ].copy()
                    if block.empty:
                        last_output_md = _collapse_short_boundary_station(
                            parts=parts,
                            generated=generated,
                            last_output_md=last_output_md,
                        )
            if not block.empty:
                parts.append(block)
                last_output_md = float(block["MD_m"].iloc[-1])

        if not parts:
            raise ValueError("trajectory contains no stations")

        stations = pd.concat(parts, ignore_index=True)
        return stations


def _collapse_short_internal_stations(generated: pd.DataFrame) -> pd.DataFrame:
    if generated.empty or "MD_m" not in generated.columns:
        return generated

    kept_rows: list[pd.Series] = []
    last_md: float | None = None
    for _, row in generated.iterrows():
        md_value = float(row["MD_m"])
        if last_md is not None and md_value <= last_md + MIN_STATION_MD_INTERVAL_M:
            if md_value >= last_md:
                kept_rows[-1] = row.copy()
                last_md = md_value
            continue
        kept_rows.append(row.copy())
        last_md = md_value

    if len(kept_rows) == len(generated):
        return generated
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _collapse_short_boundary_station(
    *,
    parts: list[pd.DataFrame],
    generated: pd.DataFrame,
    last_output_md: float,
) -> float:
    if not parts or generated.empty:
        return float(last_output_md)
    final_row = generated.iloc[-1]
    final_md = float(final_row["MD_m"])
    if final_md <= float(last_output_md):
        return float(last_output_md)
    if final_md > float(last_output_md) + MIN_STATION_MD_INTERVAL_M:
        return float(last_output_md)

    target = parts[-1]
    row_index = target.index[-1]
    for column in target.columns:
        if column == "segment" or column not in final_row.index:
            continue
        target.loc[row_index, column] = final_row[column]
    return final_md
