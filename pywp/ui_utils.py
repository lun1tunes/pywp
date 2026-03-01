from __future__ import annotations

from datetime import datetime
from time import perf_counter

import pandas as pd


def format_distance(value_m: float) -> str:
    if value_m < 1e-6:
        return "< 1e-6 m"
    if value_m < 1e-3:
        return f"{value_m:.2e} m"
    if value_m < 1.0:
        return f"{value_m:.4f} m"
    return f"{value_m:.2f} m"


def arrow_safe_text_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.columns:
        if safe[column].dtype == "object":
            safe[column] = safe[column].map(lambda item: "—" if item is None else str(item))
    return safe


def format_run_log_line(run_started_s: float, message: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_s = perf_counter() - run_started_s
    return f"[{timestamp}] elapsed={elapsed_s:.2f}s | {message}"
