from __future__ import annotations

import numpy as np


def equalized_axis_ranges(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    pad_fraction: float = 0.08,
) -> tuple[list[float], list[float], list[float]]:
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))

    xy_min = min(x_min, y_min)
    xy_max = max(x_max, y_max)
    xy_span = xy_max - xy_min

    span = max(xy_span, z_max - z_min, 1.0)
    span = span * (1.0 + pad_fraction)

    xy_center = (xy_min + xy_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    half = span / 2.0
    x_range = [xy_center - half, xy_center + half]
    y_range = [xy_center - half, xy_center + half]
    # Reverse Z to keep depth increasing downward while maintaining equal scale.
    z_range = [z_center + half, z_center - half]
    return x_range, y_range, z_range


def nice_tick_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0.0:
        return 1.0
    raw = span / max(target_ticks, 1)
    exponent = np.floor(np.log10(raw))
    base = 10.0**exponent
    fraction = raw / base
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * base)


def linear_tick_values(axis_range: list[float], step: float) -> list[float]:
    lo = min(axis_range[0], axis_range[1])
    hi = max(axis_range[0], axis_range[1])
    if step <= 0.0:
        return [float(lo), float(hi)]
    first = np.floor(lo / step) * step
    values = np.arange(first, hi + step * 0.5, step, dtype=float)
    if values.size == 0:
        values = np.array([lo, hi], dtype=float)
    return np.round(values, 6).tolist()

