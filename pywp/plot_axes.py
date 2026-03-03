from __future__ import annotations

import numpy as np


def equalized_xy_ranges(
    x_values: np.ndarray,
    y_values: np.ndarray,
    pad_fraction: float = 0.08,
) -> tuple[list[float], list[float]]:
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))

    span = max(x_max - x_min, y_max - y_min, 1.0)
    span = span * (1.0 + pad_fraction)
    half = span / 2.0

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    return [x_center - half, x_center + half], [y_center - half, y_center + half]


def equalized_axis_ranges(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    pad_fraction: float = 0.08,
) -> tuple[list[float], list[float], list[float]]:
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))

    span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
    span = span * (1.0 + pad_fraction)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    half = span / 2.0
    x_range = [x_center - half, x_center + half]
    y_range = [y_center - half, y_center + half]
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
