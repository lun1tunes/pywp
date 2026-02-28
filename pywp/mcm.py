from __future__ import annotations

import numpy as np
import pandas as pd

from pywp.models import Point3D

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def wrap_azimuth_deg(azi_deg: np.ndarray | float) -> np.ndarray:
    return np.mod(np.asarray(azi_deg, dtype=float), 360.0)


def dogleg_angle_rad(
    inc1_deg: np.ndarray | float,
    azi1_deg: np.ndarray | float,
    inc2_deg: np.ndarray | float,
    azi2_deg: np.ndarray | float,
) -> np.ndarray:
    i1 = np.asarray(inc1_deg, dtype=float) * DEG2RAD
    i2 = np.asarray(inc2_deg, dtype=float) * DEG2RAD
    a1 = np.asarray(azi1_deg, dtype=float) * DEG2RAD
    a2 = np.asarray(azi2_deg, dtype=float) * DEG2RAD

    cos_beta = np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(a2 - a1)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    return np.arccos(cos_beta)


def ratio_factor(beta_rad: np.ndarray | float, eps: float = 1e-12) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float)
    rf = np.ones_like(beta)
    mask = np.abs(beta) > eps
    rf[mask] = (2.0 / beta[mask]) * np.tan(beta[mask] / 2.0)
    return rf


def dls_deg_per_30m(
    md1_m: np.ndarray | float,
    inc1_deg: np.ndarray | float,
    azi1_deg: np.ndarray | float,
    md2_m: np.ndarray | float,
    inc2_deg: np.ndarray | float,
    azi2_deg: np.ndarray | float,
) -> np.ndarray:
    dmd = np.asarray(md2_m, dtype=float) - np.asarray(md1_m, dtype=float)
    beta_deg = dogleg_angle_rad(inc1_deg, azi1_deg, inc2_deg, azi2_deg) * RAD2DEG
    with np.errstate(divide="ignore", invalid="ignore"):
        dls = np.where(dmd > 0.0, beta_deg * (30.0 / dmd), np.nan)
    return np.asarray(dls, dtype=float)


def minimum_curvature_increment(
    md1_m: float,
    inc1_deg: float,
    azi1_deg: float,
    md2_m: float,
    inc2_deg: float,
    azi2_deg: float,
) -> tuple[float, float, float]:
    dmd = float(md2_m - md1_m)
    if dmd <= 0.0:
        return 0.0, 0.0, 0.0

    i1 = float(inc1_deg) * DEG2RAD
    i2 = float(inc2_deg) * DEG2RAD
    a1 = float(azi1_deg) * DEG2RAD
    a2 = float(azi2_deg) * DEG2RAD

    beta = float(dogleg_angle_rad(inc1_deg, azi1_deg, inc2_deg, azi2_deg))
    rf = float(ratio_factor(np.array([beta]))[0])

    d_n = (dmd / 2.0) * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    d_e = (dmd / 2.0) * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    d_tvd = (dmd / 2.0) * (np.cos(i1) + np.cos(i2)) * rf
    return float(d_n), float(d_e), float(d_tvd)


def compute_positions_min_curv(stations: pd.DataFrame, start: Point3D) -> pd.DataFrame:
    required_cols = {"MD_m", "INC_deg", "AZI_deg"}
    missing = required_cols.difference(stations.columns)
    if missing:
        raise ValueError(f"stations is missing required columns: {sorted(missing)}")

    df = stations.sort_values("MD_m").reset_index(drop=True).copy()
    df["AZI_deg"] = wrap_azimuth_deg(df["AZI_deg"].to_numpy())

    north = [start.y]
    east = [start.x]
    tvd = [start.z]

    for idx in range(1, len(df)):
        d_n, d_e, d_tvd = minimum_curvature_increment(
            md1_m=float(df.loc[idx - 1, "MD_m"]),
            inc1_deg=float(df.loc[idx - 1, "INC_deg"]),
            azi1_deg=float(df.loc[idx - 1, "AZI_deg"]),
            md2_m=float(df.loc[idx, "MD_m"]),
            inc2_deg=float(df.loc[idx, "INC_deg"]),
            azi2_deg=float(df.loc[idx, "AZI_deg"]),
        )
        north.append(north[-1] + d_n)
        east.append(east[-1] + d_e)
        tvd.append(tvd[-1] + d_tvd)

    df["N_m"] = north
    df["E_m"] = east
    df["TVD_m"] = tvd
    df["X_m"] = df["E_m"]
    df["Y_m"] = df["N_m"]
    df["Z_m"] = df["TVD_m"]
    return df


def add_dls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dls = np.full(len(out), np.nan)
    if len(out) > 1:
        dls[1:] = dls_deg_per_30m(
            out["MD_m"].to_numpy()[:-1],
            out["INC_deg"].to_numpy()[:-1],
            out["AZI_deg"].to_numpy()[:-1],
            out["MD_m"].to_numpy()[1:],
            out["INC_deg"].to_numpy()[1:],
            out["AZI_deg"].to_numpy()[1:],
        )
    out["DLS_deg_per_30m"] = dls
    return out
