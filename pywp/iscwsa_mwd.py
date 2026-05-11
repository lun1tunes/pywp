from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from pywp.constants import DEG2RAD
from pywp.mcm import minimum_curvature_increment

IscwsaVector = Literal["e", "s", "i", "a", "l"]
IscwsaPropagation = Literal["random", "systematic", "global"]


@dataclass(frozen=True)
class IscwsaMwdEnvironment:
    gtot_mps2: float = 9.80665
    mtot_nt: float = 50_000.0
    dip_deg: float = 82.35
    declination_deg: float = 0.0
    lateral_singularity_inc_deg: float = 0.01


@dataclass(frozen=True)
class IscwsaMwdToolCodeComponent:
    name: str
    vector: IscwsaVector
    propagation: IscwsaPropagation
    value_1sigma: float
    unit: str
    formula: str


@dataclass(frozen=True)
class IscwsaMwdToolCode:
    name: str
    label: str
    components: tuple[IscwsaMwdToolCodeComponent, ...]


@dataclass(frozen=True)
class IscwsaCovarianceResult:
    covariance_xyz: np.ndarray
    covariance_xyz_random: np.ndarray
    covariance_xyz_systematic: np.ndarray
    covariance_xyz_global: np.ndarray
    global_source_vectors_xyz: tuple[tuple[str, np.ndarray], ...] = ()


ISCWSA_MWD_POOR_MAGNETIC = "iscwsa_mwd_poor_magnetic"
ISCWSA_MWD_UNKNOWN_MAGNETIC = "iscwsa_mwd_unknown_magnetic"
DEFAULT_ISCWSA_MWD_ENVIRONMENT = IscwsaMwdEnvironment()


MWD_POOR_MAGNETIC_TOOL_CODE = IscwsaMwdToolCode(
    name=ISCWSA_MWD_POOR_MAGNETIC,
    label="ISCWSA MWD POOR Magnetic",
    components=(
        IscwsaMwdToolCodeComponent("drfr_e", "e", "random", 0.35, "m", "one"),
        IscwsaMwdToolCodeComponent("drfr_s", "s", "random", 2.2, "m", "one"),
        IscwsaMwdToolCodeComponent("drfs", "s", "systematic", 1.0, "m", "one"),
        IscwsaMwdToolCodeComponent("dsfs", "e", "systematic", 0.00056, "-", "tmd"),
        IscwsaMwdToolCodeComponent("dstg", "e", "global", 0.00000025, "1/m", "tmd_tvd"),
        IscwsaMwdToolCodeComponent("xym1", "i", "systematic", 0.1, "deg", "sin_inc"),
        IscwsaMwdToolCodeComponent("xym2", "l", "systematic", 0.1, "deg", "sin_inc"),
        IscwsaMwdToolCodeComponent(
            "xym3", "i", "systematic", 0.1, "deg", "cos_azi_abs_cos_inc"
        ),
        IscwsaMwdToolCodeComponent(
            "xym3", "l", "systematic", 0.1, "deg", "neg_sin_azi_abs_cos_inc"
        ),
        IscwsaMwdToolCodeComponent(
            "xym4", "i", "systematic", 0.1, "deg", "sin_azi_abs_cos_inc"
        ),
        IscwsaMwdToolCodeComponent(
            "xym4", "l", "systematic", 0.1, "deg", "cos_azi_abs_cos_inc"
        ),
        IscwsaMwdToolCodeComponent("sag", "i", "systematic", 0.2, "deg", "sin_inc"),
        IscwsaMwdToolCodeComponent("decg", "a", "global", 0.36, "deg", "one"),
        IscwsaMwdToolCodeComponent(
            "dbhg", "a", "global", 5000.0, "deg_nt", "inv_mtot_cos_dip"
        ),
        IscwsaMwdToolCodeComponent("amil", "a", "systematic", 300.0, "nt", "amil"),
        IscwsaMwdToolCodeComponent(
            "abxy_ti1", "i", "systematic", 0.008, "m/s2", "abxy_ti1_i"
        ),
        IscwsaMwdToolCodeComponent(
            "abxy_ti1", "a", "systematic", 0.008, "m/s2", "abxy_ti1_a"
        ),
        IscwsaMwdToolCodeComponent(
            "abxy_ti2", "l", "systematic", 0.008, "m/s2", "abxy_ti2_l"
        ),
        IscwsaMwdToolCodeComponent("abz", "i", "systematic", 0.008, "m/s2", "abz_i"),
        IscwsaMwdToolCodeComponent("abz", "a", "systematic", 0.008, "m/s2", "abz_a"),
        IscwsaMwdToolCodeComponent(
            "asxy_ti1", "i", "systematic", 0.0005, "-", "asxy_ti1_i"
        ),
        IscwsaMwdToolCodeComponent(
            "asxy_ti1", "a", "systematic", 0.0005, "-", "asxy_ti1_a"
        ),
        IscwsaMwdToolCodeComponent(
            "asxy_ti2", "i", "systematic", 0.0005, "-", "asxy_ti2_i"
        ),
        IscwsaMwdToolCodeComponent(
            "asxy_ti2", "a", "systematic", 0.0005, "-", "asxy_ti2_a"
        ),
        IscwsaMwdToolCodeComponent(
            "asxy_ti3", "a", "systematic", 0.0005, "-", "asxy_ti3_a"
        ),
        IscwsaMwdToolCodeComponent("asz", "i", "systematic", 0.0005, "-", "asz_i"),
        IscwsaMwdToolCodeComponent("asz", "a", "systematic", 0.0005, "-", "asz_a"),
        IscwsaMwdToolCodeComponent(
            "mbxy_ti1", "a", "systematic", 70.0, "nt", "mbxy_ti1_a"
        ),
        IscwsaMwdToolCodeComponent(
            "mbxy_ti2", "a", "systematic", 70.0, "nt", "mbxy_ti2_a"
        ),
        IscwsaMwdToolCodeComponent("mbz", "a", "systematic", 70.0, "nt", "mbz_a"),
        IscwsaMwdToolCodeComponent(
            "msxy_ti1", "a", "systematic", 0.0016, "-", "msxy_ti1_a"
        ),
        IscwsaMwdToolCodeComponent(
            "msxy_ti2", "a", "systematic", 0.0016, "-", "msxy_ti2_a"
        ),
        IscwsaMwdToolCodeComponent(
            "msxy_ti3", "a", "systematic", 0.0016, "-", "msxy_ti3_a"
        ),
        IscwsaMwdToolCodeComponent("msz", "a", "systematic", 0.0016, "-", "msz_a"),
    ),
)

def _mwd_unknown_magnetic_components() -> tuple[IscwsaMwdToolCodeComponent, ...]:
    components: list[IscwsaMwdToolCodeComponent] = []
    for component in MWD_POOR_MAGNETIC_TOOL_CODE.components:
        if component.name in {"xym1", "xym2", "xym3", "xym4"}:
            components.append(replace(component, value_1sigma=0.32))
        else:
            components.append(component)
    components.extend(
        (
            IscwsaMwdToolCodeComponent("amic", "a", "systematic", 0.5, "deg", "one"),
            IscwsaMwdToolCodeComponent(
                "amid", "a", "systematic", 1.5, "deg", "sin_inc_sin_azm"
            ),
            IscwsaMwdToolCodeComponent("azmr", "a", "random", 0.5, "deg", "one"),
            IscwsaMwdToolCodeComponent("azms", "a", "systematic", 1.1, "deg", "one"),
        )
    )
    return tuple(components)


MWD_UNKNOWN_MAGNETIC_TOOL_CODE = IscwsaMwdToolCode(
    name=ISCWSA_MWD_UNKNOWN_MAGNETIC,
    label="ISCWSA MWD Unknown Magnetic",
    components=_mwd_unknown_magnetic_components(),
)

ISCWSA_MWD_TOOL_CODES: dict[str, IscwsaMwdToolCode] = {
    MWD_POOR_MAGNETIC_TOOL_CODE.name: MWD_POOR_MAGNETIC_TOOL_CODE,
    MWD_UNKNOWN_MAGNETIC_TOOL_CODE.name: MWD_UNKNOWN_MAGNETIC_TOOL_CODE,
}


def iscwsa_mwd_covariance_xyz(
    *,
    md_m: np.ndarray,
    inc_deg: np.ndarray,
    azi_deg: np.ndarray,
    tvd_m: np.ndarray | None = None,
    tool_code: str = ISCWSA_MWD_POOR_MAGNETIC,
    environment: IscwsaMwdEnvironment = DEFAULT_ISCWSA_MWD_ENVIRONMENT,
) -> IscwsaCovarianceResult:
    md_values, inc_values_deg, azi_values_deg = _validated_survey_arrays(
        md_m=md_m,
        inc_deg=inc_deg,
        azi_deg=azi_deg,
    )
    tvd_values = _tvd_values_for_formulas(
        md_values=md_values,
        inc_values_deg=inc_values_deg,
        azi_values_deg=azi_values_deg,
        tvd_m=tvd_m,
    )
    model = ISCWSA_MWD_TOOL_CODES[str(tool_code)]
    random_nev = np.zeros((len(md_values), 3, 3), dtype=float)
    systematic_nev = np.zeros_like(random_nev)
    global_nev = np.zeros_like(random_nev)
    global_source_vectors_nev: dict[str, np.ndarray] = {}

    for (source_name, _), components in _components_by_source(model).items():
        e_dia, e_lateral = _e_dia_for_source(
            components=components,
            md_values=md_values,
            inc_values_deg=inc_values_deg,
            azi_values_deg=azi_values_deg,
            tvd_values=tvd_values,
            environment=environment,
        )
        star_nev, carry_nev = _position_error_vectors_nev(
            md_values=md_values,
            inc_values_deg=inc_values_deg,
            azi_values_deg=azi_values_deg,
            e_dia=e_dia,
            e_lateral=e_lateral,
        )
        propagation = components[0].propagation
        if propagation == "random":
            random_nev += _random_covariance_from_vectors(star_nev, carry_nev)
        elif propagation == "global":
            source_vectors = _systematic_vectors_from_vectors(star_nev, carry_nev)
            global_source_vectors_nev[str(source_name)] = source_vectors
            global_nev += _covariance_from_vectors(source_vectors)
        else:
            systematic_nev += _systematic_covariance_from_vectors(star_nev, carry_nev)

    random_xyz = _covariance_nev_to_xyz(random_nev)
    systematic_xyz = _covariance_nev_to_xyz(systematic_nev)
    global_xyz = _covariance_nev_to_xyz(global_nev)
    total_xyz = _symmetrized_covariance(random_xyz + systematic_xyz + global_xyz)
    return IscwsaCovarianceResult(
        covariance_xyz=total_xyz,
        covariance_xyz_random=_symmetrized_covariance(random_xyz),
        covariance_xyz_systematic=_symmetrized_covariance(systematic_xyz),
        covariance_xyz_global=_symmetrized_covariance(global_xyz),
        global_source_vectors_xyz=tuple(
            (source_name, _vectors_nev_to_xyz(vectors_nev))
            for source_name, vectors_nev in sorted(global_source_vectors_nev.items())
        ),
    )


def _validated_survey_arrays(
    *,
    md_m: np.ndarray,
    inc_deg: np.ndarray,
    azi_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    md_values, inc_values, azi_values = np.broadcast_arrays(
        np.asarray(md_m, dtype=float),
        np.asarray(inc_deg, dtype=float),
        np.asarray(azi_deg, dtype=float),
    )
    md_values = np.asarray(md_values, dtype=float).reshape(-1)
    inc_values = np.asarray(inc_values, dtype=float).reshape(-1)
    azi_values = np.asarray(azi_values, dtype=float).reshape(-1)
    if len(md_values) == 0:
        return md_values, inc_values, azi_values
    if not (
        np.all(np.isfinite(md_values))
        and np.all(np.isfinite(inc_values))
        and np.all(np.isfinite(azi_values))
    ):
        raise ValueError("ISCWSA MWD covariance requires finite MD/INC/AZI values.")
    if len(md_values) > 1 and np.any(np.diff(md_values) <= 0.0):
        raise ValueError(
            "ISCWSA MWD covariance requires strictly increasing MD values."
        )
    if np.any((inc_values < 0.0) | (inc_values > 180.0)):
        raise ValueError("ISCWSA MWD covariance requires INC within [0, 180].")
    return md_values, inc_values, np.mod(azi_values, 360.0)


def _tvd_values_for_formulas(
    *,
    md_values: np.ndarray,
    inc_values_deg: np.ndarray,
    azi_values_deg: np.ndarray,
    tvd_m: np.ndarray | None,
) -> np.ndarray:
    if tvd_m is not None:
        tvd_values = np.asarray(tvd_m, dtype=float).reshape(-1)
        if len(tvd_values) != len(md_values) or not np.all(np.isfinite(tvd_values)):
            raise ValueError("tvd_m must match MD length and contain finite values.")
        return tvd_values
    if len(md_values) == 0:
        return np.asarray([], dtype=float)
    tvd_values = [0.0]
    for index in range(1, len(md_values)):
        _, _, d_tvd = minimum_curvature_increment(
            md1_m=float(md_values[index - 1]),
            inc1_deg=float(inc_values_deg[index - 1]),
            azi1_deg=float(azi_values_deg[index - 1]),
            md2_m=float(md_values[index]),
            inc2_deg=float(inc_values_deg[index]),
            azi2_deg=float(azi_values_deg[index]),
        )
        tvd_values.append(float(tvd_values[-1] + d_tvd))
    return np.asarray(tvd_values, dtype=float)


def _components_by_source(
    tool_code: IscwsaMwdToolCode,
) -> dict[tuple[str, IscwsaPropagation], tuple[IscwsaMwdToolCodeComponent, ...]]:
    groups: defaultdict[
        tuple[str, IscwsaPropagation], list[IscwsaMwdToolCodeComponent]
    ] = defaultdict(list)
    for component in tool_code.components:
        groups[(component.name, component.propagation)].append(component)
    return {key: tuple(value) for key, value in groups.items()}


def _e_dia_for_source(
    *,
    components: tuple[IscwsaMwdToolCodeComponent, ...],
    md_values: np.ndarray,
    inc_values_deg: np.ndarray,
    azi_values_deg: np.ndarray,
    tvd_values: np.ndarray,
    environment: IscwsaMwdEnvironment,
) -> tuple[np.ndarray, np.ndarray]:
    inc_rad = inc_values_deg * DEG2RAD
    sin_inc = np.sin(inc_rad)
    e_dia = np.zeros((len(md_values), 3), dtype=float)
    e_lateral = np.zeros(len(md_values), dtype=float)
    lateral_threshold = max(
        abs(float(np.sin(float(environment.lateral_singularity_inc_deg) * DEG2RAD))),
        1e-9,
    )
    for component in components:
        error_value = _component_sigma(component)
        weighted = error_value * _formula_weight(
            formula=component.formula,
            md_values=md_values,
            inc_values_deg=inc_values_deg,
            azi_values_deg=azi_values_deg,
            tvd_values=tvd_values,
            environment=environment,
        )
        if component.vector in {"e", "s"}:
            e_dia[:, 0] += weighted
        elif component.vector == "i":
            e_dia[:, 1] += weighted
        elif component.vector == "a":
            e_dia[:, 2] += weighted
        elif component.vector == "l":
            near_vertical = np.abs(sin_inc) < lateral_threshold
            if np.any(~near_vertical):
                e_dia[~near_vertical, 2] += (
                    weighted[~near_vertical] / sin_inc[~near_vertical]
                )
            if np.any(near_vertical):
                e_lateral[near_vertical] += weighted[near_vertical]
    e_dia[0] = 0.0
    e_lateral[0] = 0.0
    return e_dia, e_lateral


def _component_sigma(component: IscwsaMwdToolCodeComponent) -> float:
    value = float(component.value_1sigma)
    unit = component.unit.lower()
    if unit in {"deg", "d"}:
        return value * DEG2RAD
    if unit in {"deg_nt", "dnt"}:
        return value * DEG2RAD
    return value


def _formula_weight(
    *,
    formula: str,
    md_values: np.ndarray,
    inc_values_deg: np.ndarray,
    azi_values_deg: np.ndarray,
    tvd_values: np.ndarray,
    environment: IscwsaMwdEnvironment,
) -> np.ndarray:
    inc = inc_values_deg * DEG2RAD
    azi = azi_values_deg * DEG2RAD
    azm = (azi_values_deg - float(environment.declination_deg)) * DEG2RAD
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)
    abs_cos_inc = np.sqrt(np.clip(1.0 - sin_inc * sin_inc, 0.0, None))
    sin_azi = np.sin(azi)
    cos_azi = np.cos(azi)
    sin_azm = np.sin(azm)
    cos_azm = np.cos(azm)
    tan_dip = np.tan(float(environment.dip_deg) * DEG2RAD)
    gtot = float(environment.gtot_mps2)
    mtot_cos_dip = _safe_mtot_cos_dip(environment)

    if formula == "one":
        return np.ones_like(md_values)
    if formula == "tmd":
        return md_values
    if formula == "tmd_tvd":
        return md_values * tvd_values
    if formula == "sin_inc":
        return sin_inc
    if formula == "cos_azi_abs_cos_inc":
        return cos_azi * abs_cos_inc
    if formula == "neg_sin_azi_abs_cos_inc":
        return -sin_azi * abs_cos_inc
    if formula == "sin_azi_abs_cos_inc":
        return sin_azi * abs_cos_inc
    if formula == "inv_mtot_cos_dip":
        return np.full_like(md_values, 1.0 / mtot_cos_dip)
    if formula == "amil":
        return sin_inc * sin_azm / mtot_cos_dip
    if formula == "sin_inc_sin_azm":
        return sin_inc * sin_azm
    if formula == "abxy_ti1_i":
        return -cos_inc / gtot
    if formula == "abxy_ti1_a":
        return tan_dip * cos_inc * sin_azm / gtot
    if formula == "abxy_ti2_l":
        return (cos_inc - tan_dip * cos_azm * sin_inc) / gtot
    if formula == "abz_i":
        return -sin_inc / gtot
    if formula == "abz_a":
        return tan_dip * sin_inc * sin_azm / gtot
    if formula == "asxy_ti1_i":
        return sin_inc * cos_inc / np.sqrt(2.0)
    if formula == "asxy_ti1_a":
        return -(tan_dip * sin_inc * cos_inc * sin_azm) / np.sqrt(2.0)
    if formula == "asxy_ti2_i":
        return sin_inc * cos_inc / 2.0
    if formula == "asxy_ti2_a":
        return -tan_dip * sin_inc * cos_inc * sin_azm / 2.0
    if formula == "asxy_ti3_a":
        return (tan_dip * sin_inc * cos_azm - cos_inc) / 2.0
    if formula == "asz_i":
        return -sin_inc * cos_inc
    if formula == "asz_a":
        return tan_dip * sin_inc * cos_inc * sin_azm
    if formula == "mbxy_ti1_a":
        return -(cos_inc * sin_azm) / mtot_cos_dip
    if formula == "mbxy_ti2_a":
        return cos_azm / mtot_cos_dip
    if formula == "mbz_a":
        return -sin_inc * sin_azm / mtot_cos_dip
    if formula == "msxy_ti1_a":
        return (sin_inc * sin_azm * tan_dip * cos_inc + sin_inc * cos_azm) / np.sqrt(
            2.0
        )
    if formula == "msxy_ti2_a":
        return (
            sin_azm
            * (tan_dip * sin_inc * cos_inc - cos_inc * cos_inc * cos_azm - cos_azm)
            / 2.0
        )
    if formula == "msxy_ti3_a":
        return (
            cos_inc * cos_inc * cos_azm * cos_azm
            - cos_inc * sin_azm * sin_azm
            - tan_dip * sin_inc * cos_azm
        ) / 2.0
    if formula == "msz_a":
        return -(sin_inc * cos_azm + tan_dip * cos_inc) * sin_inc * sin_azm
    raise ValueError(f"Unsupported ISCWSA MWD formula: {formula!r}")


def _safe_mtot_cos_dip(environment: IscwsaMwdEnvironment) -> float:
    value = float(environment.mtot_nt) * float(
        np.cos(float(environment.dip_deg) * DEG2RAD)
    )
    if abs(value) <= 1e-9:
        raise ValueError("ISCWSA MWD environment has mtot*cos(dip) too close to zero.")
    return value


def _position_error_vectors_nev(
    *,
    md_values: np.ndarray,
    inc_values_deg: np.ndarray,
    azi_values_deg: np.ndarray,
    e_dia: np.ndarray,
    e_lateral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(md_values)
    star = np.zeros((count, 3), dtype=float)
    carry = np.zeros((count, 3), dtype=float)
    if count <= 1:
        return star, carry
    drk, drkplus1 = _balanced_tangential_jacobians_nev(
        md_values=md_values,
        inc_values_deg=inc_values_deg,
        azi_values_deg=azi_values_deg,
    )
    star = np.einsum("nij,nj->ni", drk, e_dia)
    carry = np.einsum("nij,nj->ni", drk + drkplus1, e_dia)
    lateral_star, lateral_carry = _near_vertical_lateral_vectors_nev(
        md_values=md_values,
        azi_values_deg=azi_values_deg,
        e_lateral=e_lateral,
    )
    star += lateral_star
    carry += lateral_carry
    return star, carry


def _balanced_tangential_jacobians_nev(
    *,
    md_values: np.ndarray,
    inc_values_deg: np.ndarray,
    azi_values_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(md_values)
    drk = np.zeros((count, 3, 3), dtype=float)
    drkplus1 = np.zeros_like(drk)
    if count <= 1:
        return drk, drkplus1

    md = np.asarray(md_values, dtype=float)
    inc = np.asarray(inc_values_deg, dtype=float) * DEG2RAD
    azi = np.asarray(azi_values_deg, dtype=float) * DEG2RAD
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)
    sin_azi = np.sin(azi)
    cos_azi = np.cos(azi)
    dmd = np.diff(md)
    half_dmd = 0.5 * dmd

    drk[1:, 0, 0] = 0.5 * (sin_inc[:-1] * cos_azi[:-1] + sin_inc[1:] * cos_azi[1:])
    drk[1:, 1, 0] = 0.5 * (sin_inc[:-1] * sin_azi[:-1] + sin_inc[1:] * sin_azi[1:])
    drk[1:, 2, 0] = 0.5 * (cos_inc[:-1] + cos_inc[1:])

    drk[1:, 0, 1] = half_dmd * cos_inc[1:] * cos_azi[1:]
    drk[1:, 1, 1] = half_dmd * cos_inc[1:] * sin_azi[1:]
    drk[1:, 2, 1] = -half_dmd * sin_inc[1:]

    drk[1:, 0, 2] = -half_dmd * sin_inc[1:] * sin_azi[1:]
    drk[1:, 1, 2] = half_dmd * sin_inc[1:] * cos_azi[1:]

    drkplus1[:-1, 0, 0] = -0.5 * (
        sin_inc[:-1] * cos_azi[:-1] + sin_inc[1:] * cos_azi[1:]
    )
    drkplus1[:-1, 1, 0] = -0.5 * (
        sin_inc[:-1] * sin_azi[:-1] + sin_inc[1:] * sin_azi[1:]
    )
    drkplus1[:-1, 2, 0] = -0.5 * (cos_inc[:-1] + cos_inc[1:])

    drkplus1[:-1, 0, 1] = half_dmd * cos_inc[:-1] * cos_azi[:-1]
    drkplus1[:-1, 1, 1] = half_dmd * cos_inc[:-1] * sin_azi[:-1]
    drkplus1[:-1, 2, 1] = -half_dmd * sin_inc[:-1]

    drkplus1[:-1, 0, 2] = -half_dmd * sin_inc[:-1] * sin_azi[:-1]
    drkplus1[:-1, 1, 2] = half_dmd * sin_inc[:-1] * cos_azi[:-1]
    return drk, drkplus1


def _near_vertical_lateral_vectors_nev(
    *,
    md_values: np.ndarray,
    azi_values_deg: np.ndarray,
    e_lateral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(md_values)
    star = np.zeros((count, 3), dtype=float)
    carry = np.zeros((count, 3), dtype=float)
    if count <= 1 or not np.any(np.abs(e_lateral) > 1e-18):
        return star, carry

    md = np.asarray(md_values, dtype=float)
    azi = np.asarray(azi_values_deg, dtype=float) * DEG2RAD
    lateral_direction = np.column_stack(
        [
            -np.sin(azi),
            np.cos(azi),
            np.zeros(count, dtype=float),
        ]
    )
    previous_half_dmd = np.zeros(count, dtype=float)
    previous_half_dmd[1:] = 0.5 * np.diff(md)
    next_half_dmd = np.zeros(count, dtype=float)
    next_half_dmd[:-1] = 0.5 * np.diff(md)
    lateral = np.asarray(e_lateral, dtype=float)
    star = previous_half_dmd[:, None] * lateral[:, None] * lateral_direction
    carry = (
        (previous_half_dmd + next_half_dmd)[:, None]
        * lateral[:, None]
        * lateral_direction
    )
    return star, carry


def _random_covariance_from_vectors(
    star_nev: np.ndarray, carry_nev: np.ndarray
) -> np.ndarray:
    covariances = np.zeros((len(star_nev), 3, 3), dtype=float)
    running = np.zeros((3, 3), dtype=float)
    for index in range(len(star_nev)):
        covariances[index] = running + np.outer(star_nev[index], star_nev[index])
        running = running + np.outer(carry_nev[index], carry_nev[index])
    return covariances


def _systematic_covariance_from_vectors(
    star_nev: np.ndarray, carry_nev: np.ndarray
) -> np.ndarray:
    return _covariance_from_vectors(
        _systematic_vectors_from_vectors(star_nev, carry_nev)
    )


def _systematic_vectors_from_vectors(
    star_nev: np.ndarray, carry_nev: np.ndarray
) -> np.ndarray:
    vectors = np.zeros((len(star_nev), 3), dtype=float)
    running = np.zeros(3, dtype=float)
    for index in range(len(star_nev)):
        vector = running + star_nev[index]
        vectors[index] = vector
        running = running + carry_nev[index]
    return vectors


def _covariance_from_vectors(vectors: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", vectors, vectors)


def _covariance_nev_to_xyz(covariance_nev: np.ndarray) -> np.ndarray:
    transform = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return transform[None, :, :] @ covariance_nev @ transform.T[None, :, :]


def _vectors_nev_to_xyz(vectors_nev: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors_nev, dtype=float)
    return vectors[:, [1, 0, 2]]


def _symmetrized_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    return 0.5 * (cov + np.swapaxes(cov, -1, -2))
