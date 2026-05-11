from __future__ import annotations

import numpy as np

from pywp.iscwsa_mwd import (
    DEFAULT_ISCWSA_MWD_ENVIRONMENT,
    ISCWSA_MWD_POOR_MAGNETIC,
    ISCWSA_MWD_UNKNOWN_MAGNETIC,
    IscwsaMwdEnvironment,
    MWD_POOR_MAGNETIC_TOOL_CODE,
    MWD_UNKNOWN_MAGNETIC_TOOL_CODE,
    _formula_weight,
    iscwsa_mwd_covariance_xyz,
)


def _component_signature(tool_code):
    return tuple(
        (
            component.name,
            component.vector,
            component.propagation,
            component.value_1sigma,
            component.unit,
            component.formula,
        )
        for component in tool_code.components
    )


def test_mwd_poor_tool_code_is_digitized_from_reference_table() -> None:
    expected = (
        ("drfr_e", "e", "random", 0.35, "m", "one"),
        ("drfr_s", "s", "random", 2.2, "m", "one"),
        ("drfs", "s", "systematic", 1.0, "m", "one"),
        ("dsfs", "e", "systematic", 0.00056, "-", "tmd"),
        ("dstg", "e", "global", 0.00000025, "1/m", "tmd_tvd"),
        ("xym1", "i", "systematic", 0.1, "deg", "sin_inc"),
        ("xym2", "l", "systematic", 0.1, "deg", "sin_inc"),
        ("xym3", "i", "systematic", 0.1, "deg", "cos_azi_abs_cos_inc"),
        ("xym3", "l", "systematic", 0.1, "deg", "neg_sin_azi_abs_cos_inc"),
        ("xym4", "i", "systematic", 0.1, "deg", "sin_azi_abs_cos_inc"),
        ("xym4", "l", "systematic", 0.1, "deg", "cos_azi_abs_cos_inc"),
        ("sag", "i", "systematic", 0.2, "deg", "sin_inc"),
        ("decg", "a", "global", 0.36, "deg", "one"),
        ("dbhg", "a", "global", 5000.0, "deg_nt", "inv_mtot_cos_dip"),
        ("amil", "a", "systematic", 300.0, "nt", "amil"),
        ("abxy_ti1", "i", "systematic", 0.008, "m/s2", "abxy_ti1_i"),
        ("abxy_ti1", "a", "systematic", 0.008, "m/s2", "abxy_ti1_a"),
        ("abxy_ti2", "l", "systematic", 0.008, "m/s2", "abxy_ti2_l"),
        ("abz", "i", "systematic", 0.008, "m/s2", "abz_i"),
        ("abz", "a", "systematic", 0.008, "m/s2", "abz_a"),
        ("asxy_ti1", "i", "systematic", 0.0005, "-", "asxy_ti1_i"),
        ("asxy_ti1", "a", "systematic", 0.0005, "-", "asxy_ti1_a"),
        ("asxy_ti2", "i", "systematic", 0.0005, "-", "asxy_ti2_i"),
        ("asxy_ti2", "a", "systematic", 0.0005, "-", "asxy_ti2_a"),
        ("asxy_ti3", "a", "systematic", 0.0005, "-", "asxy_ti3_a"),
        ("asz", "i", "systematic", 0.0005, "-", "asz_i"),
        ("asz", "a", "systematic", 0.0005, "-", "asz_a"),
        ("mbxy_ti1", "a", "systematic", 70.0, "nt", "mbxy_ti1_a"),
        ("mbxy_ti2", "a", "systematic", 70.0, "nt", "mbxy_ti2_a"),
        ("mbz", "a", "systematic", 70.0, "nt", "mbz_a"),
        ("msxy_ti1", "a", "systematic", 0.0016, "-", "msxy_ti1_a"),
        ("msxy_ti2", "a", "systematic", 0.0016, "-", "msxy_ti2_a"),
        ("msxy_ti3", "a", "systematic", 0.0016, "-", "msxy_ti3_a"),
        ("msz", "a", "systematic", 0.0016, "-", "msz_a"),
    )

    assert MWD_POOR_MAGNETIC_TOOL_CODE.name == ISCWSA_MWD_POOR_MAGNETIC
    assert _component_signature(MWD_POOR_MAGNETIC_TOOL_CODE) == expected


def test_mwd_unknown_tool_code_is_conservative_superset_of_poor_table() -> None:
    expected_poor_stack = tuple(
        (
            name,
            vector,
            propagation,
            0.32 if name in {"xym1", "xym2", "xym3", "xym4"} else value,
            unit,
            formula,
        )
        for name, vector, propagation, value, unit, formula in _component_signature(
            MWD_POOR_MAGNETIC_TOOL_CODE
        )
    )
    expected_unknown_terms = (
        ("amic", "a", "systematic", 0.5, "deg", "one"),
        ("amid", "a", "systematic", 1.5, "deg", "sin_inc_sin_azm"),
        ("azmr", "a", "random", 0.5, "deg", "one"),
        ("azms", "a", "systematic", 1.1, "deg", "one"),
    )

    assert MWD_UNKNOWN_MAGNETIC_TOOL_CODE.name == ISCWSA_MWD_UNKNOWN_MAGNETIC
    assert _component_signature(MWD_UNKNOWN_MAGNETIC_TOOL_CODE) == (
        expected_poor_stack + expected_unknown_terms
    )


def test_mwd_default_environment_constants_are_explicit() -> None:
    assert DEFAULT_ISCWSA_MWD_ENVIRONMENT == IscwsaMwdEnvironment(
        gtot_mps2=9.80665,
        mtot_nt=50_000.0,
        dip_deg=82.35,
        declination_deg=0.0,
        lateral_singularity_inc_deg=0.01,
    )


def test_iscwsa_mwd_poor_covariance_is_psd_and_grows_with_md() -> None:
    result = iscwsa_mwd_covariance_xyz(
        md_m=np.asarray([0.0, 500.0, 1500.0, 3000.0], dtype=float),
        inc_deg=np.asarray([0.0, 12.0, 45.0, 86.0], dtype=float),
        azi_deg=np.asarray([0.0, 25.0, 65.0, 90.0], dtype=float),
        tvd_m=np.asarray([0.0, 490.0, 1320.0, 1850.0], dtype=float),
    )

    covariance = result.covariance_xyz
    eigenvalues = np.linalg.eigvalsh(covariance)

    assert covariance.shape == (4, 3, 3)
    assert np.allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12)
    assert float(np.min(eigenvalues)) >= -1e-8
    assert float(np.trace(covariance[-1])) > float(np.trace(covariance[1]))


def test_iscwsa_mwd_poor_separates_random_systematic_and_global_buckets() -> None:
    result = iscwsa_mwd_covariance_xyz(
        md_m=np.asarray([0.0, 800.0, 1600.0], dtype=float),
        inc_deg=np.asarray([0.0, 30.0, 75.0], dtype=float),
        azi_deg=np.asarray([0.0, 40.0, 80.0], dtype=float),
    )

    total = (
        result.covariance_xyz_random
        + result.covariance_xyz_systematic
        + result.covariance_xyz_global
    )

    assert np.allclose(result.covariance_xyz, total, atol=1e-12)
    assert float(np.trace(result.covariance_xyz_random[-1])) > 0.0
    assert float(np.trace(result.covariance_xyz_systematic[-1])) > 0.0
    assert float(np.trace(result.covariance_xyz_global[-1])) > 0.0


def test_mwd_unknown_covariance_is_component_level_conservative_over_poor() -> None:
    md_values = np.asarray([0.0, 2000.0, 5000.0, 7000.0], dtype=float)
    inc_values = np.asarray([0.0, 70.0, 90.0, 90.0], dtype=float)
    azi_values = np.asarray([0.0, 90.0, 90.0, 90.0], dtype=float)

    poor = iscwsa_mwd_covariance_xyz(
        md_m=md_values,
        inc_deg=inc_values,
        azi_deg=azi_values,
        tool_code=ISCWSA_MWD_POOR_MAGNETIC,
    ).covariance_xyz
    unknown = iscwsa_mwd_covariance_xyz(
        md_m=md_values,
        inc_deg=inc_values,
        azi_deg=azi_values,
        tool_code=ISCWSA_MWD_UNKNOWN_MAGNETIC,
    ).covariance_xyz
    covariance_delta = unknown - poor
    covariance_delta = 0.5 * (covariance_delta + np.swapaxes(covariance_delta, -1, -2))

    assert float(np.trace(unknown[-1])) > float(np.trace(poor[-1]))
    assert float(np.min(np.linalg.eigvalsh(covariance_delta))) >= -1e-8


def test_formula_weights_match_digitized_csv_expressions() -> None:
    environment = DEFAULT_ISCWSA_MWD_ENVIRONMENT
    md_values = np.asarray([0.0, 1234.5, 4321.0], dtype=float)
    tvd_values = np.asarray([0.0, 1000.0, 2100.0], dtype=float)
    inc_values_deg = np.asarray([0.0, 37.0, 86.0], dtype=float)
    azi_values_deg = np.asarray([15.0, 123.0, 271.0], dtype=float)
    inc = np.deg2rad(inc_values_deg)
    azi = np.deg2rad(azi_values_deg)
    azm = np.deg2rad(azi_values_deg - environment.declination_deg)
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)
    sin_azi = np.sin(azi)
    cos_azi = np.cos(azi)
    sin_azm = np.sin(azm)
    cos_azm = np.cos(azm)
    w34 = np.sqrt(np.clip(1.0 - sin_inc * sin_inc, 0.0, None))
    tan_dip = np.tan(np.deg2rad(environment.dip_deg))
    mtot_cos_dip = environment.mtot_nt * np.cos(np.deg2rad(environment.dip_deg))
    gtot = environment.gtot_mps2
    expected_by_formula = {
        "one": np.ones_like(md_values),
        "tmd": md_values,
        "tmd_tvd": md_values * tvd_values,
        "sin_inc": sin_inc,
        "cos_azi_abs_cos_inc": cos_azi * w34,
        "neg_sin_azi_abs_cos_inc": -sin_azi * w34,
        "sin_azi_abs_cos_inc": sin_azi * w34,
        "inv_mtot_cos_dip": np.full_like(md_values, 1.0 / mtot_cos_dip),
        "amil": sin_inc * sin_azm / mtot_cos_dip,
        "sin_inc_sin_azm": sin_inc * sin_azm,
        "abxy_ti1_i": -cos_inc / gtot,
        "abxy_ti1_a": tan_dip * cos_inc * sin_azm / gtot,
        "abxy_ti2_l": (cos_inc - tan_dip * cos_azm * sin_inc) / gtot,
        "abz_i": -sin_inc / gtot,
        "abz_a": tan_dip * sin_inc * sin_azm / gtot,
        "asxy_ti1_i": sin_inc * cos_inc / np.sqrt(2.0),
        "asxy_ti1_a": -(tan_dip * sin_inc * cos_inc * sin_azm) / np.sqrt(2.0),
        "asxy_ti2_i": sin_inc * cos_inc / 2.0,
        "asxy_ti2_a": -tan_dip * sin_inc * cos_inc * sin_azm / 2.0,
        "asxy_ti3_a": (tan_dip * sin_inc * cos_azm - cos_inc) / 2.0,
        "asz_i": -sin_inc * cos_inc,
        "asz_a": tan_dip * sin_inc * cos_inc * sin_azm,
        "mbxy_ti1_a": -(cos_inc * sin_azm) / mtot_cos_dip,
        "mbxy_ti2_a": cos_azm / mtot_cos_dip,
        "mbz_a": -sin_inc * sin_azm / mtot_cos_dip,
        "msxy_ti1_a": (
            sin_inc * sin_azm * tan_dip * cos_inc + sin_inc * cos_azm
        )
        / np.sqrt(2.0),
        "msxy_ti2_a": sin_azm
        * (
            tan_dip * sin_inc * cos_inc
            - cos_inc * cos_inc * cos_azm
            - cos_azm
        )
        / 2.0,
        "msxy_ti3_a": (
            (cos_inc * cos_azm) ** 2
            - cos_inc * sin_azm * sin_azm
            - tan_dip * sin_inc * cos_azm
        )
        / 2.0,
        "msz_a": -(sin_inc * cos_azm + tan_dip * cos_inc) * sin_inc * sin_azm,
    }

    for formula, expected in expected_by_formula.items():
        actual = _formula_weight(
            formula=formula,
            md_values=md_values,
            inc_values_deg=inc_values_deg,
            azi_values_deg=azi_values_deg,
            tvd_values=tvd_values,
            environment=environment,
        )
        assert np.allclose(actual, expected, atol=1e-14, rtol=1e-14), formula
