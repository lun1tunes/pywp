"""Shared production solver matrix, including deterministic repeated solves."""

import pytest

from scripts.check_solver_baseline import collect_baseline


@pytest.mark.integration
def test_solver_reference_matrix_preserves_engineering_invariants():
    # collect_baseline verifies strict MD, endpoint lateral/vertical tolerances,
    # INC/DLS/MD limits, exact repeatability, and unchanged input/configuration.
    results = collect_baseline()
    assert {name for name in results if name.endswith(":stations")} == {
        "ordinary:stations",
        "projected_crs:stations",
        "classic_j:stations",
        "multi_horizontal:stations",
    }
