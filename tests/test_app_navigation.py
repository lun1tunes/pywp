from __future__ import annotations

import importlib

from streamlit.testing.v1 import AppTest

app = importlib.import_module("app")


def _sidebar_page_link_labels(at: AppTest) -> list[str]:
    return [
        str(getattr(node, "label", "")).strip()
        for node in at.sidebar
        if type(node).__name__ == "UnknownElement"
        and str(getattr(node, "label", "")).strip()
    ]


def test_build_pages_sets_expected_titles_and_paths() -> None:
    assert app.NAVIGATION_PAGE_SPECS == (
        app.NavigationPageSpec(
            script_path="pages/01_trajectory_constructor.py",
            title="КПТ",
            url_path="",
            default=True,
            visible_in_sidebar=True,
        ),
        app.NavigationPageSpec(
            script_path="pages/04_crs_calculator.py",
            title="Калькулятор CRS",
            url_path="crs_calculator",
            visible_in_sidebar=True,
        ),
        app.NavigationPageSpec(
            script_path="pages/02_single_well.py",
            title="Single Well",
            url_path="single_well",
        ),
        app.NavigationPageSpec(
            script_path="pages/03_well_classification.py",
            title="Well Classification",
            url_path="well_classification",
        ),
    )


def test_app_sidebar_shows_only_kpt_and_crs_links() -> None:
    at = AppTest.from_file("app.py")
    at.run(timeout=120)

    assert not at.exception
    assert _sidebar_page_link_labels(at) == ["КПТ", "Калькулятор CRS"]


def test_hidden_pages_remain_switchable_from_app_router() -> None:
    single_well = AppTest.from_file("app.py")
    single_well.switch_page("pages/02_single_well.py")
    single_well.run(timeout=120)

    well_classification = AppTest.from_file("app.py")
    well_classification.switch_page("pages/03_well_classification.py")
    well_classification.run(timeout=120)

    assert not single_well.exception
    assert not well_classification.exception
    assert _sidebar_page_link_labels(single_well) == ["КПТ", "Калькулятор CRS"]
    assert _sidebar_page_link_labels(well_classification) == [
        "КПТ",
        "Калькулятор CRS",
    ]
