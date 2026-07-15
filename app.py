from __future__ import annotations

import logging
from dataclasses import dataclass

# Suppress noisy Streamlit warnings BEFORE importing streamlit
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)


@dataclass(frozen=True)
class NavigationPageSpec:
    script_path: str
    title: str
    url_path: str
    default: bool = False
    visible_in_sidebar: bool = False


NAVIGATION_PAGE_SPECS: tuple[NavigationPageSpec, ...] = (
    NavigationPageSpec(
        script_path="pages/01_trajectory_constructor.py",
        title="КПТ",
        url_path="",
        default=True,
        visible_in_sidebar=True,
    ),
    NavigationPageSpec(
        script_path="pages/04_crs_calculator.py",
        title="Калькулятор СК",
        url_path="crs_calculator",
        visible_in_sidebar=True,
    ),
    NavigationPageSpec(
        script_path="pages/02_single_well.py",
        title="Single Well",
        url_path="single_well",
    ),
    NavigationPageSpec(
        script_path="pages/03_well_classification.py",
        title="Well Classification",
        url_path="well_classification",
    ),
)


def _build_pages() -> tuple[st.Page, ...]:
    return tuple(
        st.Page(
            spec.script_path,
            title=spec.title,
            url_path=spec.url_path,
            default=spec.default,
        )
        for spec in NAVIGATION_PAGE_SPECS
    )


def _render_sidebar_navigation(pages: tuple[st.Page, ...]) -> None:
    with st.sidebar:
        for spec, page in zip(NAVIGATION_PAGE_SPECS, pages, strict=True):
            if spec.visible_in_sidebar:
                st.page_link(page, label=spec.title, use_container_width=True)


def run_app() -> None:
    st.set_page_config(page_title="pywp", layout="wide")
    pages = _build_pages()
    page = st.navigation(list(pages), position="hidden")
    _render_sidebar_navigation(pages)
    page.run()


def _has_streamlit_context() -> bool:
    return get_script_run_ctx(suppress_warning=True) is not None


if _has_streamlit_context():
    run_app()
elif __name__ == "__main__":
    raise SystemExit("Запустите приложение командой `streamlit run app.py`.")
