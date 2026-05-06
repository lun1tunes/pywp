import logging

# Suppress noisy Streamlit warnings BEFORE importing streamlit
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)


def run_app() -> None:
    st.set_page_config(page_title="pywp", layout="wide")
    st.title("Главная страница")
    st.info("Выберите приложение в боковом меню.")


def _has_streamlit_context() -> bool:
    return get_script_run_ctx(suppress_warning=True) is not None


if _has_streamlit_context():
    run_app()
elif __name__ == "__main__":
    raise SystemExit("Запустите приложение командой `streamlit run app.py`.")
