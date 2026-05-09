from __future__ import annotations

import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from pywp.ptc_page import run_page


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(
            "Запустите приложение командой `streamlit run pages/01_trajectory_constructor.py`."
        )
    run_page()
