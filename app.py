import logging

# Suppress noisy Streamlit warnings BEFORE importing streamlit
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import streamlit as st

st.set_page_config(page_title="pywp", layout="wide")
st.title("Главная страница")
st.info("Выберите приложение в боковом меню.")
