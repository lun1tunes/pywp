# pywp

Планировщик траектории скважины (S, t1, t3) на Python с визуализацией в Streamlit/Plotly.

## Setup (venv + requirements)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Run

```bash
streamlit run app.py
```

## Tests

```bash
pytest -q
```

## Notes

- Основные зависимости: `requirements.txt`
- Зависимости для разработки и тестов: `requirements-dev.txt`
- Если `python3 -m venv .venv` падает с ошибкой про `ensurepip`, установите пакет `python3-venv` для вашей версии Python.
