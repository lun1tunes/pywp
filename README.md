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
# Быстрый прогон (по умолчанию исключает @slow)
pytest -q

# Явные профили:
python scripts/run_tests.py unit
python scripts/run_tests.py fast
python scripts/run_tests.py integration
python scripts/run_tests.py slow
python scripts/run_tests.py full
```

Маркерная сегментация:
- `integration`: e2e-потоки с реальным планировщиком/батч-расчетом.
- `slow`: самые дорогие по времени сценарии (dense/adaptive baseline и сложные TURN-кейсы).

## Notes

- Основные зависимости: `requirements.txt`
- Зависимости для разработки и тестов: `requirements-dev.txt`
- Если `python3 -m venv .venv` падает с ошибкой про `ensurepip`, установите пакет `python3-venv` для вашей версии Python.
