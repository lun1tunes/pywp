from __future__ import annotations

import app


def test_streamlit_entrypoint_exists() -> None:
    assert callable(app.run_app)
