from __future__ import annotations


DEFAULT_3D_CAMERA: dict[str, dict[str, float]] = {
    "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
}


def trajectory_plotly_chart_config() -> dict[str, object]:
    # Plotly 3D relies on the reset-camera modebar controls; Streamlit does not
    # expose a reliable double-click reset flow for 3D scenes.
    return {
        "displayModeBar": True,
        "displaylogo": False,
    }


def default_plotly_chart_config() -> dict[str, object]:
    return {
        "displaylogo": False,
    }
