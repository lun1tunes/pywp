from __future__ import annotations

from pywp.plotly_config import (
    DEFAULT_3D_CAMERA,
    default_plotly_chart_config,
    trajectory_plotly_chart_config,
)


def test_trajectory_plotly_chart_config_shows_modebar_for_camera_reset() -> None:
    assert trajectory_plotly_chart_config() == {
        "displayModeBar": True,
        "displaylogo": False,
    }


def test_default_plotly_chart_config_does_not_force_modebar() -> None:
    assert default_plotly_chart_config() == {
        "displaylogo": False,
    }


def test_default_3d_camera_matches_plotly_default_eye() -> None:
    assert DEFAULT_3D_CAMERA["eye"] == {"x": 1.25, "y": 1.25, "z": 1.25}
