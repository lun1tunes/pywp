from __future__ import annotations

from pywp.three_config import DEFAULT_THREE_CAMERA


def test_default_three_camera_uses_z_up_eye() -> None:
    assert DEFAULT_THREE_CAMERA["up"] == {"x": 0.0, "y": 0.0, "z": 1.0}
    assert DEFAULT_THREE_CAMERA["eye"] == {"x": 1.25, "y": 1.25, "z": 1.25}
