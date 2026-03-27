from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import streamlit.components.v1 as components

_ASSETS_DIR = Path(__file__).resolve().parent / "three_viewer_assets"
_VENDOR_DIR = _ASSETS_DIR / "vendor"
_TEMPLATE_PATH = _ASSETS_DIR / "templates" / "viewer_template.html"


@lru_cache(maxsize=1)
def _three_library_text() -> str:
    return (_VENDOR_DIR / "three.min.js").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _trackball_controls_text() -> str:
    return (_VENDOR_DIR / "TrackballControls.js").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _viewer_template_text() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _viewer_template_with_libraries() -> str:
    return (
        _viewer_template_text()
        .replace("__THREE_LIBRARY__", _three_library_text())
        .replace("__TRACKBALL_CONTROLS__", _trackball_controls_text())
    )


def render_local_three_scene(
    payload: Mapping[str, object],
    *,
    height: int,
) -> None:
    payload_json = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = _viewer_template_with_libraries().replace("__SCENE_PAYLOAD__", payload_json)
    components.html(
        html,
        height=int(height),
        scrolling=False,
    )
