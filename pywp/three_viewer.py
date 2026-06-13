from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import streamlit.components.v1 as components

_ASSETS_DIR = Path(__file__).resolve().parent / "three_viewer_assets"
_VENDOR_DIR = _ASSETS_DIR / "vendor"
_TEMPLATE_PATH = _ASSETS_DIR / "templates" / "viewer_template.html"
_viewer_component = components.declare_component(
    "three_viewer_runtime",
    path=str(_ASSETS_DIR),
)
_SERIALIZED_PAYLOAD_CACHE: list[dict[str, object]] = []


@lru_cache(maxsize=1)
def _read_text_cached(path_str: str, mtime_ns: int) -> str:
    return Path(path_str).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _read_text_with_mtime(path: Path) -> str:
    stat = path.stat()
    return _read_text_cached(str(path), int(getattr(stat, "st_mtime_ns", 0)))


def _three_library_text() -> str:
    return _read_text_with_mtime(_VENDOR_DIR / "three.min.js")


def _viewer_template_text() -> str:
    return _read_text_with_mtime(_TEMPLATE_PATH)


def _orbit_controls_text() -> str:
    return _read_text_with_mtime(_VENDOR_DIR / "OrbitControls.js")


def _fast_replan_text() -> str:
    return _read_text_with_mtime(_ASSETS_DIR / "fast_replan.js")


def _viewer_template_with_libraries() -> str:
    return (
        _viewer_template_text()
        .replace("__THREE_LIBRARY__", _three_library_text())
        .replace("__ORBIT_CONTROLS__", _orbit_controls_text())
        .replace("__FAST_REPLAN__", _fast_replan_text())
    )


def _payload_digest(payload_json: str) -> str:
    return hashlib.blake2b(
        str(payload_json).encode("utf-8"),
        digest_size=20,
    ).hexdigest()


def _viewer_runtime_digest() -> str:
    digest = hashlib.blake2b(digest_size=20)
    for path in (
        _TEMPLATE_PATH,
        _VENDOR_DIR / "three.min.js",
        _VENDOR_DIR / "OrbitControls.js",
        _ASSETS_DIR / "fast_replan.js",
    ):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(int(getattr(stat, "st_mtime_ns", 0))).encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("utf-8"))
    return digest.hexdigest()


def _serialized_payload(
    payload: Mapping[str, object],
    *,
    stable_key: str,
    instance_token: int,
) -> tuple[str, str, str]:
    resolved_instance_token = int(instance_token)
    for entry in reversed(_SERIALIZED_PAYLOAD_CACHE):
        if (
            entry.get("payload") is payload
            and entry.get("stable_key") == stable_key
            and int(entry.get("instance_token", 0)) == resolved_instance_token
        ):
            return (
                str(entry.get("payload_json", "")),
                str(entry.get("payload_digest", "")),
                str(entry.get("edit_channel", "")),
            )
    channel_digest = hashlib.blake2b(
        f"{stable_key}:{resolved_instance_token}".encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    edit_channel = f"pywp_three_edit_{channel_digest}"
    payload_dict = dict(payload)
    payload_dict["edit_channel"] = edit_channel
    payload_json = json.dumps(
        payload_dict,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    payload_digest = _payload_digest(payload_json)
    _SERIALIZED_PAYLOAD_CACHE.append(
        {
            "payload": payload,
            "stable_key": stable_key,
            "instance_token": resolved_instance_token,
            "edit_channel": edit_channel,
            "payload_json": payload_json,
            "payload_digest": payload_digest,
        }
    )
    if len(_SERIALIZED_PAYLOAD_CACHE) > 4:
        del _SERIALIZED_PAYLOAD_CACHE[:-4]
    return payload_json, payload_digest, edit_channel


def render_local_three_scene(
    payload: Mapping[str, object],
    *,
    height: int,
    instance_token: int = 0,
    key: str | None = None,
) -> object:
    stable_key = str(key or payload.get("title") or "scene")
    payload_json, payload_digest, edit_channel = _serialized_payload(
        payload,
        stable_key=stable_key,
        instance_token=int(instance_token),
    )
    return _viewer_component(
        payload_json=payload_json,
        payload_digest=payload_digest,
        runtime_digest=_viewer_runtime_digest(),
        has_anticollision_payload=bool(
            payload.get("anti_collision_layer_state")
            or payload.get("collisions")
        ),
        channel=edit_channel,
        height=int(height),
        instance_token=int(instance_token),
        default=None,
        key=f"three-viewer-runtime-{stable_key}",
    )
