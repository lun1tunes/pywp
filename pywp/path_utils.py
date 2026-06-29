from __future__ import annotations

__all__ = ["normalize_user_path_text"]

_PATH_QUOTE_CHARS = "\"'"


def normalize_user_path_text(raw_path: object) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    while len(text) >= 2 and text[0] == text[-1] and text[0] in _PATH_QUOTE_CHARS:
        text = text[1:-1].strip()
    while text and text[0] in _PATH_QUOTE_CHARS:
        text = text[1:].lstrip()
    while text and text[-1] in _PATH_QUOTE_CHARS:
        text = text[:-1].rstrip()
    return text
