from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pywp.eclipse_welltrack import decode_welltrack_bytes

__all__ = ["decode_welltrack_payload", "read_welltrack_file"]

MessageSink = Callable[[str], object]


def decode_welltrack_payload(
    raw_payload: bytes,
    *,
    source_label: str,
    info: MessageSink | None = None,
    warning: MessageSink | None = None,
) -> str:
    text, encoding = decode_welltrack_bytes(raw_payload)
    if encoding == "utf-8":
        return text
    if encoding.endswith("(replace)"):
        _notify(
            warning,
            f"{source_label}: не удалось надежно определить кодировку. "
            f"Текст декодирован как `{encoding}` с заменой поврежденных символов.",
        )
        return text
    _notify(
        info,
        f"{source_label}: текст декодирован как `{encoding}` (fallback, не UTF-8). "
        "Проверьте корректность имен и комментариев.",
    )
    return text


def read_welltrack_file(
    path_text: str,
    *,
    cwd: Path | None = None,
    info: MessageSink | None = None,
    warning: MessageSink | None = None,
    error: MessageSink | None = None,
) -> str:
    file_path_raw = str(path_text or "").strip()
    if not file_path_raw:
        _notify(warning, "Укажите путь к файлу WELLTRACK.")
        return ""

    file_path = Path(file_path_raw).expanduser()
    if not file_path.is_absolute():
        file_path = ((cwd or Path.cwd()) / file_path).resolve()

    if not file_path.exists():
        _notify(error, f"Файл не найден: {file_path}")
        return ""
    if not file_path.is_file():
        _notify(error, f"Путь не является файлом: {file_path}")
        return ""

    try:
        return decode_welltrack_payload(
            file_path.read_bytes(),
            source_label=f"Файл `{file_path}`",
            info=info,
            warning=warning,
        )
    except OSError as exc:
        _notify(error, f"Не удалось прочитать файл `{file_path}`: {exc}")
        return ""


def _notify(sink: MessageSink | None, message: str) -> None:
    if sink is not None:
        sink(str(message))
