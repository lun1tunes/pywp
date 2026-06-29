from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from pywp.eclipse_welltrack import decode_welltrack_bytes
from pywp.path_utils import normalize_user_path_text

__all__ = [
    "decode_welltrack_payload",
    "read_welltrack_file",
    "read_welltrack_sources",
]

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
    file_path_raw = normalize_user_path_text(path_text)
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


def read_welltrack_sources(
    path_texts: Iterable[str],
    *,
    cwd: Path | None = None,
    info: MessageSink | None = None,
    warning: MessageSink | None = None,
    error: MessageSink | None = None,
) -> str:
    resolved_files = _resolve_welltrack_source_files(
        path_texts,
        cwd=cwd,
        warning=warning,
        error=error,
    )
    if not resolved_files:
        return ""
    texts: list[str] = []
    for file_path in resolved_files:
        try:
            text = decode_welltrack_payload(
                file_path.read_bytes(),
                source_label=f"Файл `{file_path}`",
                info=info,
                warning=warning,
            )
        except OSError as exc:
            _notify(error, f"Не удалось прочитать файл `{file_path}`: {exc}")
            continue
        if text.strip():
            texts.append(text.strip())
    return "\n\n".join(texts)


def _resolve_welltrack_source_files(
    path_texts: Iterable[str],
    *,
    cwd: Path | None = None,
    warning: MessageSink | None = None,
    error: MessageSink | None = None,
) -> list[Path]:
    resolved_files: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in path_texts:
        path_text = normalize_user_path_text(raw_path)
        if not path_text:
            continue
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = ((cwd or Path.cwd()) / path).resolve()
        if not path.exists():
            _notify(error, f"Путь не найден: {path}")
            continue
        if path.is_dir():
            try:
                inc_files = sorted(
                    (
                        child.resolve()
                        for child in path.iterdir()
                        if child.is_file() and child.suffix.casefold() == ".inc"
                    ),
                    key=lambda item: item.name.casefold(),
                )
            except OSError as exc:
                _notify(error, f"Не удалось прочитать папку `{path}`: {exc}")
                continue
            if not inc_files:
                _notify(
                    warning,
                    f"В папке `{path}` не найдено WELLTRACK файлов с расширением `.INC`.",
                )
                continue
            for inc_file in inc_files:
                if inc_file in seen_paths:
                    continue
                seen_paths.add(inc_file)
                resolved_files.append(inc_file)
            continue
        if not path.is_file():
            _notify(error, f"Путь не является файлом или папкой: {path}")
            continue
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        resolved_files.append(resolved_path)
    if not resolved_files:
        _notify(warning, "Укажите хотя бы один путь к WELLTRACK файлу или папке.")
    return resolved_files


def _notify(sink: MessageSink | None, message: str) -> None:
    if sink is not None:
        sink(str(message))
