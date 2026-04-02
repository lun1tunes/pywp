#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ARCHIVE_FILE = "all.txt"
BEGIN = "===BEGIN_FILE==="
END = "===END_FILE==="

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
}

INCLUDED_FILE_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".ini",
    ".cfg",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".html",
    ".js",
    ".css",
}

INCLUDED_FILE_NAMES = {}


def should_skip_path(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    return False


def should_include_path(path: Path) -> bool:
    if path.name in INCLUDED_FILE_NAMES:
        return True
    return path.suffix.lower() in INCLUDED_FILE_EXTENSIONS


def collect_files(root: Path, *, archive_path: Path | None = None) -> list[Path]:
    files: list[Path] = []
    archive_resolved = archive_path.resolve() if archive_path is not None else None
    default_archive_resolved = (root / ARCHIVE_FILE).resolve()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip_path(path, root):
            continue
        if not should_include_path(path):
            continue
        resolved_path = path.resolve()
        if resolved_path == default_archive_resolved:
            continue
        if archive_resolved is not None and resolved_path == archive_resolved:
            continue
        files.append(path)

    return sorted(set(files), key=lambda p: str(p.relative_to(root)))


def pack(root: Path, output_file: Path) -> None:
    files = collect_files(root, archive_path=output_file)
    with output_file.open("w", encoding="utf-8", newline="") as out:
        for path in files:
            rel = path.relative_to(root).as_posix()
            content = path.read_text(encoding="utf-8")
            out.write(f"{BEGIN}\t{rel}\t{len(content)}\n")
            out.write(content)
            out.write("\n")
            out.write(f"{END}\n")

    print(f"Packed {len(files)} files into {output_file}")


def unpack(root: Path, input_file: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Archive file not found: {input_file}")

    restored = 0

    with input_file.open("r", encoding="utf-8") as src:
        while True:
            header = src.readline()
            if not header:
                break
            if not header.startswith(f"{BEGIN}\t"):
                raise ValueError("Invalid archive format: malformed BEGIN header")

            payload = header.rstrip("\n").split("\t", maxsplit=2)
            if len(payload) != 3:
                raise ValueError("Invalid archive format: malformed BEGIN payload")
            _, rel_path, raw_len = payload
            content_len = int(raw_len)

            content = src.read(content_len)
            if len(content) != content_len:
                raise ValueError("Invalid archive format: truncated file content")

            separator = src.read(1)
            if separator != "\n":
                raise ValueError("Invalid archive format: missing content separator")

            end_line = src.readline().rstrip("\n")
            if end_line != END:
                raise ValueError("Invalid archive format: missing END marker")

            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            restored += 1

    print(f"Restored {restored} files from {input_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack project text sources/config/assets (Python, HTML/JS/CSS, JSON/TOML,"
            " Markdown, WELLTRACK .inc, etc.) to all.txt and unpack back."
        )
    )
    parser.add_argument(
        "mode",
        choices=("pack", "unpack"),
        help="pack: create all.txt, unpack: restore files from all.txt",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help=f"Archive file path (default: <root>/{ARCHIVE_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    archive = (root / ARCHIVE_FILE) if args.archive is None else Path(args.archive).resolve()

    if args.mode == "pack":
        pack(root, archive)
    else:
        unpack(root, archive)


if __name__ == "__main__":
    main()
