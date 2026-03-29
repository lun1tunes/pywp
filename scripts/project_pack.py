#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ARCHIVE_FILE = "all.txt"
BEGIN = "===BEGIN_FILE==="
END = "===END_FILE==="

# Static assets for the Three.js viewer (HTML/JS); packed alongside Python sources.
THREE_VIEWER_ASSETS_REL = Path("pywp/three_viewer_assets")

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


def should_skip_path(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    if "poetry" in path.name.lower():
        return True
    return False


def collect_three_viewer_assets(root: Path) -> list[Path]:
    assets_dir = root / THREE_VIEWER_ASSETS_REL
    if not assets_dir.is_dir():
        return []
    out: list[Path] = []
    for path in assets_dir.rglob("*"):
        if path.is_file() and not should_skip_path(path, root):
            out.append(path)
    return out


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if path.is_file() and not should_skip_path(path, root):
            files.append(path)

    req = root / "requirements.txt"
    if req.exists() and req.is_file() and not should_skip_path(req, root):
        files.append(req)

    files.extend(collect_three_viewer_assets(root))

    return sorted(set(files), key=lambda p: str(p.relative_to(root)))


def pack(root: Path, output_file: Path) -> None:
    files = collect_files(root)
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
            "Pack project .py files, requirements.txt, and pywp/three_viewer_assets "
            "to all.txt and unpack back."
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
