from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_project_pack_module():
    spec = importlib.util.spec_from_file_location(
        "project_pack_test_module",
        "scripts/project_pack.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_files_skips_excluded_and_poetry(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    root = tmp_path / "src"
    (root / "pkg").mkdir(parents=True)
    (root / ".venv" / "ignored").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)

    (root / "main.py").write_text("print('root')\n", encoding="utf-8")
    (root / "pkg" / "mod.py").write_text("def f():\n    return 'ok'\n", encoding="utf-8")
    (root / "requirements.txt").write_text("requests==2.0\n", encoding="utf-8")
    (root / ".venv" / "ignored" / "skip.py").write_text("print('skip')\n", encoding="utf-8")
    (root / "tools" / "poetry_helper.py").write_text("print('poetry')\n", encoding="utf-8")

    files = module.collect_files(root)
    relative = [path.relative_to(root).as_posix() for path in files]

    assert relative == ["main.py", "pkg/mod.py", "requirements.txt"]


def test_pack_and_unpack_restore_contents(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    root = tmp_path / "src"
    out = tmp_path / "out"
    archive = tmp_path / "archive.txt"
    (root / "pkg").mkdir(parents=True)

    original_files = {
        "main.py": "print('root')\n",
        "pkg/mod.py": "def f():\n    return 'ok'",
        "requirements.txt": "requests==2.0\n",
    }
    for relative_path, content in original_files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    module.pack(root, archive)
    module.unpack(out, archive)

    restored_files = sorted(
        path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()
    )
    assert restored_files == sorted(original_files)
    for relative_path, expected_content in original_files.items():
        restored_path = out / relative_path
        assert restored_path.read_text(encoding="utf-8") == expected_content
