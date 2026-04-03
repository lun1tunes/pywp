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
    (root / ".streamlit").mkdir(parents=True)
    (root / "pywp" / "three_viewer_assets" / "templates").mkdir(parents=True)

    (root / "main.py").write_text("print('root')\n", encoding="utf-8")
    (root / "pkg" / "mod.py").write_text("def f():\n    return 'ok'\n", encoding="utf-8")
    (root / "requirements.txt").write_text("requests==2.0\n", encoding="utf-8")
    (root / "requirements-dev.txt").write_text("pytest==8.0\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='pywp'\n", encoding="utf-8")
    (root / ".streamlit" / "config.toml").write_text("[theme]\n", encoding="utf-8")
    (root / "tools" / "viewer.json").write_text('{"ok": true}\n', encoding="utf-8")
    (root / "tools" / "sample.inc").write_text("WELLTRACK 'A'\n/\n", encoding="utf-8")
    (root / "tools" / "template.html").write_text("<html></html>\n", encoding="utf-8")
    (root / "tools" / "widget.js").write_text("console.log('ok')\n", encoding="utf-8")
    (root / "tools" / "theme.css").write_text("body{}\n", encoding="utf-8")
    (root / "tools" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (root / "pywp" / "three_viewer_assets" / "templates" / "viewer_template.html").write_text(
        "<div></div>\n",
        encoding="utf-8",
    )
    (root / ".venv" / "ignored" / "skip.py").write_text("print('skip')\n", encoding="utf-8")
    (root / "tools" / "binary.bin").write_bytes(b"\x00\x01")
    (root / "all.txt").write_text("archive\n", encoding="utf-8")

    files = module.collect_files(root)
    relative = [path.relative_to(root).as_posix() for path in files]

    assert relative == [
        ".streamlit/config.toml",
        "main.py",
        "pkg/mod.py",
        "pyproject.toml",
        "pywp/three_viewer_assets/templates/viewer_template.html",
        "requirements-dev.txt",
        "requirements.txt",
        "tools/notes.md",
        "tools/sample.inc",
        "tools/template.html",
        "tools/theme.css",
        "tools/viewer.json",
        "tools/widget.js",
    ]


def test_pack_and_unpack_restore_contents(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    root = tmp_path / "src"
    out = tmp_path / "out"
    archive = tmp_path / "archive.txt"
    (root / "pkg").mkdir(parents=True)
    (root / ".streamlit").mkdir(parents=True)
    (root / "pywp" / "three_viewer_assets" / "vendor").mkdir(parents=True)

    original_files = {
        "main.py": "print('root')\n",
        "pkg/mod.py": "def f():\n    return 'ok'",
        "requirements.txt": "requests==2.0\n",
        ".streamlit/config.toml": "[theme]\nbase='light'\n",
        "pywp/three_viewer_assets/vendor/OrbitControls.js": "window.OrbitControls = {};\n",
        "viewer.json": '{"version": 1}\n',
        "sample.inc": "WELLTRACK 'A'\n/\n",
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


def test_collect_files_excludes_custom_archive_inside_root(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    root = tmp_path / "src"
    root.mkdir(parents=True)
    archive_path = root / "snapshot.txt"

    (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
    archive_path.write_text("old archive\n", encoding="utf-8")

    files = module.collect_files(root, archive_path=archive_path)
    relative = [path.relative_to(root).as_posix() for path in files]

    assert relative == ["app.py"]
