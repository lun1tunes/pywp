from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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
    (root / "tests").mkdir(parents=True)
    (root / ".streamlit").mkdir(parents=True)
    (root / "pywp" / "three_viewer_assets" / "templates").mkdir(parents=True)

    (root / "main.py").write_text("print('root')\n", encoding="utf-8")
    (root / "pkg" / "mod.py").write_text("def f():\n    return 'ok'\n", encoding="utf-8")
    (root / "conftest.py").write_text("pytest_plugins = []\n", encoding="utf-8")
    (root / "test_root.py").write_text("def test_root():\n    assert True\n", encoding="utf-8")
    (root / "requirements.txt").write_text("requests==2.0\n", encoding="utf-8")
    (root / "requirements-dev.txt").write_text("pytest==8.0\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='pywp'\n", encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (root / ".streamlit" / "config.toml").write_text("[theme]\n", encoding="utf-8")
    (root / "tools" / "viewer.json").write_text('{"ok": true}\n', encoding="utf-8")
    (root / "tools" / "sample.inc").write_text("WELLTRACK 'A'\n/\n", encoding="utf-8")
    (root / "tools" / "template.html").write_text("<html></html>\n", encoding="utf-8")
    (root / "tools" / "widget.js").write_text("console.log('ok')\n", encoding="utf-8")
    (root / "tools" / "theme.css").write_text("body{}\n", encoding="utf-8")
    (root / "tools" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (root / "tools" / "api_test.py").write_text("def test_api():\n    assert True\n", encoding="utf-8")
    (root / "pywp" / "three_viewer_assets" / "templates" / "viewer_template.html").write_text(
        "<div></div>\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_pack.py").write_text("def test_pack():\n    assert True\n", encoding="utf-8")
    (root / ".venv" / "ignored" / "skip.py").write_text("print('skip')\n", encoding="utf-8")
    (root / "tools" / "binary.bin").write_bytes(b"\x00\x01")
    (root / "all.txt").write_text("archive\n", encoding="utf-8")

    files = module.collect_files(root)
    relative = [path.relative_to(root).as_posix() for path in files]

    assert relative == [
        "main.py",
        "pkg/mod.py",
        "pywp/three_viewer_assets/templates/viewer_template.html",
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
        "pywp/three_viewer_assets/vendor/OrbitControls.js": "window.OrbitControls = {};\n",
        "viewer.json": '{"version": 1}\n',
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


def test_pack_writes_relative_paths_and_unpack_restores_structure(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    root = tmp_path / "src"
    out = tmp_path / "out"
    archive = tmp_path / "archive.txt"

    original_files = {
        "app.py": "print('root')\n",
        "pkg/nested/mod.py": "VALUE = 1\n",
        "pywp/viewer/assets/widget.js": "console.log('ok')\n",
    }
    for relative_path, content in original_files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    module.pack(root, archive)
    archive_text = archive.read_text(encoding="utf-8")

    assert "===BEGIN_FILE===\tapp.py\t" in archive_text
    assert "===BEGIN_FILE===\tpkg/nested/mod.py\t" in archive_text
    assert "===BEGIN_FILE===\tpywp/viewer/assets/widget.js\t" in archive_text

    module.unpack(out, archive)

    restored_files = sorted(
        path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()
    )
    assert restored_files == sorted(original_files)


def test_split_and_join_restore_archive_and_unpack(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    root = tmp_path / "src"
    out = tmp_path / "out"
    archive = tmp_path / "archive.txt"

    original_files = {
        "app.py": "print('root')\n",
        "pkg/mod.py": "VALUE = 1\n",
        "assets/view.js": "console.log('ok')\n",
    }
    for relative_path, content in original_files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    module.pack(root, archive)
    original_archive = archive.read_bytes()
    chunk_paths = module.split_archive(archive, chunk_size=40)

    assert [path.name for path in chunk_paths] == [
        f"archive{index}" for index in range(1, len(chunk_paths) + 1)
    ]
    assert len(chunk_paths) > 1

    archive.unlink()
    module.join_archive(archive)

    assert archive.read_bytes() == original_archive

    module.unpack(out, archive)
    restored_files = sorted(
        path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()
    )
    assert restored_files == sorted(original_files)


def test_join_detects_missing_chunk_sequence(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    archive = tmp_path / "archive.txt"

    (tmp_path / "archive1").write_text("part1\n", encoding="utf-8")
    (tmp_path / "archive3").write_text("part3\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="expected archive2, found archive3",
    ):
        module.join_archive(archive)


def test_split_removes_stale_chunks_before_writing_new_ones(tmp_path: Path) -> None:
    module = _load_project_pack_module()
    archive = tmp_path / "all.txt"
    archive.write_text("line1\nline2\n", encoding="utf-8")

    stale_paths = [
        tmp_path / "all1",
        tmp_path / "all2",
        tmp_path / "all5",
        tmp_path / "all6",
    ]
    for stale_path in stale_paths:
        stale_path.write_text("stale\n", encoding="utf-8")

    chunk_paths = module.split_archive(archive, chunk_size=100, chunk_prefix="all")

    assert [path.name for path in chunk_paths] == ["all1"]
    assert sorted(
        path.name for path in tmp_path.iterdir() if path.name.startswith("all")
    ) == [
        "all.txt",
        "all1",
    ]
    assert (tmp_path / "all1").read_text(encoding="utf-8") == "line1\nline2\n"


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
