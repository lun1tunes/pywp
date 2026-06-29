from __future__ import annotations

from pywp.ptc_welltrack_io import (
    decode_welltrack_payload,
    read_welltrack_file,
    read_welltrack_sources,
)


def test_decode_welltrack_payload_reports_non_utf8_fallback() -> None:
    messages: list[str] = []

    text = decode_welltrack_payload(
        "WELLTRACK 'ТЕСТ'\n0 0 0 0\n/\n".encode("cp1251"),
        source_label="payload",
        info=messages.append,
    )

    assert "ТЕСТ" in text
    assert len(messages) == 1
    assert "cp1251" in messages[0]


def test_read_welltrack_file_resolves_relative_path(tmp_path) -> None:
    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "wells.inc").write_text(
        "WELLTRACK 'A'\n0 0 0 0\n/\n",
        encoding="utf-8",
    )

    text = read_welltrack_file("inputs/wells.inc", cwd=tmp_path)

    assert "WELLTRACK 'A'" in text


def test_read_welltrack_file_accepts_trailing_quote_in_path(tmp_path) -> None:
    folder = tmp_path / "inputs"
    folder.mkdir()
    source_file = folder / "quoted.inc"
    source_file.write_text(
        "WELLTRACK 'Q'\n0 0 0 0\n/\n",
        encoding="utf-8",
    )

    text = read_welltrack_file(f'{source_file}"')

    assert "WELLTRACK 'Q'" in text


def test_read_welltrack_file_reports_empty_path() -> None:
    warnings: list[str] = []

    text = read_welltrack_file("", warning=warnings.append)

    assert text == ""
    assert warnings == ["Укажите путь к файлу WELLTRACK."]


def test_read_welltrack_sources_accepts_file_and_folder_with_case_insensitive_inc(
    tmp_path,
) -> None:
    direct_file = tmp_path / "direct.inc"
    direct_file.write_text("WELLTRACK 'DIRECT'\n0 0 0 0\n/\n", encoding="utf-8")
    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "a.InC").write_text(
        "WELLTRACK 'FOLDER-A'\n0 0 0 0\n/\n",
        encoding="utf-8",
    )
    (folder / "b.INC").write_text(
        "WELLTRACK 'FOLDER-B'\n0 0 0 0\n/\n",
        encoding="utf-8",
    )
    (folder / "skip.txt").write_text("ignore", encoding="utf-8")

    text = read_welltrack_sources([str(direct_file), str(folder)], cwd=tmp_path)

    assert "WELLTRACK 'DIRECT'" in text
    assert "WELLTRACK 'FOLDER-A'" in text
    assert "WELLTRACK 'FOLDER-B'" in text
    assert "skip" not in text
