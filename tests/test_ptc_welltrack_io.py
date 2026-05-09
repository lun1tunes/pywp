from __future__ import annotations

from pywp.ptc_welltrack_io import decode_welltrack_payload, read_welltrack_file


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


def test_read_welltrack_file_reports_empty_path() -> None:
    warnings: list[str] = []

    text = read_welltrack_file("", warning=warnings.append)

    assert text == ""
    assert warnings == ["Укажите путь к файлу WELLTRACK."]
