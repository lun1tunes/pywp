from __future__ import annotations

import pytest

from pywp import ptc_batch_summary_panel as panel
from pywp.coordinate_integration import DEFAULT_CRS


@pytest.mark.parametrize(
    "cache_key,export_kind,export_format",
    [
        ("wt_survey_download_all_payload_cache", "survey", "Excel"),
        ("wt_survey_download_selected_payload_cache", "survey", "CSV"),
        ("wt_export_package_zip_payload_cache", "package", "zip"),
        ("wt_export_package_files_payload_cache", "package", "folder"),
    ],
)
@pytest.mark.parametrize("previous_version", [None, 2])
def test_prepared_export_cache_rebuilds_legacy_files_once(
    cache_key: str,
    export_kind: str,
    export_format: str,
    previous_version: int | None,
) -> None:
    items = ("calculated-pilot-and-sidetrack",)
    # Exact signature used before source N/E became mandatory in survey exports.
    legacy_signature = (
        export_kind,
        export_format,
        str(DEFAULT_CRS.value),
        str(DEFAULT_CRS.value),
        True,
        (),
        items,
    )
    if previous_version is not None:
        legacy_signature = (previous_version, *legacy_signature)
    state = {cache_key: {"signature": legacy_signature, "payload": b"old-file"}}
    signature = panel._download_signature(
        export_kind=export_kind,
        export_format=export_format,
        target_crs=DEFAULT_CRS,
        source_crs=DEFAULT_CRS,
        auto_convert=True,
        item_signature=items,
    )
    calls = []

    def build_payload() -> bytes:
        calls.append(True)
        return b"complete-source-coordinates"

    for _ in range(2):
        assert panel._download_payload_from_state_cache(
            state=state,
            cache_key=cache_key,
            signature=signature,
            build_payload=build_payload,
        ) == b"complete-source-coordinates"
    assert calls == [True]
