from __future__ import annotations

import pandas as pd
import pytest

from pywp import ptc_target_import as target_import


def test_target_source_defaults_preserve_legacy_table_mode() -> None:
    session_state = {"wt_source_mode": target_import.WT_SOURCE_MODE_TARGET_TABLE}

    target_import.init_target_source_state_defaults(session_state)

    assert (
        session_state["wt_source_format"]
        == target_import.WT_SOURCE_FORMAT_TARGET_TABLE
    )
    assert session_state["wt_source_path"] == str(
        target_import.DEFAULT_WELLTRACK_PATH
    )
    assert list(session_state["wt_source_table_df"].columns) == [
        "Wellname",
        "Point",
        "X",
        "Y",
        "Z",
    ]


def test_target_source_defaults_normalize_unknown_format() -> None:
    session_state = {"wt_source_format": "legacy-invalid"}

    target_import.init_target_source_state_defaults(session_state)

    assert session_state["wt_source_format"] == target_import.WT_SOURCE_FORMAT_WELLTRACK
    assert session_state["wt_source_mode"] == target_import.WT_SOURCE_MODE_FILE_PATH


def test_normalize_source_table_df_accepts_aliases_and_surface_names() -> None:
    normalized = target_import.normalize_source_table_df_for_ui(
        pd.DataFrame(
            [
                {
                    "well name": "TAB-01",
                    "точка": "wellhead",
                    "east": 10.0,
                    "north": 20.0,
                    "tvd": 30.0,
                },
                {
                    "well name": "TAB-01",
                    "точка": "s",
                    "east": 11.0,
                    "north": 21.0,
                    "tvd": 31.0,
                },
            ]
        )
    )

    assert list(normalized.columns) == ["Wellname", "Point", "X", "Y", "Z"]
    assert list(normalized["Point"]) == ["S", "S"]
    assert list(normalized["X"]) == [10.0, 11.0]


def test_normalize_source_table_df_accepts_excel_like_single_column_rows() -> None:
    normalized = target_import.normalize_source_table_df_for_ui(
        pd.DataFrame(
            {
                "Column 1": [
                    "TAB-01\tS\t0\t0\t0",
                    "TAB-01;t1;600,5;800,25;2400,75",
                    "TAB-01\tt3\t1500\t2000\t2500",
                ]
            }
        )
    )

    assert normalized.iloc[0].to_dict()["Point"] == "S"
    assert str(normalized.iloc[1]["X"]) == "600,5"
    assert list(normalized.columns) == ["Wellname", "Point", "X", "Y", "Z"]


def test_target_import_operation_parses_target_table_rows() -> None:
    operation = target_import.build_target_import_operation(
        target_import.WelltrackSourcePayload(
            mode=target_import.WT_SOURCE_MODE_TARGET_TABLE,
            table_rows=pd.DataFrame(
                [
                    {"Wellname": "TAB-01", "Point": "S", "X": 0, "Y": 0, "Z": 0},
                    {
                        "Wellname": "TAB-01",
                        "Point": "t1",
                        "X": 600,
                        "Y": 800,
                        "Z": 2400,
                    },
                    {
                        "Wellname": "TAB-01",
                        "Point": "t3",
                        "X": 1500,
                        "Y": 2000,
                        "Z": 2500,
                    },
                ]
            ),
        )
    )

    records = operation.parse_records()

    assert operation.status_label == "Чтение и преобразование таблицы точек..."
    assert operation.count_message(len(records)) == "Собрано скважин из таблицы: 1."
    assert operation.success_label(1.25) == "Импорт таблицы завершен за 1.25 с"
    assert [record.name for record in records] == ["TAB-01"]


def test_target_import_operation_uses_injected_welltrack_parser() -> None:
    calls: list[str] = []

    def parse_stub(text: str):
        calls.append(text)
        return []

    operation = target_import.build_target_import_operation(
        target_import.WelltrackSourcePayload(
            mode=target_import.WT_SOURCE_MODE_INLINE_TEXT,
            source_text="WELLTRACK 'A' /",
        ),
        parse_welltrack_text_func=parse_stub,
    )

    assert operation.parse_records() == []
    assert calls == ["WELLTRACK 'A' /"]
    assert operation.count_message(3) == "Найдено блоков WELLTRACK: 3."


def test_target_import_operation_rejects_empty_sources() -> None:
    with pytest.raises(target_import.TargetImportEmptySourceError, match="Таблица пуста"):
        target_import.build_target_import_operation(
            target_import.WelltrackSourcePayload(
                mode=target_import.WT_SOURCE_MODE_TARGET_TABLE,
            )
        )

    with pytest.raises(target_import.TargetImportEmptySourceError, match="Источник пустой"):
        target_import.build_target_import_operation(
            target_import.WelltrackSourcePayload(
                mode=target_import.WT_SOURCE_MODE_INLINE_TEXT,
                source_text="  ",
            )
        )


def test_store_imported_records_mutates_state_and_runs_callbacks() -> None:
    records = target_import.build_target_import_operation(
        target_import.WelltrackSourcePayload(
            mode=target_import.WT_SOURCE_MODE_TARGET_TABLE,
            table_rows=pd.DataFrame(
                [
                    {"Wellname": "TAB-01", "Point": "S", "X": 0, "Y": 0, "Z": 0},
                    {
                        "Wellname": "TAB-01",
                        "Point": "t1",
                        "X": 600,
                        "Y": 800,
                        "Z": 2400,
                    },
                    {
                        "Wellname": "TAB-01",
                        "Point": "t3",
                        "X": 1500,
                        "Y": 2000,
                        "Z": 2500,
                    },
                ]
            ),
        )
    ).parse_records()
    session_state: dict[str, object] = {}
    events: list[str] = []

    result = target_import.store_imported_records(
        session_state,
        records=records,
        loaded_at_text="2026-05-08 12:00:00",
        clear_t1_t3_order_state=lambda: events.append("clear_t1_t3"),
        clear_pad_state=lambda: events.append("clear_pad"),
        clear_results=lambda: events.append("clear_results"),
        auto_apply_pad_layout=lambda imported: bool(
            events.append(f"auto_layout:{imported[0].name}") or True
        ),
    )

    assert result.well_names == ("TAB-01",)
    assert result.auto_layout_applied is True
    assert session_state["wt_records"] == records
    assert session_state["wt_records_original"] == records
    assert session_state["wt_loaded_at"] == "2026-05-08 12:00:00"
    assert session_state["wt_last_error"] == ""
    assert session_state["wt_selected_names"] == ["TAB-01"]
    assert events == [
        "clear_t1_t3",
        "clear_pad",
        "clear_results",
        "auto_layout:TAB-01",
    ]


def test_failed_and_clear_import_state_helpers_reset_expected_keys() -> None:
    session_state: dict[str, object] = {
        "wt_records": ["old"],
        "wt_records_original": ["old"],
        "wt_reference_wells": ("actual", "approved"),
        "wt_reference_actual_wells": ("actual",),
        "wt_reference_approved_wells": ("approved",),
        "wt_selected_names": ["TAB-01"],
        "wt_loaded_at": "old",
    }
    events: list[str] = []

    target_import.reset_failed_import_state(
        session_state,
        error_message="bad source",
        clear_t1_t3_order_state=lambda: events.append("clear_t1_t3"),
        clear_pad_state=lambda: events.append("clear_pad"),
    )

    assert session_state["wt_records"] is None
    assert session_state["wt_records_original"] is None
    assert session_state["wt_last_error"] == "bad source"
    assert events == ["clear_t1_t3", "clear_pad"]

    target_import.clear_target_import_flow_state(
        session_state,
        reference_well_state_keys=(
            "wt_reference_actual_wells",
            "wt_reference_approved_wells",
        ),
        clear_t1_t3_order_state=lambda: events.append("clear_t1_t3_again"),
        clear_pad_state=lambda: events.append("clear_pad_again"),
        clear_results=lambda: events.append("clear_results"),
    )

    assert session_state["wt_records"] is None
    assert session_state["wt_records_original"] is None
    assert session_state["wt_reference_wells"] == ()
    assert session_state["wt_reference_actual_wells"] == ()
    assert session_state["wt_reference_approved_wells"] == ()
    assert session_state["wt_selected_names"] == []
    assert session_state["wt_loaded_at"] == ""
    assert events[-3:] == ["clear_t1_t3_again", "clear_pad_again", "clear_results"]
