from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
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
    assert session_state["wt_source_upload_file"] is None
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
    assert session_state["wt_source_dev_inline"] == ""
    assert session_state["wt_source_dev_upload_files"] == []


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


def test_normalize_source_table_df_resets_non_range_index_for_dynamic_editor() -> (
    None
):
    source = pd.DataFrame(
        [
            {"Wellname": "TAB-01", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {
                "Wellname": "TAB-01",
                "Point": "t1",
                "X": 600.0,
                "Y": 800.0,
                "Z": 2400.0,
            },
        ],
        index=[0, 2],
    )

    normalized = target_import.normalize_source_table_df_for_ui(source)

    assert isinstance(normalized.index, pd.RangeIndex)
    assert list(normalized.index) == [0, 1]


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


def test_target_import_operation_parses_pilot_target_table_rows() -> None:
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
                    {"Wellname": "TAB-01_PL", "Point": "S", "X": 0, "Y": 0, "Z": 0},
                    {
                        "Wellname": "TAB-01_PL",
                        "Point": "PL1",
                        "X": 300,
                        "Y": 400,
                        "Z": 1600,
                    },
                ]
            ),
        )
    )

    records = operation.parse_records()

    assert [record.name for record in records] == ["TAB-01", "TAB-01_PL"]
    assert len(records[1].points) == 2
    assert records[1].points[1].md == pytest.approx(1.0)


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


def test_target_import_operation_parses_dev_trajectory_from_inline_text() -> None:
    dev_text = Path(
        "tests/test_data/dev_target_import/j_profile_variable_pi.dev"
    ).read_text(encoding="utf-8")
    operation = target_import.build_target_import_operation(
        target_import.WelltrackSourcePayload(
            mode=target_import.WT_SOURCE_MODE_INLINE_TEXT,
            source_format=target_import.WT_SOURCE_FORMAT_DEV_TRAJECTORY,
            source_text=dev_text,
        )
    )

    parsed = operation.parse()

    assert [record.name for record in parsed.records] == ["j_profile_variable_pi"]
    assert len(parsed.dev_summaries) == 1
    assert [well.name for well in parsed.imported_dev_wells] == [
        "j_profile_variable_pi"
    ]
    summary = parsed.dev_summaries[0]
    assert summary.profile_label == "J-профиль"
    assert summary.build1_dls_deg_per_30m == (1.2, 2.4)
    assert summary.build2_dls_deg_per_30m == ()
    assert summary.horizontal_dls_deg_per_30m == ()
    assert summary.kop_md_m == pytest.approx(620.0)
    assert summary.t1_md_m == pytest.approx(1595.0)


def test_target_import_operation_parses_dev_trajectory_directory() -> None:
    operation = target_import.build_target_import_operation(
        target_import.WelltrackSourcePayload(
            mode=target_import.WT_SOURCE_MODE_FILE_PATH,
            source_format=target_import.WT_SOURCE_FORMAT_DEV_TRAJECTORY,
            source_path="tests/test_data/dev_target_import",
        )
    )

    parsed = operation.parse()

    assert len(parsed.records) == 4
    assert [summary.well_name for summary in parsed.dev_summaries] == [
        "build_hold_build_equal_pi_with_horizontal_pi",
        "build_hold_build_split_pi",
        "j_profile_constant_pi",
        "j_profile_variable_pi",
    ]
    assert [well.name for well in parsed.imported_dev_wells] == [
        "build_hold_build_equal_pi_with_horizontal_pi",
        "build_hold_build_split_pi",
        "j_profile_constant_pi",
        "j_profile_variable_pi",
    ]


def test_dev_target_import_summary_dataframe_formats_pi_columns() -> None:
    summary_df = target_import.dev_target_import_summary_dataframe(
        (
            target_import.DevTargetImportSummary(
                well_name="DEV-1",
                profile_label="BUILD-HOLD-BUILD",
                kop_md_m=550.0,
                t1_md_m=2500.0,
                t3_md_m=2980.0,
                entry_inc_deg=84.0,
                build1_dls_deg_per_30m=(2.4,),
                build2_dls_deg_per_30m=(2.4,),
                horizontal_dls_deg_per_30m=(1.2,),
                note="—",
            ),
        )
    )

    assert summary_df.iloc[0]["BUILD1 PI, deg/10m"] == "0.80"
    assert summary_df.iloc[0]["HORIZONTAL PI, deg/10m"] == "0.40"
    assert "BUILD1 DLS, deg/30m" not in summary_df.columns
    assert "BUILD2 DLS, deg/30m" not in summary_df.columns
    assert "HORIZONTAL DLS, deg/30m" not in summary_df.columns


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
    assert session_state[target_import.IMPORTED_DEV_TARGET_WELLS_STATE_KEY] == ()
    assert session_state["wt_well_calc_overrides_enabled"] is False
    assert session_state["wt_well_calc_overrides"] == {}
    assert session_state["wt_well_calc_profile_assignments"] == {}
    assert session_state["wt_well_calc_active_profile_id"] == ""
    assert session_state["wt_well_calc_active_profile_id_pending"] is None
    assert session_state["wt_well_calc_profile_import_upload"] is None
    assert events == [
        "clear_t1_t3",
        "clear_pad",
        "clear_results",
        "auto_layout:TAB-01",
    ]


def test_store_imported_records_keeps_pilot_internal_to_parent_selection() -> None:
    parent = WelltrackRecord(
        name="WELL-04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1000.0),
            WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=1500.0),
        ),
    )
    pilot = WelltrackRecord(
        name="WELL-04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=80.0, y=0.0, z=800.0, md=800.0),
        ),
    )
    session_state: dict[str, object] = {}

    result = target_import.store_imported_records(
        session_state,
        records=[parent, pilot],
        loaded_at_text="2026-05-08 12:00:00",
        clear_t1_t3_order_state=lambda: None,
        clear_pad_state=lambda: None,
        clear_results=lambda: None,
        auto_apply_pad_layout=lambda imported: False,
    )

    assert result.well_names == ("WELL-04", "WELL-04_PL")
    assert session_state["wt_records"] == [parent, pilot]
    assert session_state["wt_selected_names"] == ["WELL-04"]


def test_failed_and_clear_import_state_helpers_reset_expected_keys() -> None:
    session_state: dict[str, object] = {
        "wt_records": ["old"],
        "wt_records_original": ["old"],
        "wt_reference_wells": ("actual", "approved"),
        "wt_reference_actual_wells": ("actual",),
        "wt_reference_approved_wells": ("approved",),
        "wt_selected_names": ["TAB-01"],
        "wt_loaded_at": "old",
        "wt_well_calc_overrides_enabled": True,
        "wt_well_calc_overrides": {"TAB-01": {"values": {"dls_build_max": 0.8}}},
        "wt_well_calc_profile_assignments": {"TAB-01": "cfg-1"},
        "wt_well_calc_active_profile_id": "cfg-1",
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
    assert session_state[target_import.IMPORTED_DEV_TARGET_WELLS_STATE_KEY] == ()
    assert session_state["wt_well_calc_overrides_enabled"] is False
    assert session_state["wt_well_calc_overrides"] == {}
    assert session_state["wt_well_calc_profile_assignments"] == {}
    assert session_state["wt_well_calc_active_profile_id"] == ""
    assert session_state["wt_well_calc_active_profile_id_pending"] is None
    assert session_state["wt_well_calc_profile_import_upload"] is None
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
    assert session_state[target_import.IMPORTED_DEV_TARGET_WELLS_STATE_KEY] == ()
    assert session_state["wt_reference_actual_wells"] == ()
    assert session_state["wt_reference_approved_wells"] == ()
    assert session_state["wt_selected_names"] == []
    assert session_state["wt_loaded_at"] == ""
    assert session_state["wt_well_calc_overrides_enabled"] is False
    assert session_state["wt_well_calc_overrides"] == {}
    assert session_state["wt_well_calc_profile_assignments"] == {}
    assert session_state["wt_well_calc_active_profile_id"] == ""
    assert session_state["wt_well_calc_active_profile_id_pending"] is None
    assert session_state["wt_well_calc_profile_import_upload"] is None
    assert events[-3:] == ["clear_t1_t3_again", "clear_pad_again", "clear_results"]
