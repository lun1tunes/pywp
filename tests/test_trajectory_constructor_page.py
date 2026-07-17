from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pywp import ptc_core
from pywp import ptc_page_reference
from pywp import ptc_page_state
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import TrajectoryConfig
from pywp.reference_trajectories import parse_reference_trajectory_table
from pywp.welltrack_batch import SuccessfulWellPlan


pytestmark = pytest.mark.integration


def _records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=20.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=620.0, y=820.0, z=2410.0, md=2410.0),
                WelltrackPoint(x=1520.0, y=2020.0, z=2510.0, md=3510.0),
            ),
        ),
    ]


def _successful_plan(*, name: str, y_offset_m: float) -> SuccessfulWellPlan:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 1000.0, 2000.0],
            "INC_deg": [0.0, 90.0, 90.0],
            "AZI_deg": [0.0, 90.0, 90.0],
            "X_m": [0.0, 1000.0, 2000.0],
            "Y_m": [y_offset_m, y_offset_m, y_offset_m],
            "Z_m": [0.0, 0.0, 0.0],
            "DLS_deg_per_30m": [0.0, 0.0, 0.0],
            "segment": ["VERTICAL", "BUILD1", "HORIZONTAL"],
        }
    )
    return SuccessfulWellPlan(
        name=name,
        surface={"x": 0.0, "y": y_offset_m, "z": 0.0},
        t1={"x": 1000.0, "y": y_offset_m, "z": 0.0},
        t3={"x": 2000.0, "y": y_offset_m, "z": 0.0},
        stations=stations,
        summary={
            "trajectory_type": "Unified J Profile + Build + Azimuth Turn",
            "trajectory_target_direction": "Цели в одном направлении",
            "well_complexity": "Обычная",
            "optimization_mode": "minimize_md",
            "azimuth_turn_deg": 0.0,
            "horizontal_length_m": 1000.0,
            "entry_inc_deg": 90.0,
            "hold_inc_deg": 90.0,
            "build_dls_selected_deg_per_30m": 3.0,
            "build1_dls_selected_deg_per_30m": 3.0,
            "build2_dls_selected_deg_per_30m": 3.0,
            "max_dls_total_deg_per_30m": 3.0,
            "kop_md_m": 560.0,
            "max_inc_actual_deg": 90.0,
            "max_inc_deg": 95.0,
            "md_total_m": 2000.0,
            "max_total_md_postcheck_m": 6500.0,
            "md_postcheck_excess_m": 0.0,
        },
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        config=TrajectoryConfig(),
    )


def _reference_wells():
    return parse_reference_trajectory_table(
        [
            {
                "Wellname": "FACT-001",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "FACT-001",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 1200.0,
                "MD": 1200.0,
            },
            {
                "Wellname": "FACT-001",
                "Type": "actual",
                "X": 600.0,
                "Y": 0.0,
                "Z": 1300.0,
                "MD": 1900.0,
            },
            {
                "Wellname": "APP-001",
                "Type": "approved",
                "X": 30.0,
                "Y": 50.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "APP-001",
                "Type": "approved",
                "X": 30.0,
                "Y": 50.0,
                "Z": 1250.0,
                "MD": 1250.0,
            },
            {
                "Wellname": "APP-001",
                "Type": "approved",
                "X": 700.0,
                "Y": 80.0,
                "Z": 1360.0,
                "MD": 2050.0,
            },
        ]
    )


def test_ptc_page_shows_user_facing_import_and_run_controls() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.run()

    button_labels = {str(widget.label) for widget in at.button}
    assert "Импорт целей" in button_labels
    assert "Очистить импорт" not in button_labels


def test_ptc_page_uses_automatic_parallel_worker_selection() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    at.session_state["wt_records"] = _records()
    at.session_state["wt_records_original"] = _records()

    at.run(timeout=120)

    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    assert "Параллельный расчёт" not in selectbox_labels
    caption_values = [str(widget.value) for widget in at.caption]
    assert any("Multiprocessing" in value for value in caption_values)


def test_ptc_page_keeps_open_calc_params_panel_after_three_multi_edit_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = False

    def _fake_three_scene(_payload, **_kwargs):
        nonlocal emitted
        if emitted:
            return None
        emitted = True
        return {
            "type": "pywp:editTargets",
            "nonce": "multi-edit-1",
            "changes": [
                {
                    "name": "WELL-A",
                    "points": [
                        {"index": 1, "position": [610.0, 810.0, 2400.0]},
                        {"index": 2, "position": [1510.0, 2010.0, 2500.0]},
                    ],
                },
                {
                    "name": "WELL-B",
                    "points": [
                        {"index": 1, "position": [630.0, 830.0, 2410.0]},
                        {"index": 2, "position": [1530.0, 2030.0, 2510.0]},
                    ],
                },
            ],
        }

    monkeypatch.setattr(ptc_core, "render_local_three_scene", _fake_three_scene)

    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"
    at.session_state[ptc_page_state.PTC_CALC_PARAMS_OPEN_KEY] = True

    at.run(timeout=120)

    assert not at.exception
    button_labels = {str(widget.label) for widget in at.button}
    assert "Скрыть" in button_labels
    assert at.session_state[ptc_page_state.PTC_CALC_PARAMS_OPEN_KEY] is True
    assert at.session_state["wt_edit_targets_pending_names"] == ["WELL-A", "WELL-B"]


def test_ptc_page_hides_engineering_result_controls_and_single_well_debug_sections() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "OK", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-A", y_offset_m=0.0),
        _successful_plan(name="WELL-B", y_offset_m=25.0),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"

    at.run(timeout=120)

    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    button_labels = {str(widget.label) for widget in at.button}
    radio_labels = {str(widget.label) for widget in at.radio}
    assert "3D-режим отображения" not in selectbox_labels
    assert "3D backend" not in selectbox_labels
    assert "Пересоздать 3D viewer" not in button_labels
    assert "Режим отображения всех скважин" not in radio_labels

    view_mode_radio = next(
        widget for widget in at.radio if str(widget.label) == "Режим просмотра результатов"
    )
    view_mode_radio.set_value("Отдельная скважина")
    at.run(timeout=120)

    expander_labels = {str(widget.label) for widget in at.expander}
    assert "Контроль попадания и точность расчета" not in expander_labels
    assert "Технические параметры и диагностика решателя" not in expander_labels


def test_ptc_page_defers_anticollision_when_three_edits_are_pending() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "Не рассчитана", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-B", "Статус": "OK", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-C", "Статус": "OK", "Проблема": "", "Точек": 3},
    ]
    at.session_state["wt_successes"] = [
        _successful_plan(name="WELL-B", y_offset_m=25.0),
        _successful_plan(name="WELL-C", y_offset_m=50.0),
    ]
    at.session_state["wt_results_view_mode"] = "Все скважины"
    at.session_state["wt_results_all_view_mode"] = "Anti-collision"
    at.session_state["wt_edit_targets_pending_names"] = ["WELL-A"]
    at.session_state["wt_pending_selected_names"] = ["WELL-A"]

    at.run(timeout=120)

    radio_labels = {str(widget.label) for widget in at.radio}
    assert "Режим отображения всех скважин" not in radio_labels
    assert str(at.session_state["wt_results_all_view_mode"]) == "Anti-collision"
    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    metric_labels = {str(widget.label) for widget in at.metric}
    assert "Пресет неопределенности для anti-collision" not in selectbox_labels
    assert "Проверено пар" not in metric_labels


def test_ptc_page_wraps_pad_layout_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "@st.fragment" in source
    assert "def _render_pad_layout_section(records: list[object]) -> None:" in source
    assert "_render_pad_layout_section(records=records)" in source


def test_ptc_page_wraps_target_import_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_target_import_section_fragment() -> None:" in source
    assert "render_target_import_section()" in source
    assert "_render_target_import_section_fragment()" in source


def test_ptc_core_contains_bulk_horizontal_length_preprocess_controls() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "Изменить длину ГС" in source
    assert "Скважины для изменения длины ГС" in source
    assert "Новая длина ГС, м" in source
    assert '"wt_preprocess_select_all"' in source
    assert '"wt_preprocess_only_pad"' in source
    assert '"Применить"' in source


def test_ptc_core_keeps_explicit_pilot_and_zbs_name_matching_guidance() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "Имя `well` должно совпадать с основной скважиной." in source
    assert "`fact_well` должно совпадать с именем загруженной " in source
    assert "фактической скважины." in source
    assert "Есть ЗБС: для расчёта загрузите " in source
    assert 'фактическую основную скважину "' in source


def test_ptc_core_keeps_auto_order_guardrails_for_source_defined_wellheads() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "Разрешить редактирование позиций куста" in source
    assert "Применить авто-порядок" in source
    assert "disabled=not allow_source_surface_edit" in source
    assert "source_surfaces_defined and not allow_source_surface_edit" in source


def test_ptc_core_uses_stateful_pad_layout_details_panel() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert 'key="wt_pad_layout_details_toggle"' in source
    assert "Скрыть настройки выбранного куста" in source
    assert "Настроить положение куста, НДС и расстояние между устьями." in source
    assert "ptc_pad_state.pad_layout_details_open(st.session_state)" in source


def test_ptc_anticollision_params_limit_multiselect_height_via_scoped_container() -> None:
    source = Path("pywp/ptc_anticollision_params.py").read_text(encoding="utf-8")

    assert 'st.container(key=_REFERENCE_UNCERTAINTY_WIDGETS_CONTAINER_KEY)' in source
    assert ".st-key-wt_anticollision_reference_uncertainty_widgets" in source
    assert "max-height: 3.35rem;" in source
    assert "overflow-y: auto;" in source


def test_ptc_page_wraps_reference_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_reference_section_fragment() -> None:" in source
    assert "render_reference_section()" in source
    assert "_render_reference_section_fragment()" in source


def test_ptc_page_wraps_records_overview_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_records_overview_section(records: list[object]) -> None:" in source
    assert "wt._render_records_overview(records=records)" in source
    assert "_render_records_overview_section(records=records)" in source


def test_ptc_page_wraps_raw_records_section_in_fragment() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_raw_records_section(records: list[object]) -> None:" in source
    assert "wt._render_raw_records_table(records=records)" in source
    assert "_render_raw_records_section(records=records)" in source


def test_ptc_page_extracts_results_section_helper() -> None:
    source = Path("pywp/ptc_page.py").read_text(encoding="utf-8")

    assert "def _render_results_section(" in source
    assert "@st.fragment\ndef _render_results_section(" in source
    assert 'st.markdown("## 5. Результаты расчёта")' in source
    assert "render_success_tabs(" in source
    assert "_render_results_section(" in source


def test_ptc_page_run_wraps_run_section_in_fragment() -> None:
    source = Path("pywp/ptc_page_run.py").read_text(encoding="utf-8")

    assert "@st.fragment\ndef render_run_section(*, records: list[object]) -> None:" in source
    assert "def _rerun_app() -> None:" in source
    assert "_rerun_app()" in source


def test_ptc_fragment_sections_use_fragment_scoped_reruns_for_local_ui_updates() -> None:
    core_source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")
    reference_source = Path("pywp/ptc_page_reference.py").read_text(encoding="utf-8")
    run_source = Path("pywp/ptc_page_run.py").read_text(encoding="utf-8")
    results_source = Path("pywp/ptc_page_results.py").read_text(encoding="utf-8")

    assert 'st.rerun(scope="fragment")' in core_source
    assert 'st.rerun(scope="fragment")' in reference_source
    assert 'st.rerun(scope="fragment")' in run_source
    assert 'st.rerun(scope="fragment")' in results_source


def test_parse_reference_sources_requires_explicit_uploaded_welltrack_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    uploaded_file = SimpleNamespace(
        name="uploaded.inc",
        getvalue=lambda: b"WELLTRACK",
    )
    decode_calls: list[str] = []

    monkeypatch.setattr(
        page,
        "_pending_mixed_legacy_reference_sources",
        lambda: None,
    )
    monkeypatch.setattr(
        page,
        "_reference_uploaded_sources_by_kind",
        lambda _uploaded_files: {
            page._REFERENCE_FUND_OPTIONS[0]: [uploaded_file],
            page._REFERENCE_FUND_OPTIONS[1]: [],
        },
    )
    monkeypatch.setattr(
        page.ptc_welltrack_io,
        "decode_welltrack_payload",
        lambda *_args, **_kwargs: decode_calls.append("decode") or "decoded",
    )
    monkeypatch.setattr(
        page,
        "parse_reference_trajectory_welltrack_text",
        lambda *_args, **_kwargs: ("parsed",),
    )

    with pytest.raises(
        ptc_core.WelltrackParseError,
        match="Неподдерживаемый режим импорта фонда",
    ):
        page._parse_reference_sources(
            mode="Неизвестный режим",
            uploaded_files=[uploaded_file],
        )

    assert decode_calls == []


def test_parse_reference_sources_handles_uploaded_welltrack_only_for_explicit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    uploaded_file = SimpleNamespace(
        name="uploaded.inc",
        getvalue=lambda: b"WELLTRACK",
    )

    monkeypatch.setattr(
        page,
        "_pending_mixed_legacy_reference_sources",
        lambda: None,
    )
    monkeypatch.setattr(
        page,
        "_reference_uploaded_sources_by_kind",
        lambda _uploaded_files: {
            page._REFERENCE_FUND_OPTIONS[0]: [uploaded_file],
            page._REFERENCE_FUND_OPTIONS[1]: [],
        },
    )
    monkeypatch.setattr(
        page.ptc_welltrack_io,
        "decode_welltrack_payload",
        lambda *_args, **_kwargs: "decoded",
    )
    monkeypatch.setattr(
        page,
        "parse_reference_trajectory_welltrack_text",
        lambda payload, *, kind: (f"{kind}:{payload}",),
    )

    parsed = page._parse_reference_sources(
        mode="Загрузить WELLTRACK",
        uploaded_files=[uploaded_file],
    )

    assert parsed[page._REFERENCE_FUND_OPTIONS[0]] == ("actual:decoded",)
    assert parsed[page._REFERENCE_FUND_OPTIONS[1]] == ()


def test_reference_import_migration_preserves_legacy_uploaded_welltrack_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    actual_kind, approved_kind = page._REFERENCE_FUND_OPTIONS
    page.st.session_state[page.reference_state.reference_source_mode_key(actual_kind)] = (
        "Загрузить WELLTRACK"
    )
    page.st.session_state[
        page.reference_state.reference_source_mode_key(approved_kind)
    ] = "Путь к WELLTRACK"
    page.st.session_state[
        page.reference_state.reference_welltrack_path_key(approved_kind)
    ] = "/tmp/approved.inc"

    original_dev_paths = page.reference_state.reference_dev_folder_paths

    def _dev_paths(kind: str) -> tuple[str, ...]:
        if kind == actual_kind:
            raise AssertionError("legacy upload mode should not read .dev paths")
        return original_dev_paths(kind)

    monkeypatch.setattr(
        page.reference_state,
        "reference_dev_folder_paths",
        _dev_paths,
    )

    page._migrate_legacy_reference_import_state()

    assert (
        str(page.st.session_state[page._REFERENCE_IMPORT_MODE_KEY])
        == "Загрузить WELLTRACK"
    )
    assert (
        str(page.st.session_state[page._reference_welltrack_source_path_key(0)])
        == "/tmp/approved.inc"
    )


def test_reference_import_migration_defaults_to_uploaded_welltrack_mode() -> None:
    page = ptc_page_reference
    page.st.session_state.clear()
    actual_kind = page._REFERENCE_FUND_OPTIONS[0]
    page.st.session_state[page.reference_state.reference_source_mode_key(actual_kind)] = (
        "Загрузить WELLTRACK"
    )

    page._migrate_legacy_reference_import_state()

    assert (
        str(page.st.session_state[page._REFERENCE_IMPORT_MODE_KEY])
        == "Загрузить WELLTRACK"
    )


def test_ptc_core_keeps_full_rerun_after_successful_target_import() -> None:
    source = Path("pywp/ptc_core.py").read_text(encoding="utf-8")

    assert "label=operation.success_label(elapsed)," in source
    assert "st.rerun()" in source


def test_ptc_page_renders_target_editor_when_all_results_failed() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {
            "Скважина": "WELL-A",
            "Статус": "Ошибка расчета",
            "Проблема": "endpoint miss",
            "Точек": 3,
        },
        {
            "Скважина": "WELL-B",
            "Статус": "Ошибка расчета",
            "Проблема": "endpoint miss",
            "Точек": 3,
        },
    ]
    at.session_state["wt_successes"] = []

    at.run(timeout=120)

    warning_values = [str(widget.value) for widget in at.warning]
    markdown_values = [str(widget.value) for widget in at.markdown]
    assert any(
        "Все выбранные скважины завершились ошибками" in value
        for value in warning_values
    )
    assert any("Исходные точки для правки" in value for value in markdown_values)


def test_ptc_page_shows_recalc_info_when_all_results_are_not_run_after_target_edit() -> (
    None
):
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_summary_rows"] = [
        {"Скважина": "WELL-A", "Статус": "Не рассчитана", "Проблема": "", "Точек": 3},
        {"Скважина": "WELL-B", "Статус": "Не рассчитана", "Проблема": "", "Точек": 3},
    ]
    at.session_state["wt_successes"] = []
    at.session_state["wt_edit_targets_pending_names"] = ["WELL-A", "WELL-B"]

    at.run(timeout=120)

    info_values = [str(widget.value) for widget in at.info]
    warning_values = [str(widget.value) for widget in at.warning]
    markdown_values = [str(widget.value) for widget in at.markdown]
    assert any("Скважины требуют пересчёта" in value for value in info_values)
    assert not any(
        "Все выбранные скважины завершились ошибками" in value
        for value in warning_values
    )
    assert any("Исходные точки для правки" in value for value in markdown_values)


def test_ptc_page_wraps_reference_well_table_into_expander() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    reference_wells = _reference_wells()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_reference_actual_wells"] = [
        well for well in reference_wells if well.kind == "actual"
    ]
    at.session_state["wt_reference_approved_wells"] = [
        well for well in reference_wells if well.kind == "approved"
    ]

    at.run()

    expander_labels = {str(widget.label) for widget in at.expander}
    assert "Список загруженных фактических/ проектных скважин" in expander_labels


def test_ptc_page_renders_approved_reference_well_detail_viewer() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _records()
    reference_wells = _reference_wells()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state["wt_reference_actual_wells"] = [
        well for well in reference_wells if well.kind == "actual"
    ]
    at.session_state["wt_reference_approved_wells"] = [
        well for well in reference_wells if well.kind == "approved"
    ]
    at.session_state["wt_show_actual_fund_analysis"] = True
    at.session_state["wt_show_approved_fund_analysis"] = True

    at.run(timeout=120)

    expander_labels = {str(widget.label) for widget in at.expander}
    assert "Просмотр загруженных утверждённых проектных скважин" in expander_labels

    selectbox_labels = {str(widget.label) for widget in at.selectbox}
    assert "Просмотр фактической скважины" in selectbox_labels
    assert "Просмотр утвержденной проектной скважины" in selectbox_labels
