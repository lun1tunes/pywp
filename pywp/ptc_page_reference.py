from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

from pywp import ptc_core as wt
from pywp import ptc_reference_state as reference_state
from pywp import ptc_welltrack_io
from pywp.reference_trajectories import (
    parse_reference_trajectory_dev_directories,
    parse_reference_trajectory_welltrack_text,
)
from pywp.pilot_wells import is_zbs_record, parent_name_for_zbs, well_name_key

__all__ = ["render_reference_section"]

_REFERENCE_ANALYSIS_TOGGLE_KEYS = {
    wt.REFERENCE_WELL_ACTUAL: "wt_show_actual_fund_analysis",
    wt.REFERENCE_WELL_APPROVED: "wt_show_approved_fund_analysis",
}
_REFERENCE_SOURCE_OPTIONS = (
    "Загрузить .dev",
    "Путь к WELLTRACK",
    "Загрузить WELLTRACK",
)
_REFERENCE_FUND_OPTIONS = (
    wt.REFERENCE_WELL_ACTUAL,
    wt.REFERENCE_WELL_APPROVED,
)
_REFERENCE_FUND_LABELS = {
    wt.REFERENCE_WELL_ACTUAL: "Фактический",
    wt.REFERENCE_WELL_APPROVED: "Утверждённый проектный",
}
_REFERENCE_IMPORT_MODE_KEY = "ptc_reference_import_source_mode"
_REFERENCE_IMPORT_MIGRATED_KEY = "ptc_reference_import_state_migrated"
_REFERENCE_UPLOAD_NONCE_KEY = "ptc_reference_upload_nonce"
_REFERENCE_PENDING_LEGACY_DEV_ROWS_KEY = (
    "ptc_reference_pending_legacy_dev_rows"
)
_REFERENCE_PENDING_LEGACY_WELLTRACK_ROWS_KEY = (
    "ptc_reference_pending_legacy_welltrack_rows"
)
_REFERENCE_DEV_SOURCE_COUNT_KEY = "wt_reference_import_dev_source_count"
_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY = (
    "wt_reference_import_welltrack_source_count"
)
_REFERENCE_DEV_SOURCE_HELP = (
    "Импортируются все `.dev` файлы из папок. "
    "Имя берётся из файла без `.dev`, координаты - из колонок `MD X Y Z`."
)


def _reference_analysis_toggle_key(kind: str) -> str:
    return _REFERENCE_ANALYSIS_TOGGLE_KEYS.get(
        str(kind),
        f"wt_show_reference_analysis::{str(kind)}",
    )


def _rerun_fragment() -> None:
    try:
        st.rerun(scope="fragment")
    except (TypeError, StreamlitAPIException):
        st.rerun()


def _reset_reference_analysis_visibility(kind: str) -> None:
    st.session_state[_reference_analysis_toggle_key(kind)] = False


def _after_reference_data_change(kind: str) -> None:
    wt._reset_anticollision_view_state(clear_prepared=True)
    _reset_reference_analysis_visibility(kind)


def _clear_reference_import_state(kind: str) -> None:
    reference_state.clear_reference_import_state(
        kind,
        on_clear=lambda: _after_reference_data_change(kind),
    )


def _normalize_reference_import_mode_state() -> str:
    mode = str(
        st.session_state.get(
            _REFERENCE_IMPORT_MODE_KEY,
            _REFERENCE_SOURCE_OPTIONS[0],
        )
    ).strip()
    if mode not in _REFERENCE_SOURCE_OPTIONS:
        mode = _REFERENCE_SOURCE_OPTIONS[0]
    st.session_state[_REFERENCE_IMPORT_MODE_KEY] = mode
    return mode


def _reference_import_mode() -> str:
    mode = str(
        st.session_state.get(
            _REFERENCE_IMPORT_MODE_KEY,
            _REFERENCE_SOURCE_OPTIONS[0],
        )
    ).strip()
    if mode not in _REFERENCE_SOURCE_OPTIONS:
        mode = _REFERENCE_SOURCE_OPTIONS[0]
    for kind in _REFERENCE_FUND_OPTIONS:
        st.session_state[reference_state.reference_source_mode_key(kind)] = mode
        st.session_state[f"ptc_reference_source_mode::{kind}"] = mode
    return mode


def _reference_dev_source_path_key(index: int) -> str:
    return f"wt_reference_import_dev_source_path_{int(index)}"


def _reference_dev_source_kind_key(index: int) -> str:
    return f"wt_reference_import_dev_source_kind_{int(index)}"


def _reference_welltrack_source_path_key(index: int) -> str:
    return f"wt_reference_import_welltrack_source_path_{int(index)}"


def _reference_welltrack_source_kind_key(index: int) -> str:
    return f"wt_reference_import_welltrack_source_kind_{int(index)}"


def _reference_upload_kind_key(index: int) -> str:
    return f"wt_reference_import_upload_kind_{int(index)}"


def _reference_upload_widget_key() -> str:
    nonce = int(st.session_state.get(_REFERENCE_UPLOAD_NONCE_KEY, 0) or 0)
    return f"ptc_reference_upload_files::{nonce}"


def _normalize_reference_fund_kind(value: object) -> str:
    return (
        wt.REFERENCE_WELL_APPROVED
        if str(value) == wt.REFERENCE_WELL_APPROVED
        else wt.REFERENCE_WELL_ACTUAL
    )


def _reference_fund_label(kind: str) -> str:
    return _REFERENCE_FUND_LABELS.get(
        str(kind),
        _REFERENCE_FUND_LABELS[wt.REFERENCE_WELL_ACTUAL],
    )


def _normalize_count(count_key: str) -> int:
    try:
        count = int(st.session_state.get(count_key, 1))
    except (TypeError, ValueError):
        count = 1
    count = max(1, count)
    st.session_state[count_key] = count
    return count


def _dev_source_rows_present() -> bool:
    count = _normalize_count(_REFERENCE_DEV_SOURCE_COUNT_KEY)
    return count > 1 or any(
        str(st.session_state.get(_reference_dev_source_path_key(index), "")).strip()
        for index in range(count)
    )


def _welltrack_source_rows_present() -> bool:
    count = _normalize_count(_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY)
    return count > 1 or any(
        str(
            st.session_state.get(
                _reference_welltrack_source_path_key(index),
                "",
            )
        ).strip()
        for index in range(count)
    )


def _set_reference_dev_rows(rows: list[tuple[str, str]]) -> None:
    row_count = max(1, len(rows))
    st.session_state[_REFERENCE_DEV_SOURCE_COUNT_KEY] = row_count
    for index in range(row_count):
        path, kind = rows[index] if index < len(rows) else ("", wt.REFERENCE_WELL_ACTUAL)
        st.session_state[_reference_dev_source_path_key(index)] = str(path).strip()
        st.session_state[_reference_dev_source_kind_key(index)] = (
            _normalize_reference_fund_kind(kind)
        )


def _set_reference_welltrack_rows(rows: list[tuple[str, str]]) -> None:
    row_count = max(1, len(rows))
    st.session_state[_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY] = row_count
    for index in range(row_count):
        path, kind = rows[index] if index < len(rows) else ("", wt.REFERENCE_WELL_ACTUAL)
        st.session_state[_reference_welltrack_source_path_key(index)] = (
            str(path).strip()
        )
        st.session_state[_reference_welltrack_source_kind_key(index)] = (
            _normalize_reference_fund_kind(kind)
        )


def _reference_dev_rows() -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    source_count = _normalize_count(_REFERENCE_DEV_SOURCE_COUNT_KEY)
    for index in range(source_count):
        source_path = str(
            st.session_state.get(_reference_dev_source_path_key(index), "")
        ).strip()
        if not source_path:
            continue
        rows.append(
            (
                source_path,
                _normalize_reference_fund_kind(
                    st.session_state.get(_reference_dev_source_kind_key(index))
                ),
            )
        )
    return tuple(rows)


def _reference_welltrack_rows() -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    source_count = _normalize_count(_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY)
    for index in range(source_count):
        source_path = str(
            st.session_state.get(
                _reference_welltrack_source_path_key(index),
                "",
            )
        ).strip()
        if not source_path:
            continue
        rows.append(
            (
                source_path,
                _normalize_reference_fund_kind(
                    st.session_state.get(
                        _reference_welltrack_source_kind_key(index)
                    )
                ),
            )
        )
    return tuple(rows)


def _clear_pending_legacy_reference_rows() -> None:
    st.session_state[_REFERENCE_PENDING_LEGACY_DEV_ROWS_KEY] = ()
    st.session_state[_REFERENCE_PENDING_LEGACY_WELLTRACK_ROWS_KEY] = ()


def _migrate_legacy_reference_import_state() -> None:
    if bool(st.session_state.get(_REFERENCE_IMPORT_MIGRATED_KEY)):
        return

    legacy_dev_rows: list[tuple[str, str]] = []
    legacy_welltrack_rows: list[tuple[str, str]] = []
    legacy_upload_selected = False
    for kind in _REFERENCE_FUND_OPTIONS:
        legacy_mode = str(
            st.session_state.get(
                f"ptc_reference_source_mode::{kind}",
                st.session_state.get(
                    reference_state.reference_source_mode_key(kind),
                    "",
                ),
            )
        ).strip()
        if legacy_mode == "Путь к WELLTRACK":
            legacy_welltrack_rows.extend(
                (path, kind)
                for path in reference_state.reference_welltrack_paths(kind)
                if str(path).strip()
            )
        elif legacy_mode == "Загрузить WELLTRACK":
            legacy_upload_selected = True
        else:
            legacy_dev_rows.extend(
                (path, kind)
                for path in reference_state.reference_dev_folder_paths(kind)
                if str(path).strip()
            )

    if legacy_dev_rows and not _dev_source_rows_present():
        _set_reference_dev_rows(legacy_dev_rows)
    if legacy_welltrack_rows and not _welltrack_source_rows_present():
        _set_reference_welltrack_rows(legacy_welltrack_rows)

    if legacy_dev_rows and legacy_welltrack_rows:
        st.session_state[_REFERENCE_PENDING_LEGACY_DEV_ROWS_KEY] = tuple(
            (str(path).strip(), _normalize_reference_fund_kind(kind))
            for path, kind in legacy_dev_rows
            if str(path).strip()
        )
        st.session_state[_REFERENCE_PENDING_LEGACY_WELLTRACK_ROWS_KEY] = tuple(
            (str(path).strip(), _normalize_reference_fund_kind(kind))
            for path, kind in legacy_welltrack_rows
            if str(path).strip()
        )
    else:
        _clear_pending_legacy_reference_rows()

    if _REFERENCE_IMPORT_MODE_KEY not in st.session_state:
        if legacy_upload_selected:
            st.session_state[_REFERENCE_IMPORT_MODE_KEY] = "Загрузить WELLTRACK"
        elif legacy_welltrack_rows and not legacy_dev_rows:
            st.session_state[_REFERENCE_IMPORT_MODE_KEY] = "Путь к WELLTRACK"
        elif legacy_dev_rows:
            st.session_state[_REFERENCE_IMPORT_MODE_KEY] = "Загрузить .dev"
        else:
            st.session_state[_REFERENCE_IMPORT_MODE_KEY] = _REFERENCE_SOURCE_OPTIONS[0]
    st.session_state[_REFERENCE_IMPORT_MIGRATED_KEY] = True


def _ensure_reference_import_state() -> None:
    st.session_state.setdefault(_REFERENCE_UPLOAD_NONCE_KEY, 0)
    st.session_state.setdefault(_REFERENCE_DEV_SOURCE_COUNT_KEY, 1)
    st.session_state.setdefault(_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY, 1)
    st.session_state.setdefault(_reference_dev_source_path_key(0), "")
    st.session_state.setdefault(
        _reference_dev_source_kind_key(0),
        wt.REFERENCE_WELL_ACTUAL,
    )
    st.session_state.setdefault(_reference_welltrack_source_path_key(0), "")
    st.session_state.setdefault(
        _reference_welltrack_source_kind_key(0),
        wt.REFERENCE_WELL_ACTUAL,
    )
    _migrate_legacy_reference_import_state()
    st.session_state.setdefault(
        _REFERENCE_IMPORT_MODE_KEY,
        _REFERENCE_SOURCE_OPTIONS[0],
    )
    _normalize_reference_import_mode_state()
    _reference_import_mode()


def _render_reference_kind_header(container: object) -> None:
    container.markdown(
        (
            "<div style='text-align: left; font-size: 0.875rem; "
            "opacity: 0.7;'>Тип фонда</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_reference_source_header(*, source_label: str = "Источник") -> None:
    source_col, kind_col = st.columns(
        (0.6, 0.4),
        gap="small",
        vertical_alignment="center",
    )
    source_col.caption(source_label)
    _render_reference_kind_header(kind_col)


def _render_reference_fund_switch(*, key: str) -> str:
    normalized_kind = _normalize_reference_fund_kind(st.session_state.get(key))
    st.session_state[key] = normalized_kind
    return str(
        st.radio(
            "Тип фонда",
            options=_REFERENCE_FUND_OPTIONS,
            key=key,
            format_func=_reference_fund_label,
            horizontal=True,
            label_visibility="collapsed",
        )
    )


def _render_reference_dev_source_inputs() -> None:
    source_count = _normalize_count(_REFERENCE_DEV_SOURCE_COUNT_KEY)
    _render_reference_source_header(source_label="Папка с .dev файлами")
    for index in range(source_count):
        source_col, kind_col = st.columns(
            (0.6, 0.4),
            gap="small",
            vertical_alignment="center",
        )
        source_col.text_input(
            "Папка с .dev файлами"
            if index == 0
            else f"Папка с .dev файлами #{index + 1}",
            key=_reference_dev_source_path_key(index),
            placeholder="tests/test_data/dev_fact",
            help=_REFERENCE_DEV_SOURCE_HELP,
            label_visibility="collapsed",
        )
        with kind_col:
            _render_reference_fund_switch(
                key=_reference_dev_source_kind_key(index),
            )
    if st.button(
        "Добавить папку",
        key="ptc_reference_add_dev_source",
        icon=":material/create_new_folder:",
        use_container_width=True,
    ):
        st.session_state[_REFERENCE_DEV_SOURCE_COUNT_KEY] = source_count + 1
        _rerun_fragment()


def _render_reference_welltrack_source_inputs() -> None:
    source_count = _normalize_count(_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY)
    _render_reference_source_header()
    for index in range(source_count):
        source_col, kind_col = st.columns(
            (0.6, 0.4),
            gap="small",
            vertical_alignment="center",
        )
        source_col.text_input(
            f"Путь к WELLTRACK или папке #{index + 1}",
            key=_reference_welltrack_source_path_key(index),
            placeholder="tests/test_data/WELLTRACKS3.INC",
            label_visibility="collapsed",
        )
        with kind_col:
            _render_reference_fund_switch(
                key=_reference_welltrack_source_kind_key(index),
            )
    if st.button(
        "Добавить путь",
        key="ptc_reference_add_welltrack_source",
        icon=":material/note_add:",
        use_container_width=True,
    ):
        st.session_state[_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY] = source_count + 1
        _rerun_fragment()
    st.caption(
        "Можно указать несколько файлов WELLTRACK и/или папок. "
        "Из папок импортируются только файлы с расширением `.INC` без учёта регистра."
    )


def _uploaded_source_label(uploaded_file: object) -> str:
    file_name = str(getattr(uploaded_file, "name", "uploaded"))
    size = getattr(uploaded_file, "size", None)
    if size in (None, ""):
        return file_name
    return f"{file_name} ({int(size)} B)"


def _render_reference_upload_inputs() -> list[object]:
    uploaded_files = list(
        st.file_uploader(
            "WELLTRACK файлы",
            type=["inc", "txt", "data", "ecl"],
            accept_multiple_files=True,
            key=_reference_upload_widget_key(),
        )
        or []
    )
    if uploaded_files:
        _render_reference_source_header()
        for index, uploaded_file in enumerate(uploaded_files):
            source_col, kind_col = st.columns(
                (0.6, 0.4),
                gap="small",
                vertical_alignment="center",
            )
            source_col.caption(_uploaded_source_label(uploaded_file))
            with kind_col:
                _render_reference_fund_switch(
                    key=_reference_upload_kind_key(index),
                )
    else:
        st.caption("Добавьте один или несколько WELLTRACK файлов.")
    return uploaded_files


def _reference_dev_sources_by_kind() -> dict[str, list[str]]:
    sources_by_kind = {kind: [] for kind in _REFERENCE_FUND_OPTIONS}
    for source_path, kind in _reference_dev_rows():
        sources_by_kind[kind].append(source_path)
    return sources_by_kind


def _reference_welltrack_sources_by_kind() -> dict[str, list[str]]:
    sources_by_kind = {kind: [] for kind in _REFERENCE_FUND_OPTIONS}
    for source_path, kind in _reference_welltrack_rows():
        sources_by_kind[kind].append(source_path)
    return sources_by_kind


def _reference_uploaded_sources_by_kind(
    uploaded_files: list[object],
) -> dict[str, list[object]]:
    sources_by_kind: dict[str, list[object]] = {
        kind: [] for kind in _REFERENCE_FUND_OPTIONS
    }
    for index, uploaded_file in enumerate(uploaded_files):
        kind = _normalize_reference_fund_kind(
            st.session_state.get(_reference_upload_kind_key(index))
        )
        sources_by_kind[kind].append(uploaded_file)
    return sources_by_kind


def _reference_sources_by_kind_from_rows(
    rows: tuple[tuple[str, str], ...],
) -> dict[str, list[str]]:
    sources_by_kind = {kind: [] for kind in _REFERENCE_FUND_OPTIONS}
    for source_path, kind in rows:
        normalized_path = str(source_path).strip()
        if not normalized_path:
            continue
        sources_by_kind[_normalize_reference_fund_kind(kind)].append(
            normalized_path
        )
    return sources_by_kind


def _pending_mixed_legacy_reference_sources() -> (
    tuple[dict[str, list[str]], dict[str, list[str]]] | None
):
    pending_dev_rows = tuple(
        st.session_state.get(_REFERENCE_PENDING_LEGACY_DEV_ROWS_KEY) or ()
    )
    pending_welltrack_rows = tuple(
        st.session_state.get(_REFERENCE_PENDING_LEGACY_WELLTRACK_ROWS_KEY) or ()
    )
    if not pending_dev_rows or not pending_welltrack_rows:
        return None
    if _reference_dev_rows() != pending_dev_rows:
        return None
    if _reference_welltrack_rows() != pending_welltrack_rows:
        return None
    return (
        _reference_sources_by_kind_from_rows(pending_dev_rows),
        _reference_sources_by_kind_from_rows(pending_welltrack_rows),
    )


def _parse_reference_sources(
    *,
    mode: str,
    uploaded_files: list[object],
) -> dict[str, tuple[object, ...]]:
    parsed_by_kind: dict[str, tuple[object, ...]] = {
        kind: () for kind in _REFERENCE_FUND_OPTIONS
    }
    pending_mixed_sources = _pending_mixed_legacy_reference_sources()
    if pending_mixed_sources is not None:
        parsed_lists_by_kind: dict[str, list[object]] = {
            kind: [] for kind in _REFERENCE_FUND_OPTIONS
        }
        dev_sources_by_kind, welltrack_sources_by_kind = pending_mixed_sources
        for kind, source_paths in dev_sources_by_kind.items():
            if not source_paths:
                continue
            parsed_lists_by_kind[kind].extend(
                parse_reference_trajectory_dev_directories(
                    source_paths,
                    kind=kind,
                )
            )
        for kind, source_paths in welltrack_sources_by_kind.items():
            if not source_paths:
                continue
            payload = ptc_welltrack_io.read_welltrack_sources(
                source_paths,
                info=st.info,
                warning=st.warning,
                error=st.error,
            )
            parsed_lists_by_kind[kind].extend(
                parse_reference_trajectory_welltrack_text(
                    payload,
                    kind=kind,
                )
            )
        return {
            kind: tuple(parsed_lists_by_kind[kind])
            for kind in _REFERENCE_FUND_OPTIONS
        }

    if mode == "Загрузить .dev":
        sources_by_kind = _reference_dev_sources_by_kind()
        if not any(sources_by_kind.values()):
            raise wt.WelltrackParseError(
                "Укажите хотя бы одну папку с `.dev` файлами."
            )
        for kind, source_paths in sources_by_kind.items():
            if not source_paths:
                continue
            parsed_by_kind[kind] = tuple(
                parse_reference_trajectory_dev_directories(
                    source_paths,
                    kind=kind,
                )
            )
        return parsed_by_kind

    if mode == "Путь к WELLTRACK":
        sources_by_kind = _reference_welltrack_sources_by_kind()
        if not any(sources_by_kind.values()):
            raise wt.WelltrackParseError(
                "Укажите хотя бы один путь к WELLTRACK файлу или папке."
            )
        for kind, source_paths in sources_by_kind.items():
            if not source_paths:
                continue
            payload = ptc_welltrack_io.read_welltrack_sources(
                source_paths,
                info=st.info,
                warning=st.warning,
                error=st.error,
            )
            parsed_by_kind[kind] = tuple(
                parse_reference_trajectory_welltrack_text(
                    payload,
                    kind=kind,
                )
            )
        return parsed_by_kind

    if mode == "Загрузить WELLTRACK":
        if not uploaded_files:
            raise wt.WelltrackParseError(
                "Загрузите хотя бы один WELLTRACK файл."
            )
        for kind, files_for_kind in _reference_uploaded_sources_by_kind(
            uploaded_files
        ).items():
            if not files_for_kind:
                continue
            payload = "\n\n".join(
                ptc_welltrack_io.decode_welltrack_payload(
                    getattr(uploaded_file, "getvalue", lambda: b"")(),
                    source_label=(
                        f"WELLTRACK `{getattr(uploaded_file, 'name', 'uploaded')}`"
                    ),
                    info=st.info,
                    warning=st.warning,
                )
                for uploaded_file in files_for_kind
            )
            parsed_by_kind[kind] = tuple(
                parse_reference_trajectory_welltrack_text(
                    payload,
                    kind=kind,
                )
            )
        return parsed_by_kind

    raise wt.WelltrackParseError(
        f"Неподдерживаемый режим импорта фонда: {mode!r}."
    )


def _clear_combined_reference_import_state() -> None:
    _clear_reference_import_state(wt.REFERENCE_WELL_ACTUAL)
    _clear_reference_import_state(wt.REFERENCE_WELL_APPROVED)
    _clear_pending_legacy_reference_rows()
    source_count = _normalize_count(_REFERENCE_DEV_SOURCE_COUNT_KEY)
    for index in range(source_count):
        st.session_state[_reference_dev_source_path_key(index)] = ""
        st.session_state[_reference_dev_source_kind_key(index)] = (
            wt.REFERENCE_WELL_ACTUAL
        )
    st.session_state[_REFERENCE_DEV_SOURCE_COUNT_KEY] = 1
    path_count = _normalize_count(_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY)
    for index in range(path_count):
        st.session_state[_reference_welltrack_source_path_key(index)] = ""
        st.session_state[_reference_welltrack_source_kind_key(index)] = (
            wt.REFERENCE_WELL_ACTUAL
        )
    st.session_state[_REFERENCE_WELLTRACK_SOURCE_COUNT_KEY] = 1
    st.session_state[_REFERENCE_IMPORT_MODE_KEY] = _REFERENCE_SOURCE_OPTIONS[0]
    st.session_state[_REFERENCE_UPLOAD_NONCE_KEY] = (
        int(st.session_state.get(_REFERENCE_UPLOAD_NONCE_KEY, 0) or 0) + 1
    )
    for kind in _REFERENCE_FUND_OPTIONS:
        st.session_state[reference_state.reference_source_mode_key(kind)] = (
            _REFERENCE_SOURCE_OPTIONS[0]
        )
        st.session_state[f"ptc_reference_source_mode::{kind}"] = (
            _REFERENCE_SOURCE_OPTIONS[0]
        )


def _render_combined_reference_import_block() -> None:
    _ensure_reference_import_state()
    mode_label_col, mode_radio_col = st.columns(
        [0.9, 4.7], gap="small", vertical_alignment="center"
    )
    with mode_label_col:
        st.markdown("**Источник данных:**")
    with mode_radio_col:
        mode = st.radio(
            "Источник данных",
            options=_REFERENCE_SOURCE_OPTIONS,
            key=_REFERENCE_IMPORT_MODE_KEY,
            horizontal=True,
            label_visibility="collapsed",
        )
    _reference_import_mode()

    uploaded_files: list[object] = []
    if mode == "Загрузить .dev":
        _render_reference_dev_source_inputs()
    elif mode == "Путь к WELLTRACK":
        _render_reference_welltrack_source_inputs()
    else:
        uploaded_files = _render_reference_upload_inputs()

    action_col, clear_col = st.columns(2, gap="small")
    import_clicked = action_col.button(
        "Загрузить фонд",
        key="ptc_reference_import_combined",
        type="primary",
        icon=":material/upload_file:",
        use_container_width=True,
    )
    clear_col.button(
        "Очистить фонд",
        key="ptc_reference_clear_combined",
        icon=":material/delete:",
        use_container_width=True,
        on_click=_clear_combined_reference_import_state,
    )

    if import_clicked:
        with st.status("Импорт фонда...", expanded=True) as status:
            try:
                parsed_by_kind = _parse_reference_sources(
                    mode=mode,
                    uploaded_files=uploaded_files,
                )
                for kind in _REFERENCE_FUND_OPTIONS:
                    reference_state.set_reference_wells_for_kind(
                        kind=kind,
                        wells=parsed_by_kind[kind],
                    )
                    _after_reference_data_change(kind)
                actual_count = len(parsed_by_kind[wt.REFERENCE_WELL_ACTUAL])
                approved_count = len(parsed_by_kind[wt.REFERENCE_WELL_APPROVED])
                status.write(f"Фактических скважин: {actual_count}.")
                status.write(
                    "Утверждённых проектных скважин: "
                    f"{approved_count}."
                )
                status.update(
                    label="Фонд импортирован",
                    state="complete",
                    expanded=False,
                )
                _clear_pending_legacy_reference_rows()
                st.rerun()
            except wt.WelltrackParseError as exc:
                status.write(str(exc))
                status.update(
                    label="Ошибка импорта фонда",
                    state="error",
                    expanded=True,
                )

    actual_loaded = len(
        reference_state.reference_kind_wells(wt.REFERENCE_WELL_ACTUAL)
    )
    approved_loaded = len(
        reference_state.reference_kind_wells(wt.REFERENCE_WELL_APPROVED)
    )
    st.caption(
        "Сейчас загружено: "
        f"{actual_loaded} фактических, {approved_loaded} утверждённых проектных."
    )


def render_reference_section() -> None:
    st.markdown("## 3. Загрузка фактического и проектного фонда")
    _render_combined_reference_import_block()

    reference_wells = tuple(reference_state.reference_wells_from_state())
    if reference_wells:
        with st.expander(
            "Список загруженных фактических/ проектных скважин",
            expanded=False,
        ):
            st.dataframe(
                wt.arrow_safe_text_dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Скважина": wt.reference_well_display_label(
                                    item
                                ),
                                "Точек": int(len(item.stations)),
                                "MD max, м": float(
                                    item.stations["MD_m"].iloc[-1]
                                ),
                            }
                            for item in reference_wells
                        ]
                    )
                ),
                width="stretch",
                hide_index=True,
            )

    actual_wells = tuple(
        reference_state.reference_kind_wells(wt.REFERENCE_WELL_ACTUAL)
    )
    if actual_wells:
        _render_zbs_actual_match_info(actual_wells)
        show_actual_analysis = bool(
            st.toggle(
                "Показать анализ фактического фонда",
                key=_reference_analysis_toggle_key(wt.REFERENCE_WELL_ACTUAL),
                help=(
                    "Анализ строится по кнопке — страница откроется быстрее."
                ),
            )
        )
        if show_actual_analysis:
            analyses = wt._actual_fund_analyses(actual_wells)
            wt._render_actual_fund_analysis_panel(analyses=analyses)

    approved_wells = tuple(
        reference_state.reference_kind_wells(wt.REFERENCE_WELL_APPROVED)
    )
    if approved_wells:
        show_approved_analysis = bool(
            st.toggle(
                "Показать просмотр утверждённого проектного фонда",
                key=_reference_analysis_toggle_key(wt.REFERENCE_WELL_APPROVED),
                help=(
                    "Просмотр утверждённого фонда строится по запросу, "
                    "чтобы не утяжелять обычный расчётный flow."
                ),
            )
        )
        if show_approved_analysis:
            try:
                approved_analyses = wt._approved_fund_analyses(approved_wells)
            except Exception as exc:
                st.error(
                    "Не удалось построить просмотр утверждённого проектного фонда."
                )
                st.caption(f"{type(exc).__name__}: {exc}")
            else:
                with st.expander(
                    "Просмотр загруженных утверждённых проектных скважин",
                    expanded=False,
                ):
                    wt._render_reference_well_detail(
                        approved_analyses,
                        select_label="Просмотр утвержденной проектной скважины",
                        selected_key="wt_approved_fund_selected_well",
                    )
        else:
            st.caption(
                "Просмотр утверждённого проектного фонда скрыт до явного открытия."
            )


def _render_zbs_actual_match_info(actual_wells: tuple[object, ...]) -> None:
    records = tuple(st.session_state.get("wt_records") or ())
    zbs_records = [
        record
        for record in records
        if is_zbs_record(record)
    ]
    if not zbs_records:
        return
    active_t1_t3_issue_names = {
        str(getattr(item, "well_name", "")).strip()
        for item in wt._current_t1_t3_order_issues(list(records))
        if str(getattr(item, "well_name", "")).strip()
    }
    actual_name_by_key = {
        well_name_key(getattr(well, "name", "")): str(getattr(well, "name", ""))
        for well in actual_wells
    }
    matched: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for record in zbs_records:
        zbs_name = str(getattr(record, "name", ""))
        parent_name = parent_name_for_zbs(zbs_name)
        actual_name = actual_name_by_key.get(well_name_key(parent_name))
        if actual_name is None:
            missing.append((zbs_name, parent_name))
        else:
            matched.append((zbs_name, actual_name))
    if matched:
        st.info(
            "Подцепили фактическую скважину для бокового ствола: "
            + ", ".join(
                f"{zbs_name} -> {actual_name}"
                for zbs_name, actual_name in matched
            )
            + "."
        )
        matched_issue_names = [
            zbs_name
            for zbs_name, _actual_name in matched
            if zbs_name in active_t1_t3_issue_names
        ]
        if matched_issue_names:
            st.warning(
                "После подцепления фактической скважины проверьте порядок целей `t1/t3` "
                "для боковых стволов: "
                + ", ".join(matched_issue_names)
                + ". "
                + f"[Перейти к блоку проверки t1/t3](#{wt.WT_T1T3_ORDER_PANEL_ANCHOR_ID})."
            )
    if missing:
        st.warning(
            "Для боковых стволов не найдены фактические скважины: "
            + ", ".join(
                f'{zbs_name} ждёт "{parent_name}"'
                for zbs_name, parent_name in missing
            )
            + "."
        )
