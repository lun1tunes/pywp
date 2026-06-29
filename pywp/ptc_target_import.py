from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from pywp.eclipse_welltrack import (
    WelltrackRecord,
    WelltrackParseError,
    decode_welltrack_bytes,
    parse_welltrack_points_table,
    parse_welltrack_text,
)
from pywp.path_utils import normalize_user_path_text
from pywp.pilot_wells import visible_well_names
from pywp.reference_trajectories import ImportedTrajectoryWell
from pywp.ptc_target_import_dev import (
    DevTargetImportSummary,
    dev_target_import_summary_dataframe,
    dev_trajectory_text_name,
    parse_dev_target_directory,
    parse_dev_target_file,
    parse_dev_target_payloads,
)

__all__ = [
    "AUTO_LAYOUT_APPLIED_MESSAGE",
    "DEFAULT_WELLTRACK_PATH",
    "DEFAULT_DEV_TRAJECTORY_PATH",
    "DevTargetImportSummary",
    "TargetImportEmptySourceError",
    "TargetImportOperation",
    "TargetImportParseResult",
    "TargetImportStoreResult",
    "WT_SOURCE_FORMAT_DEV_TRAJECTORY",
    "WT_SOURCE_FORMAT_OPTIONS",
    "WT_SOURCE_FORMAT_TARGET_TABLE",
    "WT_SOURCE_FORMAT_WELLTRACK",
    "WT_SOURCE_KIND_DEV_TRAJECTORY",
    "WT_SOURCE_MODE_FILE_PATH",
    "WT_SOURCE_MODE_INLINE_TEXT",
    "WT_SOURCE_MODE_TARGET_TABLE",
    "WT_SOURCE_MODE_UPLOAD",
    "WT_SOURCE_WELLTRACK_MODES",
    "WelltrackSourcePayload",
    "build_target_import_operation",
    "clear_target_import_flow_state",
    "coerce_source_table_df_columns",
    "dev_source_preview_well_names",
    "dev_target_import_summary_dataframe",
    "empty_source_table_df",
    "expand_single_column_source_table_df",
    "init_target_source_state_defaults",
    "normalize_source_table_df_for_ui",
    "reset_failed_import_state",
    "store_imported_records",
]

DEFAULT_WELLTRACK_PATH = Path("tests/test_data/WELLTRACKS4.INC")
DEFAULT_DEV_TRAJECTORY_PATH = Path("tests/test_data/dev_target_import")
WT_SOURCE_FORMAT_WELLTRACK = "WELLTRACK"
WT_SOURCE_FORMAT_DEV_TRAJECTORY = ".dev траектория"
WT_SOURCE_FORMAT_TARGET_TABLE = "Таблица с точками целей"
WT_SOURCE_FORMAT_OPTIONS: tuple[str, ...] = (
    WT_SOURCE_FORMAT_WELLTRACK,
    WT_SOURCE_FORMAT_DEV_TRAJECTORY,
    WT_SOURCE_FORMAT_TARGET_TABLE,
)
WT_SOURCE_MODE_FILE_PATH = "Файл по пути"
WT_SOURCE_MODE_UPLOAD = "Загрузить файл"
WT_SOURCE_MODE_INLINE_TEXT = "Вставить текст"
WT_SOURCE_MODE_TARGET_TABLE = "Вставить таблицу"
WT_SOURCE_WELLTRACK_MODES: tuple[str, ...] = (
    WT_SOURCE_MODE_FILE_PATH,
    WT_SOURCE_MODE_UPLOAD,
    WT_SOURCE_MODE_INLINE_TEXT,
)
AUTO_LAYOUT_APPLIED_MESSAGE = (
    "Обнаружены кусты с общим исходным S: устья автоматически "
    "разведены по параметрам блока 'Кусты и расчет устьев'. "
    "При необходимости можно нажать 'Вернуть исходные устья'."
)
IMPORTED_DEV_TARGET_WELLS_STATE_KEY = "wt_imported_dev_target_wells"
_TARGET_IMPORT_KIND_TABLE = "target_table"
_TARGET_IMPORT_KIND_WELLTRACK = "welltrack"
WT_SOURCE_KIND_DEV_TRAJECTORY = "dev_trajectory"
_SOURCE_TABLE_COLUMNS = ("Wellname", "Point", "X", "Y", "Z")
_SOURCE_TABLE_ALIAS_MAP = {
    "wellname": "Wellname",
    "well_name": "Wellname",
    "well name": "Wellname",
    "well": "Wellname",
    "name": "Wellname",
    "point": "Point",
    "pointname": "Point",
    "point_name": "Point",
    "point name": "Point",
    "точка": "Point",
    "x": "X",
    "east": "X",
    "easting": "X",
    "x_m": "X",
    "y": "Y",
    "north": "Y",
    "northing": "Y",
    "y_m": "Y",
    "z": "Z",
    "tvd": "Z",
    "z_tvd": "Z",
    "z_m": "Z",
}


@dataclass(frozen=True)
class WelltrackSourcePayload:
    """Normalized payload returned by the target-source UI block."""

    mode: str
    source_format: str = WT_SOURCE_FORMAT_WELLTRACK
    source_text: str = ""
    source_path: str = ""
    source_files: tuple[tuple[str, bytes], ...] = ()
    table_rows: pd.DataFrame | None = None
    dev_fixed_t1_inc_by_well: tuple[tuple[str, float], ...] = ()


class TargetImportEmptySourceError(ValueError):
    """Raised when the user requested import from an empty source."""


@dataclass(frozen=True)
class TargetImportParseResult:
    records: tuple[WelltrackRecord, ...]
    dev_summaries: tuple[DevTargetImportSummary, ...] = ()
    imported_dev_wells: tuple[ImportedTrajectoryWell, ...] = ()


@dataclass(frozen=True)
class TargetImportOperation:
    """Executable import operation with UI-facing status labels."""

    source_kind: str
    status_label: str
    progress_message: str
    count_message_template: str
    success_label_template: str
    error_label: str
    source_text: str = ""
    source_path: str = ""
    source_files: tuple[tuple[str, bytes], ...] = ()
    table_rows: pd.DataFrame | None = None
    dev_fixed_t1_inc_by_well: tuple[tuple[str, float], ...] = ()
    parse_welltrack_text_func: Callable[[str], list[WelltrackRecord]] = (
        parse_welltrack_text
    )

    def parse(self) -> TargetImportParseResult:
        if self.source_kind == _TARGET_IMPORT_KIND_TABLE:
            return TargetImportParseResult(
                records=tuple(
                    parse_welltrack_points_table(
                        pd.DataFrame(self.table_rows).to_dict(orient="records")
                    )
                )
            )
        if self.source_kind == WT_SOURCE_KIND_DEV_TRAJECTORY:
            return _parse_dev_target_payload(
                source_text=self.source_text,
                source_path=self.source_path,
                source_files=self.source_files,
                fixed_t1_inc_by_well=dict(self.dev_fixed_t1_inc_by_well),
            )
        return TargetImportParseResult(
            records=tuple(self.parse_welltrack_text_func(str(self.source_text)))
        )

    def parse_records(self) -> list[WelltrackRecord]:
        return list(self.parse().records)

    def count_message(self, record_count: int) -> str:
        return self.count_message_template.format(record_count=int(record_count))

    def success_label(self, elapsed_s: float) -> str:
        return self.success_label_template.format(elapsed_s=float(elapsed_s))


@dataclass(frozen=True)
class TargetImportStoreResult:
    """State mutation summary after imported target records are stored."""

    well_names: tuple[str, ...]
    auto_layout_applied: bool


def empty_source_table_df() -> pd.DataFrame:
    """Return the blank editable target-points table used by the PTC UI."""

    return pd.DataFrame(
        [
            {"Wellname": "", "Point": "", "X": np.nan, "Y": np.nan, "Z": np.nan},
            {"Wellname": "", "Point": "", "X": np.nan, "Y": np.nan, "Z": np.nan},
            {"Wellname": "", "Point": "", "X": np.nan, "Y": np.nan, "Z": np.nan},
        ]
    )


def normalize_source_table_df_for_ui(table_df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize pasted target-point rows to Wellname / Point / X / Y / Z."""

    if table_df is None:
        return empty_source_table_df()
    normalized_df = coerce_source_table_df_columns(pd.DataFrame(table_df).copy())
    normalized_df = expand_single_column_source_table_df(normalized_df)
    if "Point" in normalized_df.columns:
        normalized_df["Point"] = normalized_df["Point"].map(
            _normalize_source_table_point_value
        )
    for column in _SOURCE_TABLE_COLUMNS:
        if column not in normalized_df.columns:
            normalized_df[column] = "" if column in {"Wellname", "Point"} else np.nan
    return normalized_df.loc[:, list(_SOURCE_TABLE_COLUMNS)].reset_index(drop=True)


def _normalize_source_table_point_value(value: object) -> object:
    text = str(value).strip()
    if text.lower() in {"wellhead", "s"}:
        return "S"
    multi_match = re.match(r"^([1-9]\d*)_t([13])$", text, flags=re.IGNORECASE)
    if multi_match is not None:
        return f"{int(multi_match.group(1))}_t{int(multi_match.group(2))}"
    pilot_match = re.match(r"^pl([1-9]\d*)$", text, flags=re.IGNORECASE)
    if pilot_match is not None:
        return f"PL{int(pilot_match.group(1))}"
    return value


def init_target_source_state_defaults(
    session_state: MutableMapping[str, object],
) -> None:
    """Initialize target-source session state without overwriting user input."""

    if "wt_source_format" not in session_state:
        session_state["wt_source_format"] = (
            WT_SOURCE_FORMAT_TARGET_TABLE
            if str(session_state.get("wt_source_mode", "")).strip()
            == WT_SOURCE_MODE_TARGET_TABLE
            else WT_SOURCE_FORMAT_WELLTRACK
        )
    if str(session_state.get("wt_source_format", "")).strip() not in set(
        WT_SOURCE_FORMAT_OPTIONS
    ):
        session_state["wt_source_format"] = WT_SOURCE_FORMAT_WELLTRACK
    session_state.setdefault("wt_source_mode", WT_SOURCE_MODE_FILE_PATH)
    session_state.setdefault("wt_source_path", str(DEFAULT_WELLTRACK_PATH))
    session_state.setdefault("wt_source_inline", "")
    if session_state.get("wt_source_upload_file") is None:
        session_state.pop("wt_source_upload_file", None)
    session_state.setdefault("wt_source_dev_inline", "")
    raw_dev_upload_files = session_state.get("wt_source_dev_upload_files")
    if raw_dev_upload_files in (None, []):
        session_state.pop("wt_source_dev_upload_files", None)
    session_state.setdefault("wt_source_dev_fixed_t1_enabled", False)
    session_state.setdefault("wt_source_dev_fixed_t1_well_names", [])
    session_state.setdefault("wt_source_dev_fixed_t1_inc_deg", 86.0)
    session_state.setdefault("wt_source_table_df", empty_source_table_df())
    session_state.setdefault("wt_source_table_editor_nonce", 0)


def build_target_import_operation(
    source_payload: WelltrackSourcePayload,
    *,
    parse_welltrack_text_func: Callable[[str], list[WelltrackRecord]] = (
        parse_welltrack_text
    ),
) -> TargetImportOperation:
    """Build a parse operation for the selected target import source."""

    if source_payload.mode == WT_SOURCE_MODE_TARGET_TABLE:
        table_rows = source_payload.table_rows
        if table_rows is None:
            raise TargetImportEmptySourceError(
                "Таблица пуста. Вставьте строки в формате Wellname / Point / X / Y / Z."
            )
        return TargetImportOperation(
            source_kind=_TARGET_IMPORT_KIND_TABLE,
            status_label="Чтение и преобразование таблицы точек...",
            progress_message="Разбор таблицы точек...",
            count_message_template="Собрано скважин из таблицы: {record_count}.",
            success_label_template="Импорт таблицы завершен за {elapsed_s:.2f} с",
            error_label="Ошибка разбора табличного WELLTRACK",
            table_rows=pd.DataFrame(table_rows),
            parse_welltrack_text_func=parse_welltrack_text_func,
        )

    if source_payload.source_format == WT_SOURCE_FORMAT_DEV_TRAJECTORY:
        if source_payload.mode == WT_SOURCE_MODE_FILE_PATH and not normalize_user_path_text(
            source_payload.source_path
        ):
            raise TargetImportEmptySourceError(
                "Источник пустой. Укажите путь к .dev файлу или папке."
            )
        if (
            source_payload.mode == WT_SOURCE_MODE_UPLOAD
            and not tuple(source_payload.source_files)
        ):
            raise TargetImportEmptySourceError(
                "Источник пустой. Загрузите хотя бы один .dev файл."
            )
        if (
            source_payload.mode == WT_SOURCE_MODE_INLINE_TEXT
            and not str(source_payload.source_text).strip()
        ):
            raise TargetImportEmptySourceError(
                "Источник пустой. Вставьте текст .dev."
            )
        return TargetImportOperation(
            source_kind=WT_SOURCE_KIND_DEV_TRAJECTORY,
            status_label="Чтение и разбор .dev траекторий...",
            progress_message="Поиск S / KOP / t1 / t3 и параметров траектории.",
            count_message_template="Прочитано .dev траекторий: {record_count}.",
            success_label_template="Импорт .dev завершен за {elapsed_s:.2f} с",
            error_label="Ошибка разбора .dev",
            source_text=str(source_payload.source_text),
            source_path=str(source_payload.source_path),
            source_files=tuple(source_payload.source_files),
            dev_fixed_t1_inc_by_well=tuple(source_payload.dev_fixed_t1_inc_by_well),
            parse_welltrack_text_func=parse_welltrack_text_func,
        )

    source_text = str(source_payload.source_text)
    if not source_text.strip():
        raise TargetImportEmptySourceError(
            "Источник пустой. Загрузите файл или вставьте текст WELLTRACK."
        )
    return TargetImportOperation(
        source_kind=_TARGET_IMPORT_KIND_WELLTRACK,
        status_label="Чтение и парсинг WELLTRACK...",
        progress_message="Проверка структуры WELLTRACK-блоков.",
        count_message_template="Найдено блоков WELLTRACK: {record_count}.",
        success_label_template="Импорт завершен за {elapsed_s:.2f} с",
        error_label="Ошибка парсинга WELLTRACK",
        source_text=source_text,
        parse_welltrack_text_func=parse_welltrack_text_func,
    )


def store_imported_records(
    session_state: MutableMapping[str, object],
    *,
    records: Sequence[WelltrackRecord],
    dev_summaries: Sequence[DevTargetImportSummary] = (),
    imported_dev_wells: Sequence[ImportedTrajectoryWell] = (),
    source_kind: str = _TARGET_IMPORT_KIND_WELLTRACK,
    loaded_at_text: str,
    clear_t1_t3_order_state: Callable[[], None],
    clear_pad_state: Callable[[], None],
    clear_results: Callable[[], None],
    auto_apply_pad_layout: Callable[[list[WelltrackRecord]], bool],
) -> TargetImportStoreResult:
    """Store parsed target records and run import-time pad auto-layout."""

    normalized_records = list(records)
    well_names = tuple(str(record.name) for record in normalized_records)
    session_state["wt_records"] = list(normalized_records)
    session_state["wt_records_original"] = list(normalized_records)
    session_state["wt_loaded_at"] = str(loaded_at_text)
    session_state["wt_target_import_source_kind"] = str(source_kind)
    session_state["wt_imported_dev_params"] = tuple(dev_summaries)
    session_state[IMPORTED_DEV_TARGET_WELLS_STATE_KEY] = tuple(imported_dev_wells)
    session_state["wt_well_calc_overrides_enabled"] = False
    session_state["wt_well_calc_overrides"] = {}
    session_state["wt_well_calc_profile_assignments"] = {}
    session_state["wt_well_calc_active_profile_id"] = ""
    session_state["wt_well_calc_active_profile_id_pending"] = None
    session_state.pop("wt_well_calc_profile_import_upload", None)
    session_state["wt_well_calc_override_selected_names"] = []
    session_state["wt_well_calc_override_selected_signature"] = ()
    session_state["wt_well_calc_override_feedback"] = ""
    session_state["wt_well_calc_override_name_input_active_profile_id"] = ""
    session_state["wt_raw_records_edit_mode"] = False
    session_state["wt_raw_records_editor_nonce"] = 0
    for key in list(session_state.keys()):
        if str(key).startswith("wt_well_calc_override_profile_name__"):
            session_state.pop(key, None)
    session_state.pop("wt_preprocess_horizontal_length_m", None)
    session_state.pop("wt_records_overview_expand_once", None)
    session_state.pop("wt_edit_targets_applied_note", None)
    clear_t1_t3_order_state()
    clear_pad_state()
    session_state["wt_last_error"] = ""
    clear_results()
    auto_layout_applied = bool(auto_apply_pad_layout(list(normalized_records)))
    session_state["wt_selected_names"] = visible_well_names(normalized_records)
    return TargetImportStoreResult(
        well_names=well_names,
        auto_layout_applied=auto_layout_applied,
    )


def reset_failed_import_state(
    session_state: MutableMapping[str, object],
    *,
    error_message: str,
    clear_t1_t3_order_state: Callable[[], None],
    clear_pad_state: Callable[[], None],
) -> None:
    """Reset target records after a failed parse without touching old results."""

    session_state["wt_records"] = None
    session_state["wt_records_original"] = None
    session_state["wt_target_import_source_kind"] = ""
    session_state["wt_imported_dev_params"] = ()
    session_state[IMPORTED_DEV_TARGET_WELLS_STATE_KEY] = ()
    session_state["wt_well_calc_overrides_enabled"] = False
    session_state["wt_well_calc_overrides"] = {}
    session_state["wt_well_calc_profile_assignments"] = {}
    session_state["wt_well_calc_active_profile_id"] = ""
    session_state["wt_well_calc_active_profile_id_pending"] = None
    session_state.pop("wt_well_calc_profile_import_upload", None)
    session_state["wt_well_calc_override_selected_names"] = []
    session_state["wt_well_calc_override_selected_signature"] = ()
    session_state["wt_well_calc_override_feedback"] = ""
    session_state["wt_well_calc_override_name_input_active_profile_id"] = ""
    session_state["wt_raw_records_edit_mode"] = False
    session_state["wt_raw_records_editor_nonce"] = 0
    for key in list(session_state.keys()):
        if str(key).startswith("wt_well_calc_override_profile_name__"):
            session_state.pop(key, None)
    session_state.pop("wt_preprocess_horizontal_length_m", None)
    session_state.pop("wt_records_overview_expand_once", None)
    session_state.pop("wt_edit_targets_applied_note", None)
    clear_t1_t3_order_state()
    clear_pad_state()
    session_state["wt_last_error"] = str(error_message)


def clear_target_import_flow_state(
    session_state: MutableMapping[str, object],
    *,
    reference_well_state_keys: Sequence[str],
    clear_t1_t3_order_state: Callable[[], None],
    clear_pad_state: Callable[[], None],
    clear_results: Callable[[], None],
) -> None:
    """Clear imported targets, reference wells, selection, and result state."""

    session_state["wt_records"] = None
    session_state["wt_records_original"] = None
    session_state["wt_target_import_source_kind"] = ""
    session_state["wt_imported_dev_params"] = ()
    session_state[IMPORTED_DEV_TARGET_WELLS_STATE_KEY] = ()
    session_state["wt_well_calc_overrides_enabled"] = False
    session_state["wt_well_calc_overrides"] = {}
    session_state["wt_well_calc_profile_assignments"] = {}
    session_state["wt_well_calc_active_profile_id"] = ""
    session_state["wt_well_calc_active_profile_id_pending"] = None
    session_state.pop("wt_well_calc_profile_import_upload", None)
    session_state["wt_well_calc_override_selected_names"] = []
    session_state["wt_well_calc_override_selected_signature"] = ()
    session_state["wt_well_calc_override_feedback"] = ""
    session_state["wt_well_calc_override_name_input_active_profile_id"] = ""
    session_state["wt_raw_records_edit_mode"] = False
    session_state["wt_raw_records_editor_nonce"] = 0
    for key in list(session_state.keys()):
        if str(key).startswith("wt_well_calc_override_profile_name__"):
            session_state.pop(key, None)
    session_state["wt_reference_wells"] = ()
    for key in reference_well_state_keys:
        session_state[str(key)] = ()
    session_state["wt_selected_names"] = []
    session_state["wt_loaded_at"] = ""
    session_state.pop("wt_preprocess_horizontal_length_m", None)
    session_state.pop("wt_records_overview_expand_once", None)
    session_state.pop("wt_edit_targets_applied_note", None)
    clear_t1_t3_order_state()
    clear_pad_state()
    clear_results()


def coerce_source_table_df_columns(table_df: pd.DataFrame) -> pd.DataFrame:
    """Keep and rename known target-table columns while preserving raw rows."""

    renamed: dict[object, str] = {}
    for raw_column in list(table_df.columns):
        column_text = str(raw_column).strip()
        normalized = re.sub(r"[\s\-/(),.:]+", "_", column_text.lower()).strip(
            "_"
        )
        if column_text.lower().startswith("unnamed"):
            continue
        if normalized in _SOURCE_TABLE_ALIAS_MAP:
            renamed[raw_column] = _SOURCE_TABLE_ALIAS_MAP[normalized]
            continue
        if column_text in _SOURCE_TABLE_COLUMNS:
            renamed[raw_column] = column_text
    kept_columns = [column for column in table_df.columns if column in renamed]
    if not kept_columns and len(table_df.columns) == 1:
        return table_df
    if kept_columns:
        table_df = table_df.loc[:, kept_columns].rename(columns=renamed)
    return table_df


def expand_single_column_source_table_df(table_df: pd.DataFrame) -> pd.DataFrame:
    """Expand Excel-like pasted rows that arrive as one tab/semicolon column."""

    if len(table_df.columns) != 1:
        return table_df
    series = table_df.iloc[:, 0]
    non_blank_values = [
        str(value).strip()
        for value in series
        if not pd.isna(value) and str(value).strip()
    ]
    if not non_blank_values:
        return table_df
    if not any("\t" in value or ";" in value for value in non_blank_values):
        return table_df
    rows: list[dict[str, object]] = []
    for raw_value in non_blank_values:
        tokens = [
            token.strip()
            for token in re.split(r"[\t;]+", raw_value)
            if token.strip()
        ]
        if len(tokens) not in {5, 6}:
            return table_df
        rows.append(
            {
                "Wellname": tokens[0],
                "Point": tokens[1],
                "X": tokens[2],
                "Y": tokens[3],
                "Z": tokens[4],
            }
        )
    return pd.DataFrame(rows, columns=list(_SOURCE_TABLE_COLUMNS))


def _parse_dev_target_payload(
    *,
    source_text: str,
    source_path: str,
    source_files: Sequence[tuple[str, bytes]],
    fixed_t1_inc_by_well: dict[str, float] | None = None,
) -> TargetImportParseResult:
    parsed_wells = []
    normalized_path = normalize_user_path_text(source_path)
    if normalized_path:
        path_obj = Path(normalized_path).expanduser()
        if not path_obj.exists():
            raise WelltrackParseError(f"Путь .dev не найден: `{path_obj}`.")
        if path_obj.is_dir():
            parsed_wells.extend(
                parse_dev_target_directory(
                    path_obj,
                    fixed_t1_inc_by_well=fixed_t1_inc_by_well,
                )
            )
        else:
            parsed_wells.append(
                parse_dev_target_file(
                    path_obj,
                    fixed_t1_inc_by_well=fixed_t1_inc_by_well,
                )
            )
    elif source_files:
        parsed_wells.extend(
            parse_dev_target_payloads(
                tuple(source_files),
                fixed_t1_inc_by_well=fixed_t1_inc_by_well,
            )
        )
    else:
        fallback_name = dev_trajectory_text_name(source_text)
        parsed_wells.extend(
            parse_dev_target_payloads(
                ((f"{fallback_name}.dev", str(source_text).encode("utf-8")),),
                fixed_t1_inc_by_well=fixed_t1_inc_by_well,
            )
        )
    return TargetImportParseResult(
        records=tuple(item.record for item in parsed_wells),
        dev_summaries=tuple(item.summary for item in parsed_wells),
        imported_dev_wells=tuple(item.imported_well for item in parsed_wells),
    )

def dev_source_preview_well_names(
    *,
    source_mode: str,
    source_path: str = "",
    source_files: Sequence[tuple[str, bytes]] = (),
    source_text: str = "",
) -> tuple[str, ...]:
    mode = str(source_mode).strip()
    names: list[str] = []
    seen: set[str] = set()

    def add_name(raw_name: object) -> None:
        normalized = str(raw_name).strip()
        if not normalized:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        names.append(normalized)

    try:
        if mode == WT_SOURCE_MODE_FILE_PATH:
            normalized_path = normalize_user_path_text(source_path)
            if not normalized_path:
                return ()
            path_obj = Path(normalized_path).expanduser()
            if not path_obj.exists():
                return ()
            if path_obj.is_dir():
                for child in sorted(
                    path_obj.iterdir(),
                    key=lambda item: item.name.casefold(),
                ):
                    if child.is_file() and child.suffix.lower() == ".dev":
                        add_name(child.stem)
                return tuple(names)
            if path_obj.is_file():
                add_name(path_obj.stem)
                return tuple(names)
            return ()

        if mode == WT_SOURCE_MODE_UPLOAD:
            for index, (file_name, raw_bytes) in enumerate(source_files, start=1):
                text, _encoding = decode_welltrack_bytes(bytes(raw_bytes))
                fallback_name = Path(str(file_name or f"dev_import_{index}")).stem
                add_name(dev_trajectory_text_name(text, fallback_name=fallback_name))
            return tuple(names)

        if mode == WT_SOURCE_MODE_INLINE_TEXT and str(source_text).strip():
            add_name(
                dev_trajectory_text_name(
                    str(source_text),
                    fallback_name="dev_import_1",
                )
            )
            return tuple(names)
    except OSError:
        return ()
    except WelltrackParseError:
        return ()
    return ()
