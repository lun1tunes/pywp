from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from pywp.eclipse_welltrack import (
    WelltrackParseError,
    WelltrackPoint,
    WelltrackRecord,
    decode_welltrack_bytes,
)
from pywp.reference_trajectories import (
    ImportedTrajectoryWell,
    REFERENCE_WELL_APPROVED,
    parse_reference_trajectory_dev_directories,
    parse_reference_trajectory_dev_file,
    parse_reference_trajectory_dev_text,
)
from pywp.ui_utils import dls_to_pi

__all__ = [
    "DevTargetImportParsedWell",
    "DevTargetImportSummary",
    "dev_target_import_summary_dataframe",
    "dev_trajectory_text_name",
    "parse_dev_target_directory",
    "parse_dev_target_file",
    "parse_dev_target_payloads",
    "target_record_and_summary_from_dev_well",
]

_DEV_DLS_EPSILON_DEG_PER_30M = 0.05
_DEV_INCL_EPSILON_DEG = 0.05
# BUILD1/BUILD2 can contain short or medium constant-INC pauses inside the same
# section; the true BUILD-HOLD-BUILD split is anchored to the hold plateau.
_DEV_HOLD_MIN_ROWS = 4
_DEV_WELL_NAME_RE = re.compile(
    r"^\s*#\s*WELL NAME:\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class DevTargetImportSummary:
    well_name: str
    profile_label: str
    kop_md_m: float
    t1_md_m: float
    t3_md_m: float
    entry_inc_deg: float
    build1_dls_deg_per_30m: tuple[float, ...] = ()
    build2_dls_deg_per_30m: tuple[float, ...] = ()
    horizontal_dls_deg_per_30m: tuple[float, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class DevTargetImportParsedWell:
    record: WelltrackRecord
    summary: DevTargetImportSummary
    imported_well: ImportedTrajectoryWell


def dev_trajectory_text_name(text: str, *, fallback_name: str = "dev_import_1") -> str:
    match = _DEV_WELL_NAME_RE.search(str(text or ""))
    if match is None:
        return str(fallback_name)
    normalized = str(match.group(1)).strip()
    return normalized or str(fallback_name)


def parse_dev_target_file(path: str | Path) -> DevTargetImportParsedWell:
    well = parse_reference_trajectory_dev_file(path, kind=REFERENCE_WELL_APPROVED)
    return _parsed_dev_target_well(well)


def parse_dev_target_directory(
    path: str | Path,
) -> list[DevTargetImportParsedWell]:
    wells = parse_reference_trajectory_dev_directories(
        [path],
        kind=REFERENCE_WELL_APPROVED,
    )
    return [_parsed_dev_target_well(well) for well in wells]


def parse_dev_target_payloads(
    payloads: Sequence[tuple[str, bytes]],
) -> list[DevTargetImportParsedWell]:
    results: list[DevTargetImportParsedWell] = []
    for fallback_index, (file_name, raw_bytes) in enumerate(payloads, start=1):
        text, _encoding = decode_welltrack_bytes(bytes(raw_bytes))
        fallback_name = Path(str(file_name or f"dev_import_{fallback_index}")).stem
        well_name = dev_trajectory_text_name(text, fallback_name=fallback_name)
        well = parse_reference_trajectory_dev_text(
            text,
            well_name=well_name,
            kind=REFERENCE_WELL_APPROVED,
        )
        results.append(_parsed_dev_target_well(well))
    return results


def _parsed_dev_target_well(well: ImportedTrajectoryWell) -> DevTargetImportParsedWell:
    record, summary = target_record_and_summary_from_dev_well(well)
    return DevTargetImportParsedWell(
        record=record,
        summary=summary,
        imported_well=well,
    )


def target_record_and_summary_from_dev_well(
    well: ImportedTrajectoryWell,
) -> tuple[WelltrackRecord, DevTargetImportSummary]:
    stations = pd.DataFrame(well.stations).reset_index(drop=True)
    if stations.empty or len(stations.index) < 2:
        raise WelltrackParseError(
            f".dev '{well.name}': требуется минимум 2 станции для импорта целей."
        )
    rows = _dev_analysis_rows(well)
    groups = _dev_activity_groups(rows)
    if not groups:
        raise WelltrackParseError(
            f".dev '{well.name}': не найден ни один участок изменения INC/DLS."
        )

    profile_label = "J-профиль" if len(groups) == 1 else "BUILD-HOLD-BUILD"
    first_dynamic_index = int(groups[0][0])
    kop_index = max(first_dynamic_index - 1, 0)
    build1_start, build1_end = groups[0]
    build2_start, build2_end = groups[1] if len(groups) > 1 else (0, 0)
    t1_index = build2_end if len(groups) > 1 else build1_end
    if t1_index <= 0 or t1_index >= len(rows.index) - 1:
        raise WelltrackParseError(
            f".dev '{well.name}': не удалось надежно определить точку t1."
        )

    if len(groups) == 1:
        build1_slice = rows.iloc[build1_start : build1_end + 1]
        build2_slice = rows.iloc[0:0]
    else:
        build1_slice = rows.iloc[build1_start : build1_end + 1]
        build2_slice = rows.iloc[build2_start : build2_end + 1]
    horizontal_slice = rows.iloc[t1_index + 1 :]

    surface_row = stations.iloc[0]
    t1_row = stations.iloc[t1_index]
    t3_row = stations.iloc[-1]
    record = WelltrackRecord(
        name=str(well.name),
        points=(
            WelltrackPoint(
                x=float(surface_row["X_m"]),
                y=float(surface_row["Y_m"]),
                z=float(surface_row["Z_m"]),
                md=float(rows["MD"].iloc[0]),
            ),
            WelltrackPoint(
                x=float(t1_row["X_m"]),
                y=float(t1_row["Y_m"]),
                z=float(t1_row["Z_m"]),
                md=float(rows["MD"].iloc[t1_index]),
            ),
            WelltrackPoint(
                x=float(t3_row["X_m"]),
                y=float(t3_row["Y_m"]),
                z=float(t3_row["Z_m"]),
                md=float(rows["MD"].iloc[-1]),
            ),
        ),
    )

    summary = DevTargetImportSummary(
        well_name=str(well.name),
        profile_label=profile_label,
        kop_md_m=float(rows["MD"].iloc[kop_index]),
        t1_md_m=float(rows["MD"].iloc[t1_index]),
        t3_md_m=float(rows["MD"].iloc[-1]),
        entry_inc_deg=float(rows["INCL"].iloc[t1_index]),
        build1_dls_deg_per_30m=_stable_unique_dls(build1_slice),
        build2_dls_deg_per_30m=_stable_unique_dls(build2_slice),
        horizontal_dls_deg_per_30m=_stable_unique_dls(horizontal_slice),
        note=_summary_note(
            profile_label=profile_label,
            build1_dls=_stable_unique_dls(build1_slice),
            build2_dls=_stable_unique_dls(build2_slice),
            horizontal_dls=_stable_unique_dls(horizontal_slice),
        ),
    )
    return record, summary


def dev_target_import_summary_dataframe(
    summaries: Sequence[DevTargetImportSummary],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary in summaries:
        rows.append(
            {
                "Скважина": str(summary.well_name),
                "Профиль": str(summary.profile_label),
                "KOP MD, м": float(summary.kop_md_m),
                "t1 MD, м": float(summary.t1_md_m),
                "t3 MD, м": float(summary.t3_md_m),
                "INC в t1, deg": float(summary.entry_inc_deg),
                "BUILD1 PI, deg/10m": _format_value_list(
                    tuple(float(dls_to_pi(value)) for value in summary.build1_dls_deg_per_30m)
                ),
                "BUILD2 PI, deg/10m": _format_value_list(
                    tuple(float(dls_to_pi(value)) for value in summary.build2_dls_deg_per_30m)
                ),
                "HORIZONTAL PI, deg/10m": _format_value_list(
                    tuple(
                        float(dls_to_pi(value))
                        for value in summary.horizontal_dls_deg_per_30m
                    )
                ),
                "Примечание": str(summary.note or "—"),
            }
        )
    return pd.DataFrame(rows)


def _dev_analysis_rows(well: ImportedTrajectoryWell) -> pd.DataFrame:
    if well.dev_export_rows is not None and not well.dev_export_rows.empty:
        return pd.DataFrame(well.dev_export_rows).reset_index(drop=True)
    stations = pd.DataFrame(well.stations).reset_index(drop=True)
    if "DLS_deg_per_30m" not in stations.columns or "INC_deg" not in stations.columns:
        raise WelltrackParseError(
            f".dev '{well.name}': отсутствуют колонки INC/DLS для чтения параметров."
        )
    return pd.DataFrame(
        {
            "MD": stations["MD_m"].astype(float),
            "X": stations["X_m"].astype(float),
            "Y": stations["Y_m"].astype(float),
            "Z": -stations["Z_m"].astype(float),
            "INCL": stations["INC_deg"].astype(float),
            "DLS": stations["DLS_deg_per_30m"].astype(float),
        }
    )


def _true_groups(mask: Sequence[bool]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
            continue
        if not value and start is not None:
            groups.append((start, index - 1))
            start = None
    if start is not None:
        groups.append((start, len(mask) - 1))
    return groups


def _dev_activity_groups(rows: pd.DataFrame) -> list[tuple[int, int]]:
    incl_values = rows["INCL"].astype(float).to_numpy(dtype=float)
    dls_values = rows["DLS"].abs().to_numpy(dtype=float)
    incl_deltas = np.abs(np.diff(incl_values, prepend=incl_values[0]))
    activity_mask = (
        (incl_deltas > _DEV_INCL_EPSILON_DEG)
        | (dls_values > _DEV_DLS_EPSILON_DEG_PER_30M)
    )
    raw_groups = _true_groups(activity_mask.tolist())
    if not raw_groups:
        return []
    if len(raw_groups) == 1:
        return raw_groups
    hold_separator = _dev_hold_separator_index(rows, raw_groups)
    if hold_separator is None:
        return [(raw_groups[0][0], raw_groups[-1][1])]
    return [
        (raw_groups[0][0], raw_groups[hold_separator][1]),
        (raw_groups[hold_separator + 1][0], raw_groups[-1][1]),
    ]


def _dev_hold_separator_index(
    rows: pd.DataFrame,
    groups: Sequence[tuple[int, int]],
) -> int | None:
    candidates: list[tuple[int, float, int]] = []
    for group_index in range(len(groups) - 1):
        gap_start = int(groups[group_index][1] + 1)
        gap_end = int(groups[group_index + 1][0] - 1)
        if gap_start > gap_end:
            continue
        gap_incl = rows["INCL"].iloc[gap_start : gap_end + 1].astype(float)
        if gap_incl.empty:
            continue
        if float(gap_incl.max() - gap_incl.min()) > _DEV_INCL_EPSILON_DEG:
            continue
        gap_rows = int(gap_end - gap_start + 1)
        if gap_rows < _DEV_HOLD_MIN_ROWS:
            continue
        gap_md = float(rows["MD"].iloc[gap_end] - rows["MD"].iloc[gap_start])
        candidates.append((gap_rows, gap_md, group_index))
    if not candidates:
        return None
    _gap_rows, _gap_md, separator_index = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return int(separator_index)


def _stable_unique_dls(rows: pd.DataFrame) -> tuple[float, ...]:
    values: list[float] = []
    seen: set[float] = set()
    for value in rows.get("DLS", pd.Series(dtype=float)).astype(float).tolist():
        rounded = round(float(value), 3)
        if abs(rounded) <= _DEV_DLS_EPSILON_DEG_PER_30M or rounded in seen:
            continue
        seen.add(rounded)
        values.append(rounded)
    return tuple(values)


def _format_value_list(values: Sequence[float]) -> str:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        return "—"
    if len(normalized) == 1:
        return f"{normalized[0]:.2f}"
    joined = " / ".join(f"{value:.2f}" for value in normalized)
    avg = sum(normalized) / float(len(normalized))
    return f"{joined} (avg {avg:.2f})"


def _summary_note(
    *,
    profile_label: str,
    build1_dls: Sequence[float],
    build2_dls: Sequence[float],
    horizontal_dls: Sequence[float],
) -> str:
    notes: list[str] = []
    if len(build1_dls) > 1:
        notes.append("BUILD1 с переменным ПИ")
    if len(build2_dls) > 1:
        notes.append("BUILD2 с переменным ПИ")
    if len(horizontal_dls) > 1:
        notes.append("HORIZONTAL с переменным ПИ")
    if str(profile_label) == "J-профиль" and build2_dls:
        notes.append("Второй BUILD не ожидается для J-профиля")
    return "; ".join(notes) if notes else "—"
