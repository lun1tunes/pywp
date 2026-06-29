from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
import math

import pandas as pd

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.pilot_wells import (
    is_pilot_name,
    is_zbs_name,
    parent_name_for_pilot,
    well_name_key,
)
from pywp import ptc_target_records
from pywp.ptc_target_records import record_horizontal_length_preprocess_skip_reason
from pywp.ptc_sidetrack_state import queue_editor_sidetrack_window_override

__all__ = [
    "apply_edit_targets_changes",
    "bulk_horizontal_length_changes",
    "edit_target_point",
    "handle_three_edit_event",
    "invalidate_results_for_edited_targets",
    "pending_edit_target_names",
    "raw_records_editor_changes",
    "queue_all_wells_results_focus",
    "records_with_edit_targets",
    "unique_well_names",
]

BaseRowFactory = Callable[[WelltrackRecord], dict[str, object]]
ApplyChangesCallback = Callable[[object, str], list[str]]


def edit_target_point(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        point = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in point):
        return None
    return point


def bulk_horizontal_length_changes(
    records: Iterable[WelltrackRecord],
    *,
    target_length_m: float,
) -> tuple[list[dict[str, object]], list[str]]:
    try:
        normalized_target_length_m = float(target_length_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("Новая длина ГС должна быть числом.") from exc
    if not math.isfinite(normalized_target_length_m) or normalized_target_length_m <= 0.0:
        raise ValueError("Новая длина ГС должна быть положительным конечным числом.")

    records_list = list(records)
    pilot_parent_keys = {
        well_name_key(parent_name_for_pilot(record.name))
        for record in records_list
        if is_pilot_name(record.name)
    }
    changes: list[dict[str, object]] = []
    skipped_names: list[str] = []
    for record in records_list:
        if (
            record_horizontal_length_preprocess_skip_reason(
                record,
                has_pilot=well_name_key(record.name) in pilot_parent_keys,
            )
            != "—"
        ):
            skipped_names.append(str(record.name))
            continue
        pair_indices = _horizontal_length_pair_indices(record)
        if pair_indices is None:
            if tuple(record.points):
                skipped_names.append(str(record.name))
            continue
        point_updates: list[dict[str, object]] = []
        failed = False
        for t1_index, t3_index in pair_indices:
            next_t3_position = _scaled_t3_position(
                record.points[t1_index],
                record.points[t3_index],
                target_length_m=normalized_target_length_m,
            )
            if next_t3_position is None:
                failed = True
                break
            if _point_matches_xyz(record.points[t3_index], next_t3_position):
                continue
            point_updates.append(
                {
                    "index": int(t3_index),
                    "position": next_t3_position,
                }
            )
        if failed:
            skipped_names.append(str(record.name))
            continue
        if point_updates:
            changes.append(
                {
                    "name": str(record.name),
                    "points": point_updates,
                }
            )
    return changes, unique_well_names(skipped_names)


def _horizontal_length_pair_indices(
    record: WelltrackRecord,
) -> tuple[tuple[int, int], ...] | None:
    points = tuple(record.points)
    if is_zbs_name(record.name):
        if len(points) < 2 or len(points) % 2 != 0:
            return None
        return tuple((index, index + 1) for index in range(0, len(points), 2))
    if len(points) < 3 or (len(points) - 1) % 2 != 0:
        return None
    return tuple((index, index + 1) for index in range(1, len(points), 2))


def _scaled_t3_position(
    t1_point: WelltrackPoint,
    t3_point: WelltrackPoint,
    *,
    target_length_m: float,
) -> list[float] | None:
    coords = (
        float(t1_point.x),
        float(t1_point.y),
        float(t1_point.z),
        float(t3_point.x),
        float(t3_point.y),
        float(t3_point.z),
    )
    if not all(math.isfinite(value) for value in coords):
        return None
    dx = float(t3_point.x) - float(t1_point.x)
    dy = float(t3_point.y) - float(t1_point.y)
    dz = float(t3_point.z) - float(t1_point.z)
    current_length_m = math.sqrt(dx * dx + dy * dy + dz * dz)
    if current_length_m <= 1e-9:
        return None
    scale = float(target_length_m) / current_length_m
    return [
        float(t1_point.x) + dx * scale,
        float(t1_point.y) + dy * scale,
        float(t1_point.z) + dz * scale,
    ]


def _point_matches_xyz(point: WelltrackPoint, xyz: list[float]) -> bool:
    return (
        math.isclose(float(point.x), float(xyz[0]), abs_tol=1e-9)
        and math.isclose(float(point.y), float(xyz[1]), abs_tol=1e-9)
        and math.isclose(float(point.z), float(xyz[2]), abs_tol=1e-9)
    )


def _sidetrack_window_change(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    kind = str(value.get("kind", "md")).strip().lower()
    if kind not in {"md", "z"}:
        return None
    raw_value = value.get("value_m", value.get("md_m", value.get("z_m")))
    try:
        value_m = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_m):
        return None
    position = edit_target_point(value.get("position"))
    result: dict[str, object] = {
        "kind": kind,
        "value_m": value_m,
    }
    if position is not None:
        result["position"] = position
    return result


def records_with_edit_targets(
    records: Iterable[WelltrackRecord],
    change_map: Mapping[str, Mapping[str, object]],
) -> tuple[list[WelltrackRecord], list[str]]:
    updated_records: list[WelltrackRecord] = []
    updated_names: list[str] = []
    for record in records:
        record_name = str(record.name)
        if record_name not in change_map:
            updated_records.append(record)
            continue
        delta = change_map[record_name]
        raw_points = delta.get("points")
        if isinstance(raw_points, list):
            old_points = list(record.points)
            changed = False
            for raw_entry in raw_points:
                if not isinstance(raw_entry, Mapping):
                    continue
                try:
                    point_index = int(raw_entry.get("index", -1))
                except (TypeError, ValueError):
                    continue
                new_point = edit_target_point(raw_entry.get("position"))
                if (
                    new_point is None
                    or point_index < 0
                    or point_index >= len(old_points)
                ):
                    continue
                old_point = old_points[point_index]
                if (
                    math.isclose(float(old_point.x), float(new_point[0]), abs_tol=1e-9)
                    and math.isclose(
                        float(old_point.y), float(new_point[1]), abs_tol=1e-9
                    )
                    and math.isclose(
                        float(old_point.z), float(new_point[2]), abs_tol=1e-9
                    )
                ):
                    continue
                old_points[point_index] = WelltrackPoint(
                    x=float(new_point[0]),
                    y=float(new_point[1]),
                    z=float(new_point[2]),
                    md=float(old_point.md),
                )
                changed = True
            if changed:
                updated_records.append(
                    WelltrackRecord(name=record.name, points=tuple(old_points))
                )
                updated_names.append(record_name)
            else:
                updated_records.append(record)
            continue

        if "t1" not in delta or "t3" not in delta:
            updated_records.append(record)
            continue
        new_t1 = delta["t1"]
        new_t3 = delta["t3"]
        if len(record.points) != 3:
            updated_records.append(record)
            continue
        old_points = record.points
        new_points = (
            old_points[0],
            WelltrackPoint(
                x=float(new_t1[0]),
                y=float(new_t1[1]),
                z=float(new_t1[2]),
                md=float(old_points[1].md),
            ),
            WelltrackPoint(
                x=float(new_t3[0]),
                y=float(new_t3[1]),
                z=float(new_t3[2]),
                md=float(old_points[2].md),
            ),
        )
        updated_records.append(
            WelltrackRecord(name=record.name, points=new_points)
        )
        updated_names.append(record_name)
    return updated_records, updated_names


def unique_well_names(names: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in names:
        name = str(value).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def raw_records_editor_changes(
    records: Iterable[WelltrackRecord],
    edited_rows: object,
) -> list[dict[str, object]]:
    expected_df = ptc_target_records.raw_records_dataframe(list(records)).reset_index(
        drop=True
    )
    edited_df = pd.DataFrame(edited_rows).reset_index(drop=True)
    required_columns = ("Скважина", "Точка", "X, м", "Y, м", "Z, м")
    missing_columns = [
        column for column in required_columns if column not in edited_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "В таблице текущих точек отсутствуют обязательные столбцы: "
            + ", ".join(missing_columns)
            + "."
        )
    if len(edited_df) != len(expected_df):
        raise ValueError(
            "В текущих точках нельзя добавлять или удалять строки. "
            "Изменяйте только координаты X/Y/Z."
        )

    point_index_by_well: dict[str, int] = {}
    changes_by_name: dict[str, list[dict[str, object]]] = {}
    ordered_names: list[str] = []

    for row_no, (_, expected_row) in enumerate(expected_df.iterrows(), start=1):
        edited_row = edited_df.iloc[row_no - 1]
        well_name = str(expected_row["Скважина"]).strip()
        point_name = str(expected_row["Точка"]).strip()
        if (
            str(edited_row.get("Скважина", "")).strip() != well_name
            or str(edited_row.get("Точка", "")).strip() != point_name
        ):
            raise ValueError(
                "В текущих точках нельзя менять столбцы «Скважина» и «Точка»."
            )
        try:
            x = float(edited_row["X, м"])
            y = float(edited_row["Y, м"])
            z = float(edited_row["Z, м"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Строка {row_no}: координаты X/Y/Z должны быть числами."
            ) from exc
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError(
                f"Строка {row_no}: координаты X/Y/Z должны быть конечными числами."
            )

        point_index = point_index_by_well.get(well_name, 0)
        point_index_by_well[well_name] = point_index + 1

        if (
            math.isclose(float(expected_row["X, м"]), x, abs_tol=1e-9)
            and math.isclose(float(expected_row["Y, м"]), y, abs_tol=1e-9)
            and math.isclose(float(expected_row["Z, м"]), z, abs_tol=1e-9)
        ):
            continue

        if well_name not in changes_by_name:
            changes_by_name[well_name] = []
            ordered_names.append(well_name)
        changes_by_name[well_name].append(
            {
                "index": point_index,
                "position": [x, y, z],
            }
        )

    return [
        {"name": well_name, "points": changes_by_name[well_name]}
        for well_name in ordered_names
    ]


def _expanded_invalidated_names(
    records: Iterable[WelltrackRecord],
    edited_names: Iterable[object],
) -> list[str]:
    canonical_name_by_key = {
        well_name_key(record.name): str(record.name)
        for record in records
    }
    result: list[str] = []
    seen_keys: set[str] = set()

    def _append(name: object) -> None:
        text = str(name).strip()
        if not text:
            return
        canonical = canonical_name_by_key.get(well_name_key(text), text)
        canonical_key = well_name_key(canonical)
        if canonical_key in seen_keys:
            return
        seen_keys.add(canonical_key)
        result.append(canonical)

    for name in edited_names:
        _append(name)
    for name in tuple(result):
        if not is_pilot_name(name):
            continue
        _append(parent_name_for_pilot(name))
    return result


def pending_edit_target_names(
    session_state: Mapping[str, object],
) -> list[str]:
    pending = unique_well_names(
        session_state.get("wt_edit_targets_pending_names") or []
    )
    if pending:
        return pending
    return unique_well_names(
        session_state.get("wt_edit_targets_highlight_names") or []
    )


def _edit_target_highlight_indices(
    delta: Mapping[str, object],
    *,
    record: WelltrackRecord | None = None,
) -> list[int]:
    def _point_changed(point_index: int, next_position: list[float] | None) -> bool:
        if record is None or next_position is None:
            return True
        if point_index < 0 or point_index >= len(record.points):
            return False
        old_point = record.points[point_index]
        return not (
            math.isclose(float(old_point.x), float(next_position[0]), abs_tol=1e-9)
            and math.isclose(float(old_point.y), float(next_position[1]), abs_tol=1e-9)
            and math.isclose(float(old_point.z), float(next_position[2]), abs_tol=1e-9)
        )

    raw_points = delta.get("points")
    if isinstance(raw_points, list):
        result: list[int] = []
        for raw_entry in raw_points:
            if not isinstance(raw_entry, Mapping):
                continue
            try:
                point_index = int(raw_entry.get("index", -1))
            except (TypeError, ValueError):
                continue
            position = edit_target_point(raw_entry.get("position"))
            if (
                point_index >= 0
                and point_index not in result
                and _point_changed(point_index, position)
            ):
                result.append(point_index)
        return result
    if "t1" in delta and "t3" in delta:
        result = []
        t1 = edit_target_point(delta.get("t1"))
        t3 = edit_target_point(delta.get("t3"))
        if _point_changed(1, t1):
            result.append(1)
        if _point_changed(2, t3):
            result.append(2)
        return result
    return []


def invalidate_results_for_edited_targets(
    session_state: MutableMapping[str, object],
    *,
    records: Iterable[WelltrackRecord],
    edited_names: Iterable[str],
    base_row_factory: BaseRowFactory,
) -> None:
    edited_name_set = {
        str(name) for name in edited_names if str(name).strip()
    }
    if not edited_name_set:
        return

    ordered_records = list(records)
    existing_rows = session_state.get("wt_summary_rows")
    if existing_rows is not None:
        existing_rows_by_name = {
            str(row.get("Скважина", "")).strip(): dict(row)
            for row in existing_rows
            if isinstance(row, Mapping)
        }
        session_state["wt_summary_rows"] = [
            (
                base_row_factory(record)
                if str(record.name) in edited_name_set
                else existing_rows_by_name.get(
                    str(record.name),
                    base_row_factory(record),
                )
            )
            for record in ordered_records
        ]

    existing_successes = session_state.get("wt_successes")
    if existing_successes is not None:
        session_state["wt_successes"] = [
            success
            for success in existing_successes
            if (
                str(getattr(success, "name", "")).strip()
                not in edited_name_set
            )
        ]

    session_state["wt_last_anticollision_resolution"] = None
    session_state["wt_last_anticollision_previous_successes"] = {}
    session_state["wt_prepared_well_overrides"] = {}
    session_state["wt_prepared_override_message"] = ""
    session_state["wt_prepared_recommendation_id"] = ""
    session_state["wt_anticollision_prepared_cluster_id"] = ""
    session_state["wt_prepared_recommendation_snapshot"] = None


def apply_edit_targets_changes(
    session_state: MutableMapping[str, object],
    changes: object,
    *,
    source: str,
    base_row_factory: BaseRowFactory,
) -> list[str]:
    if not isinstance(changes, list) or not changes:
        return []
    records = session_state.get("wt_records")
    if not records:
        return []
    change_map: dict[str, dict[str, object]] = {}
    sidetrack_window_map: dict[str, dict[str, object]] = {}
    for entry in changes:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()
        sidetrack_window = _sidetrack_window_change(entry.get("sidetrack_window"))
        if name and sidetrack_window is not None:
            sidetrack_window_map[name] = sidetrack_window
        raw_points = entry.get("points")
        parsed_points: list[dict[str, object]] = []
        if isinstance(raw_points, list):
            for raw_point in raw_points:
                if not isinstance(raw_point, Mapping):
                    continue
                try:
                    point_index = int(raw_point.get("index", -1))
                except (TypeError, ValueError):
                    continue
                point = edit_target_point(raw_point.get("position"))
                if point is None or point_index < 0:
                    continue
                parsed_points.append(
                    {
                        "index": point_index,
                        "position": point,
                    }
                )
        if name and parsed_points:
            change_map[name] = {"points": parsed_points}
            continue
        t1 = edit_target_point(entry.get("t1"))
        t3 = edit_target_point(entry.get("t3"))
        if name and t1 is not None and t3 is not None:
            change_map[name] = {"t1": t1, "t3": t3}
    if not change_map and not sidetrack_window_map:
        return []
    record_by_name = {
        str(record.name): record
        for record in records  # type: ignore[union-attr]
        if isinstance(record, WelltrackRecord)
    }
    updated_records = list(records)  # type: ignore[arg-type]
    updated_target_names: list[str] = []
    if change_map:
        updated_records, updated_target_names = records_with_edit_targets(
            records=records,  # type: ignore[arg-type]
            change_map=change_map,
        )
    sidetrack_window_names: list[str] = []
    for name, window_change in sidetrack_window_map.items():
        queue_editor_sidetrack_window_override(
            session_state,
            well_name=name,
            kind=str(window_change["kind"]),
            value_m=float(window_change["value_m"]),
        )
        sidetrack_window_names.append(name)
    updated_names = unique_well_names(
        [*updated_target_names, *sidetrack_window_names]
    )
    if not updated_names:
        return []
    invalidated_names = _expanded_invalidated_names(updated_records, updated_names)

    effective_change_map = {
        name: change_map[name] for name in updated_target_names if name in change_map
    }
    original_records = session_state.get("wt_records_original")
    if original_records and effective_change_map:
        updated_original_records, _ = records_with_edit_targets(
            records=original_records,  # type: ignore[arg-type]
            change_map=effective_change_map,
        )
        session_state["wt_records_original"] = updated_original_records

    if updated_target_names:
        session_state["wt_records"] = updated_records
    invalidate_results_for_edited_targets(
        session_state,
        records=updated_records,
        edited_names=invalidated_names,
        base_row_factory=base_row_factory,
    )
    pending_names = unique_well_names(
        [
            *pending_edit_target_names(session_state),
            *invalidated_names,
        ]
    )
    session_state["wt_edit_targets_pending_names"] = pending_names
    session_state["wt_edit_targets_applied"] = updated_names
    session_state["wt_edit_targets_applied_source"] = str(source or "").strip()
    existing_highlight_points = session_state.get("wt_edit_targets_highlight_points")
    if isinstance(existing_highlight_points, Mapping):
        highlight_points: dict[str, list[int]] = {
            str(name): [
                int(index)
                for index in indices
                if isinstance(index, int) or str(index).lstrip("-").isdigit()
            ]
            for name, indices in existing_highlight_points.items()
            if isinstance(indices, list)
        }
    else:
        highlight_points = {}
    for name in updated_names:
        point_indices = _edit_target_highlight_indices(
            change_map.get(name, {}),
            record=record_by_name.get(str(name)),
        )
        if point_indices:
            highlight_points[str(name)] = point_indices
    session_state["wt_edit_targets_highlight_points"] = {
        name: indices
        for name, indices in highlight_points.items()
        if name in pending_names
    }
    session_state["wt_edit_targets_highlight_names"] = [
        name
        for name in pending_names
        if name in session_state["wt_edit_targets_highlight_points"]
    ]
    session_state["wt_edit_targets_last_source"] = str(source)
    session_state["wt_edit_targets_highlight_version"] = (
        int(session_state.get("wt_edit_targets_highlight_version", 0)) + 1
    )
    session_state["wt_last_error"] = ""
    session_state["wt_pending_selected_names"] = list(pending_names)
    queue_all_wells_results_focus(session_state)
    return updated_names


def handle_three_edit_event(
    session_state: MutableMapping[str, object],
    event: object,
    *,
    apply_changes: ApplyChangesCallback,
    bump_three_viewer_nonce: Callable[[], None],
) -> bool:
    if not isinstance(event, Mapping):
        return False
    if str(event.get("type") or "") != "pywp:editTargets":
        return False
    nonce = str(event.get("nonce") or "")
    if nonce and nonce == str(session_state.get("wt_last_edit_targets_nonce", "")):
        return False
    updated_names = apply_changes(event.get("changes"), "three_viewer")
    if not updated_names:
        return False
    if nonce:
        session_state["wt_last_edit_targets_nonce"] = nonce
    bump_three_viewer_nonce()
    return True


def queue_all_wells_results_focus(
    session_state: MutableMapping[str, object],
) -> None:
    session_state["wt_pending_all_wells_results_focus"] = True
