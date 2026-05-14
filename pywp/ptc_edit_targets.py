from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
import math

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord

__all__ = [
    "apply_edit_targets_changes",
    "edit_target_point",
    "handle_three_edit_event",
    "invalidate_results_for_edited_targets",
    "pending_edit_target_names",
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
    for entry in changes:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()
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
    if not change_map:
        return []
    updated_records, updated_names = records_with_edit_targets(
        records=records,  # type: ignore[arg-type]
        change_map=change_map,
    )
    if not updated_names:
        return []

    effective_change_map = {
        name: change_map[name] for name in updated_names if name in change_map
    }
    original_records = session_state.get("wt_records_original")
    if original_records:
        updated_original_records, _ = records_with_edit_targets(
            records=original_records,  # type: ignore[arg-type]
            change_map=effective_change_map,
        )
        session_state["wt_records_original"] = updated_original_records

    session_state["wt_records"] = updated_records
    invalidate_results_for_edited_targets(
        session_state,
        records=updated_records,
        edited_names=updated_names,
        base_row_factory=base_row_factory,
    )
    pending_names = unique_well_names(
        [
            *pending_edit_target_names(session_state),
            *updated_names,
        ]
    )
    session_state["wt_edit_targets_pending_names"] = pending_names
    session_state["wt_edit_targets_applied"] = updated_names
    session_state["wt_edit_targets_highlight_names"] = pending_names
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
