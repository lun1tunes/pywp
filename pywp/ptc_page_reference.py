from __future__ import annotations

import pandas as pd
import streamlit as st

from pywp import ptc_core as wt
from pywp import ptc_reference_state as reference_state
from pywp import ptc_welltrack_io
from pywp.reference_trajectories import (
    parse_reference_trajectory_dev_directories,
    parse_reference_trajectory_welltrack_text,
)
from pywp.pilot_wells import is_zbs_name, parent_name_for_zbs, well_name_key

__all__ = ["render_reference_section"]


def _clear_reference_import_state(kind: str) -> None:
    reference_state.clear_reference_import_state(
        kind,
        on_clear=lambda: wt._reset_anticollision_view_state(clear_prepared=True),
    )


def _render_reference_kind_import_block(*, kind: str) -> None:
    title = reference_state.reference_kind_title(kind)
    state_mode_key = f"ptc_reference_source_mode::{kind}"
    source_options = [
        "Загрузить .dev",
        "Путь к WELLTRACK",
        "Загрузить WELLTRACK",
    ]
    if state_mode_key not in st.session_state:
        legacy_mode = str(
            st.session_state.get(
                reference_state.reference_source_mode_key(kind),
                "",
            )
        ).strip()
        st.session_state[state_mode_key] = (
            legacy_mode if legacy_mode in source_options else "Загрузить .dev"
        )
    if str(st.session_state.get(state_mode_key, "")).strip() not in set(
        source_options
    ):
        st.session_state[state_mode_key] = "Загрузить .dev"
    mode = st.radio(
        f"Источник для {title.lower()}",
        options=source_options,
        key=state_mode_key,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state[reference_state.reference_source_mode_key(kind)] = mode
    uploaded_file = None
    if mode == "Загрузить .dev":
        folder_count_key = reference_state.reference_dev_folder_count_key(kind)
        try:
            folder_count = int(st.session_state.get(folder_count_key, 1))
        except (TypeError, ValueError):
            folder_count = 1
        folder_count = max(1, folder_count)
        st.session_state[folder_count_key] = folder_count
        for index in range(folder_count):
            st.text_input(
                "Папка с .dev файлами"
                if index == 0
                else f"Папка с .dev файлами #{index + 1}",
                key=reference_state.reference_dev_folder_path_key(kind, index),
                placeholder="tests/test_data/dev_fact",
            )
        if st.button(
            "Добавить ещё папку",
            key=f"ptc_reference_{kind}_add_dev_folder",
            icon=":material/create_new_folder:",
            use_container_width=True,
        ):
            st.session_state[folder_count_key] = folder_count + 1
            st.rerun()
        st.caption(
            "Импортируются все `.dev` файлы из папок. "
            "Имя берется из файла без `.dev`, "
            "координаты - из колонок `MD X Y Z`."
        )
    elif mode == "Путь к WELLTRACK":
        st.text_input(
            "Путь к WELLTRACK",
            key=reference_state.reference_welltrack_path_key(kind),
            placeholder="tests/test_data/WELLTRACKS3.INC",
        )
    else:
        uploaded_file = st.file_uploader(
            f"WELLTRACK файл для {title.lower()}",
            type=["inc", "txt", "data", "ecl"],
            key=f"ptc_reference_{kind}_welltrack_file",
        )

    action_col, clear_col = st.columns(2, gap="small")
    import_clicked = action_col.button(
        f"Загрузить {title.lower()}",
        key=f"ptc_reference_import_{kind}",
        type="primary",
        icon=":material/upload_file:",
        use_container_width=True,
    )
    clear_col.button(
        f"Очистить {title.lower()}",
        key=f"ptc_reference_clear_{kind}",
        icon=":material/delete:",
        use_container_width=True,
        on_click=_clear_reference_import_state,
        kwargs={"kind": kind},
    )

    if import_clicked:
        with st.status(f"Импорт {title.lower()}...", expanded=True) as status:
            try:
                if mode == "Загрузить .dev":
                    parsed = parse_reference_trajectory_dev_directories(
                        reference_state.reference_dev_folder_paths(kind),
                        kind=kind,
                    )
                elif mode == "Путь к WELLTRACK":
                    parsed = parse_reference_trajectory_welltrack_text(
                        ptc_welltrack_io.read_welltrack_file(
                            str(
                                st.session_state.get(
                                    reference_state.reference_welltrack_path_key(
                                        kind
                                    ),
                                    "",
                                )
                            ),
                            info=st.info,
                            warning=st.warning,
                            error=st.error,
                        ),
                        kind=kind,
                    )
                else:
                    payload = (
                        b""
                        if uploaded_file is None
                        else uploaded_file.getvalue()
                    )
                    parsed = parse_reference_trajectory_welltrack_text(
                        ptc_welltrack_io.decode_welltrack_payload(
                            payload,
                            source_label=(
                                f"WELLTRACK `{getattr(uploaded_file, 'name', 'uploaded')}`"
                            ),
                            info=st.info,
                            warning=st.warning,
                        ),
                        kind=kind,
                    )
                reference_state.set_reference_wells_for_kind(
                    kind=kind,
                    wells=parsed,
                )
                wt._reset_anticollision_view_state(clear_prepared=True)
                status.write(f"Загружено скважин: {len(parsed)}.")
                status.update(
                    label=f"{title} импортированы",
                    state="complete",
                    expanded=False,
                )
                st.rerun()
            except wt.WelltrackParseError as exc:
                status.write(str(exc))
                status.update(
                    label=f"Ошибка импорта: {title.lower()}",
                    state="error",
                    expanded=True,
                )

    current_wells = tuple(reference_state.reference_kind_wells(kind))
    if current_wells:
        st.caption(f"Загружено {len(current_wells)} скважин.")
    else:
        st.caption("Скважины этого типа не загружены.")


def render_reference_section() -> None:
    st.markdown("## 3. Загрузка фактического фонда")
    st.caption(
        "Если на месторождении уже есть фактический фонд или утверждённый "
        "проектный (проработанный в ЦСБ) - загрузите его из папок `.dev` "
        "или в формате WELLTRACK."
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("### Фактический фонд")
        _render_reference_kind_import_block(kind=wt.REFERENCE_WELL_ACTUAL)
    with c2:
        st.markdown("### Утверждённый проектный фонд")
        _render_reference_kind_import_block(kind=wt.REFERENCE_WELL_APPROVED)

    reference_wells = tuple(reference_state.reference_wells_from_state())
    if reference_wells:
        m1, m2, m3 = st.columns(3, gap="small")
        m1.metric("Доп. скважин", f"{len(reference_wells)}")
        m2.metric(
            "Фактических",
            f"{sum(1 for item in reference_wells if str(item.kind) == wt.REFERENCE_WELL_ACTUAL)}",
        )
        m3.metric(
            "Проектных утверждённых",
            f"{sum(1 for item in reference_wells if str(item.kind) == wt.REFERENCE_WELL_APPROVED)}",
        )

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
        analyses = wt._actual_fund_analyses(actual_wells)
        wt._render_actual_fund_analysis_panel(analyses=analyses)

    approved_wells = tuple(
        reference_state.reference_kind_wells(wt.REFERENCE_WELL_APPROVED)
    )
    if approved_wells:
        with st.expander(
            "Просмотр загруженных утверждённых проектных скважин",
            expanded=False,
        ):
            try:
                approved_analyses = wt.build_actual_fund_well_analyses(
                    approved_wells
                )
            except Exception as exc:
                st.error(
                    "Не удалось построить просмотр утверждённого проектного фонда."
                )
                st.caption(f"{type(exc).__name__}: {exc}")
            else:
                wt._render_reference_well_detail(
                    approved_analyses,
                    select_label="Просмотр утвержденной проектной скважины",
                    selected_key="wt_approved_fund_selected_well",
                )


def _render_zbs_actual_match_info(actual_wells: tuple[object, ...]) -> None:
    records = tuple(st.session_state.get("wt_records") or ())
    zbs_records = [
        record
        for record in records
        if is_zbs_name(getattr(record, "name", ""))
    ]
    if not zbs_records:
        return
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
    if missing:
        st.warning(
            "Для боковых стволов не найдены фактические скважины: "
            + ", ".join(
                f'{zbs_name} ждёт "{parent_name}"'
                for zbs_name, parent_name in missing
            )
            + "."
        )
