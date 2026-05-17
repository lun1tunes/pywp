from __future__ import annotations

from types import SimpleNamespace

import pytest

from pywp import ptc_core
from pywp import ptc_page_results
from pywp import welltrack_batch
from pywp.anticollision import AntiCollisionAnalysis
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord


class _RerunRequested(Exception):
    pass


def test_full_anticollision_recalc_button_resets_cache_and_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class FakeStreamlit:
        def button(self, *args: object, **kwargs: object) -> bool:
            assert args[0] == "Полный пересчёт anti-collision"
            assert kwargs["use_container_width"] is True
            return True

        def rerun(self) -> None:
            raise _RerunRequested

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_reset_anticollision_view_state",
        lambda *, clear_prepared: calls.append(bool(clear_prepared)),
    )

    with pytest.raises(_RerunRequested):
        ptc_page_results._render_full_anticollision_recalc_button()

    assert calls == [True]


def test_full_anticollision_recalc_button_does_not_reset_when_not_clicked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class FakeStreamlit:
        def button(self, *args: object, **kwargs: object) -> bool:
            return False

        def rerun(self) -> None:
            raise AssertionError("rerun should not be called")

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_reset_anticollision_view_state",
        lambda *, clear_prepared: calls.append(bool(clear_prepared)),
    )

    assert ptc_page_results._render_full_anticollision_recalc_button() is False
    assert calls == []


def test_anticollision_panel_pauses_on_pending_target_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, str]] = []

    class FakeStreamlit:
        session_state: dict[str, object] = {}

        def info(self, message: str) -> None:
            messages.append(("info", str(message)))

        def caption(self, message: str) -> None:
            messages.append(("caption", str(message)))

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_pending_edit_target_names",
        lambda: ["WELL-A"],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_cached_anti_collision_view_model",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("anti-collision must wait for recalculated edited wells")
        ),
    )

    ptc_page_results._render_anticollision_panel(
        successes=[object(), object()],
        records=[],
        summary_rows=[],
        focus_pad_id="",
        focus_pad_well_names=[],
    )

    assert messages[0] == (
        "info",
        "Anti-collision анализ приостановлен: есть изменённые в 3D точки "
        "t1/t3, которые ещё не пересчитаны.",
    )
    assert messages[1][0] == "caption"
    assert "WELL-A" in messages[1][1]
    assert "Предыдущий anti-collision расчёт сохранён" in messages[1][1]


def test_anticollision_panel_shows_cached_snapshot_when_targets_are_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=1,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )
    target_only = SimpleNamespace(name="WELL-A")
    calls: dict[str, object] = {}

    class FakeColumn:
        def plotly_chart(self, figure: object, **kwargs: object) -> None:
            calls["plotly"] = (figure, kwargs)

    class FakeStreamlit:
        session_state: dict[str, object] = {
            "wt_anticollision_analysis_cache": {
                "analysis": analysis,
                "recommendations": (),
                "clusters": (),
            }
        }

        def info(self, message: str) -> None:
            calls["info"] = str(message)

        def caption(self, message: str) -> None:
            calls.setdefault("captions", []).append(str(message))

        def columns(self, *args: object, **kwargs: object) -> list[FakeColumn]:
            return [FakeColumn(), FakeColumn()]

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_pending_edit_target_names",
        lambda: ["WELL-A"],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_cached_anti_collision_view_model",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("anti-collision must not recalculate pending edits")
        ),
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_failed_target_only_wells",
        lambda **_kwargs: [target_only],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_well_color_map",
        lambda _records: {"WELL-A": "#123456"},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_all_wells_anticollision_three_payload",
        lambda *args, **kwargs: calls.setdefault("payload_kwargs", kwargs) or {},
    )

    def fake_overrides(**kwargs: object) -> dict[str, object]:
        calls["override_kwargs"] = kwargs
        return {"edit_wells": []}

    monkeypatch.setattr(
        ptc_page_results.wt,
        "_anticollision_three_payload_overrides",
        fake_overrides,
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_render_three_payload",
        lambda **kwargs: calls.setdefault("render_payload", kwargs),
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_all_wells_anticollision_plan_figure",
        lambda *args, **kwargs: "plan-figure",
    )

    ptc_page_results._render_anticollision_panel(
        successes=[],
        records=[],
        summary_rows=[],
        focus_pad_id="",
        focus_pad_well_names=[],
    )

    assert "приостановлен" in str(calls["info"])
    assert any("последний anti-collision снимок" in item for item in calls["captions"])
    assert calls["plotly"][0] == "plan-figure"
    assert calls["override_kwargs"]["target_only_wells"] == [target_only]
    assert calls["override_kwargs"]["target_only_name_to_color"] == {"WELL-A": "#123456"}


def test_target_edit_overview_keeps_reference_wells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_only = SimpleNamespace(name="WELL-A")
    reference_wells = (object(),)
    calls: dict[str, object] = {}

    class FakeColumn:
        def plotly_chart(self, figure: object, **kwargs: object) -> None:
            calls["plotly"] = (figure, kwargs)

    class FakeStreamlit:
        def info(self, message: str) -> None:
            calls["info"] = str(message)

        def markdown(self, message: str) -> None:
            calls["markdown"] = str(message)

        def caption(self, message: str) -> None:
            calls["caption"] = str(message)

        def columns(self, *args: object, **kwargs: object) -> list[FakeColumn]:
            return [FakeColumn(), FakeColumn()]

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_failed_target_only_wells",
        lambda **_kwargs: [target_only],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_well_color_map",
        lambda _records: {},
    )
    monkeypatch.setattr(
        ptc_page_results.reference_state,
        "reference_wells_from_state",
        lambda: reference_wells,
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_all_wells_three_payload",
        lambda *args, **kwargs: calls.setdefault("three_kwargs", kwargs) or {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_trajectory_three_payload_overrides",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_render_three_payload",
        lambda **kwargs: calls.setdefault("render_payload", kwargs),
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_all_wells_plan_figure",
        lambda *args, **kwargs: calls.setdefault("plan_kwargs", kwargs) or "plan",
    )

    rendered = ptc_page_results._render_target_edit_overview(
        successes=[],
        records=[],
        summary_rows=[],
        title="### Test",
        empty_message="empty",
        focus_pad_well_names=[],
        show_focus_selector=False,
    )

    assert rendered is True
    assert calls["three_kwargs"]["reference_wells"] == reference_wells
    assert calls["plan_kwargs"]["reference_wells"] == reference_wells


def test_apply_pad_order_optimization_updates_source_records_and_keeps_ac_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1000.0),
                WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=1200.0),
            ),
        )
    ]
    new_records = [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=50.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1000.0),
                WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=1200.0),
            ),
        )
    ]
    success = SimpleNamespace(name="WELL-A")
    ac_cache = {"key": "old", "pair_cache": {("WELL-A", "WELL-B"): object()}}
    state: dict[str, object] = {
        "wt_records": old_records,
        "wt_records_original": old_records,
        "wt_successes": [SimpleNamespace(name="OLD")],
        "wt_summary_rows": [{"Скважина": "OLD"}],
        "wt_anticollision_analysis_cache": ac_cache,
    }
    reset_calls: list[bool] = []

    class FakeStreamlit:
        session_state = state

    def fake_reset(*, clear_prepared: bool) -> None:
        reset_calls.append(bool(clear_prepared))
        state["wt_anticollision_analysis_cache"] = {}

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_reset_anticollision_view_state",
        fake_reset,
    )
    monkeypatch.setattr(
        welltrack_batch.WelltrackBatchPlanner,
        "_row_from_success",
        staticmethod(
            lambda *, record, success: {
                "Скважина": str(record.name),
                "Статус": "ok",
            }
        ),
    )
    monkeypatch.setattr(
        welltrack_batch,
        "merge_batch_results",
        lambda **kwargs: (
            list(kwargs["new_rows"]),
            list(kwargs["new_successes"]),
        ),
    )

    ptc_page_results._apply_pad_order_optimization_result(
        new_records=new_records,
        new_success_dict={"WELL-A": success},
    )

    assert state["wt_records"] == new_records
    assert state["wt_records_original"] == new_records
    assert state["wt_successes"] == [success]
    assert state["wt_summary_rows"] == [{"Скважина": "WELL-A", "Статус": "ok"}]
    assert state["wt_anticollision_analysis_cache"] is ac_cache
    assert reset_calls == [True]


def test_format_duration_ru_carries_rounded_seconds() -> None:
    assert ptc_core._format_duration_ru(59.6) == "1 мин 00 с"
    assert ptc_core._format_duration_ru(119.6) == "2 мин 00 с"
    assert ptc_core._format_duration_ru(3600.0) == "1 ч 00 мин"
