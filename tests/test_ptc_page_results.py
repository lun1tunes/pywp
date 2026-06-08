from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
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


def test_anticollision_panel_requires_explicit_run_before_first_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {"buttons": []}

    class FakeStreamlit:
        session_state: dict[str, object] = {}

        def info(self, message: str) -> None:
            calls["info"] = str(message)

        def button(self, label: str, **_kwargs: object) -> bool:
            calls["buttons"].append(str(label))
            return False

        def selectbox(
            self,
            _label: str,
            *,
            options: list[str],
            format_func=None,
            key: str,
        ) -> str:
            value = str(options[0])
            self.session_state[key] = value
            return value

        def markdown(self, _message: str) -> None:
            pass

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_pending_edit_target_names",
        lambda: [],
    )
    monkeypatch.setattr(
        ptc_page_results.reference_state,
        "reference_wells_from_state",
        lambda: (),
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_current_anti_collision_cache_snapshot",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_cached_anti_collision_view_model",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("anti-collision must wait for explicit launch")
        ),
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_render_status_run_log",
        lambda **kwargs: calls.setdefault("log_kwargs", kwargs),
    )

    ptc_page_results._render_anticollision_panel(
        successes=[object(), object()],
        records=[],
        summary_rows=[],
        focus_pad_id="",
        focus_pad_well_names=[],
    )

    assert calls["buttons"] == ["Расчёт пересечений"]
    assert "Запустите anti-collision отдельным шагом" in str(calls["info"])


def test_anticollision_panel_reruns_after_fresh_analysis_when_visual_is_external(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=0,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )
    calls: dict[str, object] = {}

    class FakeStreamlit:
        session_state: dict[str, object] = {}

        def button(self, label: str, **_kwargs: object) -> bool:
            calls.setdefault("buttons", []).append(str(label))
            return str(label) == "Расчёт пересечений"

        def selectbox(
            self,
            _label: str,
            *,
            options: list[str],
            format_func=None,
            key: str,
        ) -> str:
            value = str(options[0])
            self.session_state[key] = value
            return value

        def markdown(self, _message: str) -> None:
            pass

        def progress(self, _value: int, *, text: str):
            calls["progress_text"] = str(text)

            class _Progress:
                def progress(self, value: int, *, text: str) -> None:
                    calls["progress_update"] = (int(value), str(text))

                def empty(self) -> None:
                    calls["progress_emptied"] = True

            return _Progress()

        def rerun(self) -> None:
            raise _RerunRequested

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_pending_edit_target_names",
        lambda: [],
    )
    monkeypatch.setattr(
        ptc_page_results.reference_state,
        "reference_wells_from_state",
        lambda: (),
    )
    monkeypatch.setattr(
        ptc_page_results.ptc_anticollision_params,
        "reference_uncertainty_models_from_state",
        lambda _reference_wells: {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_current_anti_collision_cache_snapshot",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_cached_anti_collision_view_model",
        lambda **_kwargs: (analysis, (), ()),
    )

    with pytest.raises(_RerunRequested):
        ptc_page_results._render_anticollision_panel(
            successes=[object(), object()],
            records=[],
            summary_rows=[],
            focus_pad_id="",
            focus_pad_well_names=[],
            show_visualization=False,
        )

    assert calls["buttons"] == ["Расчёт пересечений"]
    assert calls["progress_emptied"] is True


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
    old_edited_well = SimpleNamespace(name="WELL-A", is_reference_only=False)
    kept_well = SimpleNamespace(name="WELL-B", is_reference_only=False)
    analysis = AntiCollisionAnalysis(
        wells=(old_edited_well, kept_well),
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

    class _DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

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

        def markdown(self, message: str) -> None:
            calls.setdefault("markdown", []).append(str(message))

        def container(self) -> _DummyContainer:
            return _DummyContainer()

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
    def fake_anticollision_payload(analysis: object, *args: object, **kwargs: object):
        calls["payload_analysis"] = analysis
        calls["payload_kwargs"] = kwargs
        return {}

    monkeypatch.setattr(
        ptc_page_results.wt,
        "_all_wells_anticollision_three_payload",
        fake_anticollision_payload,
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

    ptc_page_results._render_anticollision_panel(
        successes=[],
        records=[],
        summary_rows=[],
        focus_pad_id="",
        focus_pad_well_names=[],
    )

    assert "приостановлен" in str(calls["info"])
    assert any("последний anti-collision снимок" in item for item in calls["captions"])
    rendered_analysis = calls["payload_analysis"]
    assert [str(well.name) for well in rendered_analysis.wells] == ["WELL-B"]
    assert calls["payload_kwargs"]["target_only_wells"] == [target_only]
    assert calls["payload_kwargs"]["show_sidetrack_relative_cones"] is False
    assert "plotly" not in calls
    assert calls["override_kwargs"]["target_only_wells"] == [target_only]
    assert calls["override_kwargs"]["target_only_name_to_color"] == {"WELL-A": "#123456"}


def test_cached_snapshot_filter_removes_pending_well_geometry_and_conflicts() -> None:
    edited_well = SimpleNamespace(name="well_08", is_reference_only=False, samples=())
    kept_well = SimpleNamespace(name="well_09", is_reference_only=False, samples=())
    reference_well = SimpleNamespace(name="fact_01", is_reference_only=True, samples=())
    stale_corridor = SimpleNamespace(
        well_a="well_08",
        well_b="well_09",
        md_a_start_m=100.0,
        md_a_end_m=200.0,
        md_b_start_m=110.0,
        md_b_end_m=210.0,
        classification="trajectory",
        priority_rank=2,
    )
    kept_corridor = SimpleNamespace(
        well_a="well_09",
        well_b="fact_01",
        md_a_start_m=300.0,
        md_a_end_m=360.0,
        md_b_start_m=310.0,
        md_b_end_m=370.0,
        classification="target-trajectory",
        priority_rank=1,
    )
    stale_zone = SimpleNamespace(
        well_a="well_08",
        well_b="well_09",
        separation_factor=0.4,
    )
    kept_zone = SimpleNamespace(
        well_a="well_09",
        well_b="fact_01",
        separation_factor=0.8,
    )
    analysis = AntiCollisionAnalysis(
        wells=(edited_well, kept_well, reference_well),
        corridors=(stale_corridor, kept_corridor),
        well_segments=(
            SimpleNamespace(well_name="well_08"),
            SimpleNamespace(well_name="well_09"),
        ),
        zones=(stale_zone, kept_zone),
        pair_count=3,
        overlapping_pair_count=2,
        target_overlap_pair_count=1,
        worst_separation_factor=0.4,
    )
    stale_recommendation = SimpleNamespace(
        well_a="well_08",
        well_b="well_09",
        affected_wells=("well_08",),
    )
    kept_recommendation = SimpleNamespace(
        well_a="well_09",
        well_b="fact_01",
        affected_wells=("well_09",),
    )
    stale_cluster = SimpleNamespace(
        well_names=("well_08", "well_09"),
        affected_wells=("well_08",),
    )
    kept_cluster = SimpleNamespace(
        well_names=("well_09", "fact_01"),
        affected_wells=("well_09",),
    )

    filtered_analysis, filtered_recommendations, filtered_clusters = (
        ptc_page_results._filter_cached_anticollision_snapshot_for_pending_edits(
            analysis=analysis,
            recommendations=(stale_recommendation, kept_recommendation),
            clusters=(stale_cluster, kept_cluster),
            pending_edit_names=["well_08"],
        )
    )

    assert [str(well.name) for well in filtered_analysis.wells] == [
        "well_09",
        "fact_01",
    ]
    assert filtered_analysis.corridors == (kept_corridor,)
    assert filtered_analysis.zones == (kept_zone,)
    assert {segment.well_name for segment in filtered_analysis.well_segments} == {
        "well_09",
        "fact_01",
    }
    assert filtered_analysis.pair_count == 1
    assert filtered_analysis.overlapping_pair_count == 1
    assert filtered_analysis.target_overlap_pair_count == 1
    assert filtered_analysis.worst_separation_factor == pytest.approx(0.8)
    assert filtered_recommendations == (kept_recommendation,)
    assert filtered_clusters == (kept_cluster,)


def test_target_edit_overview_keeps_reference_wells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_only = SimpleNamespace(name="WELL-A")
    reference_wells = (object(),)
    calls: dict[str, object] = {}

    class _DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeStreamlit:
        def info(self, message: str) -> None:
            calls["info"] = str(message)

        def markdown(self, message: str) -> None:
            calls["markdown"] = str(message)

        def caption(self, message: str) -> None:
            calls["caption"] = str(message)

        def container(self) -> _DummyContainer:
            return _DummyContainer()

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
    assert "plan_kwargs" not in calls


def test_render_success_tabs_shows_trajectory_overview_before_anticollision_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"overview": 0, "anticollision": 0}

    class FakeStreamlit:
        session_state = {
            "wt_results_view_mode": "Все скважины",
            "wt_anticollision_uncertainty_preset": ptc_core.DEFAULT_UNCERTAINTY_PRESET,
        }

        def radio(self, *_args: object, **_kwargs: object) -> str:
            return "Все скважины"

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(ptc_page_results.wt, "_well_color_map", lambda _records: {})
    monkeypatch.setattr(ptc_page_results.wt, "_pad_membership", lambda _records: ([], {}, {}))
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_normalize_focus_pad_id",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_focus_pad_well_names",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results.reference_state,
        "reference_wells_from_state",
        lambda: (),
    )
    monkeypatch.setattr(
        ptc_page_results.ptc_anticollision_params,
        "reference_uncertainty_models_from_state",
        lambda _reference_wells: {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_current_anti_collision_cache_snapshot",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_failed_target_only_wells",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_render_target_edit_overview",
        lambda **_kwargs: calls.__setitem__("overview", calls["overview"] + 1) or True,
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_render_anticollision_panel",
        lambda **_kwargs: calls.__setitem__(
            "anticollision", calls["anticollision"] + 1
        ),
    )

    ptc_page_results.render_success_tabs(
        successes=[SimpleNamespace(name="WELL-A")],
        records=[],
        summary_rows=[],
    )

    assert calls == {"overview": 1, "anticollision": 1}


def test_render_success_tabs_skips_trajectory_overview_when_current_ac_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"overview": 0, "anticollision": 0, "visual": 0}
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=0,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )

    class FakeStreamlit:
        session_state = {
            "wt_results_view_mode": "Все скважины",
            "wt_anticollision_uncertainty_preset": ptc_core.DEFAULT_UNCERTAINTY_PRESET,
        }

        def radio(self, *_args: object, **_kwargs: object) -> str:
            return "Все скважины"

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(ptc_page_results.wt, "_well_color_map", lambda _records: {})
    monkeypatch.setattr(ptc_page_results.wt, "_pad_membership", lambda _records: ([], {}, {}))
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_normalize_focus_pad_id",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_focus_pad_well_names",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results.reference_state,
        "reference_wells_from_state",
        lambda: (),
    )
    monkeypatch.setattr(
        ptc_page_results.ptc_anticollision_params,
        "reference_uncertainty_models_from_state",
        lambda _reference_wells: {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_current_anti_collision_cache_snapshot",
        lambda **_kwargs: (analysis, (), ()),
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_failed_target_only_wells",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_pending_edit_target_names",
        lambda: [],
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_render_target_edit_overview",
        lambda **_kwargs: calls.__setitem__("overview", calls["overview"] + 1) or True,
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_render_anticollision_visual_overview",
        lambda **_kwargs: calls.__setitem__("visual", calls["visual"] + 1),
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_render_anticollision_panel",
        lambda **_kwargs: calls.__setitem__(
            "anticollision", calls["anticollision"] + 1
        ),
    )

    ptc_page_results.render_success_tabs(
        successes=[SimpleNamespace(name="WELL-A")],
        records=[],
        summary_rows=[],
    )

    assert calls == {"overview": 0, "anticollision": 1, "visual": 1}


def test_render_success_tabs_keeps_report_panel_when_ac_visual_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"panel": 0, "failures": 0, "errors": []}
    analysis = AntiCollisionAnalysis(
        wells=(),
        corridors=(),
        well_segments=(),
        zones=(),
        pair_count=0,
        overlapping_pair_count=0,
        target_overlap_pair_count=0,
        worst_separation_factor=None,
    )

    class FakeStreamlit:
        session_state = {
            "wt_results_view_mode": "Все скважины",
            "wt_anticollision_uncertainty_preset": ptc_core.DEFAULT_UNCERTAINTY_PRESET,
        }

        def radio(self, *_args: object, **_kwargs: object) -> str:
            return "Все скважины"

        def error(self, message: str) -> None:
            calls["errors"].append(str(message))

        def caption(self, _message: str) -> None:
            return None

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(ptc_page_results.wt, "_well_color_map", lambda _records: {})
    monkeypatch.setattr(ptc_page_results.wt, "_pad_membership", lambda _records: ([], {}, {}))
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_normalize_focus_pad_id",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_focus_pad_well_names",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results.reference_state,
        "reference_wells_from_state",
        lambda: (),
    )
    monkeypatch.setattr(
        ptc_page_results.ptc_anticollision_params,
        "reference_uncertainty_models_from_state",
        lambda _reference_wells: {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_current_anti_collision_cache_snapshot",
        lambda **_kwargs: (analysis, (), ()),
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_failed_target_only_wells",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_pending_edit_target_names",
        lambda: [],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_clusters_touching_focus_pad",
        lambda **kwargs: kwargs["clusters"],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_anticollision_focus_well_names",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_store_anticollision_failure_state",
        lambda _exc: calls.__setitem__("failures", calls["failures"] + 1),
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_render_anticollision_visual_overview",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("visual boom")),
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_render_anticollision_panel",
        lambda **_kwargs: calls.__setitem__("panel", calls["panel"] + 1),
    )

    ptc_page_results.render_success_tabs(
        successes=[SimpleNamespace(name="WELL-A")],
        records=[],
        summary_rows=[],
    )

    assert calls["failures"] == 1
    assert calls["panel"] == 1
    assert any("Не удалось отрисовать anti-collision визуализацию" in item for item in calls["errors"])


def test_target_edit_overview_uses_fast_3d_payload_before_anticollision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def plotly_chart(self, *_args: object, **_kwargs: object) -> None:
            return None

    class FakeStreamlit:
        session_state: dict[str, object] = {}

        def markdown(self, *_args: object, **_kwargs: object) -> None:
            return None

        def caption(self, *_args: object, **_kwargs: object) -> None:
            return None

        def info(self, *_args: object, **_kwargs: object) -> None:
            return None

        def selectbox(self, *_args: object, **_kwargs: object) -> str:
            return ""

        def container(self) -> _DummyContext:
            return _DummyContext()

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_failed_target_only_wells",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_well_color_map",
        lambda _records: {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_well_label_display_names",
        lambda _records: {},
    )
    monkeypatch.setattr(
        ptc_page_results.reference_state,
        "reference_wells_from_state",
        lambda: (),
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_pilot_study_points_by_name",
        lambda _records: {},
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_all_wells_three_payload",
        lambda _successes, **kwargs: (
            captured.__setitem__("render_mode", kwargs["render_mode"]),
            {
                "lines": [],
                "points": [],
                "labels": [],
                "legend": [],
                "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
            },
        )[1],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_render_three_payload",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_trajectory_three_payload_overrides",
        lambda **_kwargs: {},
    )
    rendered = ptc_page_results._render_target_edit_overview(
        successes=[SimpleNamespace(name="WELL-A")],
        records=[],
        summary_rows=[],
        title="### Overview",
        empty_message="empty",
        focus_pad_well_names=[],
        show_focus_selector=False,
    )

    assert rendered is True
    assert captured["render_mode"] == ptc_core.WT_3D_RENDER_FAST


def test_render_success_tabs_hides_plotly_panels_for_single_well_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    success = SimpleNamespace(
        name="WELL-A",
        surface=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        t1=SimpleNamespace(x=100.0, y=0.0, z=1000.0),
        t3=SimpleNamespace(x=200.0, y=0.0, z=1100.0),
        target_pairs=(),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 1000.0],
                "X_m": [0.0, 200.0],
                "Y_m": [0.0, 0.0],
                "Z_m": [0.0, 1100.0],
                "INC_deg": [0.0, 90.0],
                "AZI_deg": [0.0, 90.0],
                "DLS_deg_per_30m": [0.0, 0.0],
                "segment": ["VERTICAL", "HORIZONTAL"],
            }
        ),
        summary={},
        config=SimpleNamespace(dls_limits_deg_per_30m={}),
        azimuth_deg=90.0,
        md_t1_m=1000.0,
        runtime_s=1.0,
        md_postcheck_message="",
        md_postcheck_exceeded=False,
    )

    class FakeStreamlit:
        session_state = {"wt_results_view_mode": "Отдельная скважина"}

        def radio(self, *_args: object, **_kwargs: object) -> str:
            return "Отдельная скважина"

        def selectbox(self, _label: str, *, options: list[str]) -> str:
            assert options == ["WELL-A"]
            return "WELL-A"

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(ptc_page_results, "get_input_crs", lambda: "input-crs")
    monkeypatch.setattr(ptc_page_results, "get_selected_crs", lambda: "selected-crs")
    monkeypatch.setattr(ptc_page_results, "should_auto_convert", lambda: False)
    monkeypatch.setattr(ptc_page_results, "csv_export_crs", lambda *_args, **_kwargs: "input-crs")
    monkeypatch.setattr(ptc_page_results.wt, "_well_color_map", lambda _records: {})
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_find_selected_success",
        lambda **_kwargs: success,
    )
    monkeypatch.setattr(
        ptc_page_results,
        "_single_well_pilot_context",
        lambda **_kwargs: (None, None, ()),
    )
    monkeypatch.setattr(
        ptc_page_results,
        "render_key_metrics",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        ptc_page_results,
        "render_result_plots",
        lambda **kwargs: calls.setdefault("plots_kwargs", kwargs),
    )
    monkeypatch.setattr(
        ptc_page_results,
        "render_result_tables",
        lambda **_kwargs: None,
    )

    ptc_page_results.render_success_tabs(
        successes=[success],
        records=[],
        summary_rows=[],
    )

    assert calls["plots_kwargs"]["show_plotly_panels"] is False


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
