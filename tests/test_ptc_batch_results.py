from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from pywp import ptc_batch_results
from pywp.coordinate_systems import CoordinateSystem
from pywp.models import Point3D, TrajectoryConfig
from pywp.welltrack_batch import SuccessfulWellPlan


def _success(
    name: str = "WELL-01",
    *,
    stations: pd.DataFrame | None = None,
) -> SuccessfulWellPlan:
    return SuccessfulWellPlan(
        name=name,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(100.0, 200.0, 1000.0),
        t3=Point3D(200.0, 400.0, 1000.0),
        stations=(
            stations
            if stations is not None
            else pd.DataFrame(
                {
                    "MD_m": [0.0],
                    "X_m": [10.0],
                    "Y_m": [20.0],
                    "Z_m": [0.0],
                    "INC_deg": [0.0],
                    "AZI_deg": [0.0],
                    "DLS_deg_per_30m": [3.0],
                }
            )
        ),
        summary={"md_total_m": 0.0},
        azimuth_deg=0.0,
        md_t1_m=0.0,
        config=TrajectoryConfig(),
    )


def test_build_batch_survey_csv_labels_transformed_geographic_coordinates() -> None:
    def fake_transform(
        stations: pd.DataFrame,
        _target_crs: CoordinateSystem,
        _source_crs: CoordinateSystem,
        *,
        rename_columns: bool = True,
    ) -> pd.DataFrame:
        transformed = stations.copy()
        transformed["X_m"] = transformed["X_m"].astype(float) + 100.0
        transformed["Y_m"] = transformed["Y_m"].astype(float) + 200.0
        return transformed

    payload = ptc_batch_results.build_batch_survey_csv(
        [_success()],
        target_crs=CoordinateSystem.WGS84,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_ZONE_16,
        transform_stations_func=fake_transform,
    )
    result = pd.read_csv(BytesIO(payload))

    assert "X_m" not in result.columns
    assert "Y_m" not in result.columns
    assert result["X_WGS_deg"].iloc[0] == pytest.approx(110.0)
    assert result["Y_WGS_deg"].iloc[0] == pytest.approx(220.0)
    assert result["PI_deg_per_10m"].iloc[0] == pytest.approx(1.0)


def test_build_batch_survey_csv_returns_empty_payload_without_station_frames() -> None:
    assert ptc_batch_results.build_batch_survey_csv([]) == b""
    assert (
        ptc_batch_results.build_batch_survey_csv([_success(stations=pd.DataFrame())])
        == b""
    )


def test_batch_summary_display_df_renames_and_orders_columns() -> None:
    source = pd.DataFrame(
        [
            {
                "Статус": "OK",
                "Скважина": "WELL-01",
                "Рестарты решателя": "1",
                "Классификация целей": "В прямом направлении",
                "Длина ГС, м": 450.0,
                "extra": "kept",
            }
        ]
    )

    display_df = ptc_batch_results.batch_summary_display_df(source)

    assert list(display_df.columns) == [
        "Скважина",
        "Цели",
        "ГС, м",
        "Рестарты",
        "Статус",
        "extra",
    ]
    assert display_df.iloc[0]["Цели"] == "В прямом направлении"


def test_batch_summary_status_counts_classifies_rows() -> None:
    counts = ptc_batch_results.batch_summary_status_counts(
        pd.DataFrame(
            [
                {"Статус": "OK", "Проблема": ""},
                {"Статус": "OK", "Проблема": pd.NA},
                {"Статус": "OK", "Проблема": "ОК"},
                {"Статус": "OK", "Проблема": "nan"},
                {"Статус": "OK", "Проблема": "warning"},
                {"Статус": "Не рассчитана", "Проблема": ""},
                {"Статус": "Ошибка", "Проблема": "bad"},
            ]
        )
    )

    assert counts.ok_count == 4
    assert counts.warning_count == 1
    assert counts.not_run_count == 1
    assert counts.error_count == 1


def test_has_md_postcheck_warning_detects_known_problem_text() -> None:
    assert ptc_batch_results.has_md_postcheck_warning(
        pd.DataFrame([{"Проблема": "Превышен лимит итоговой MD (постпроверка)."}])
    )
    assert not ptc_batch_results.has_md_postcheck_warning(
        pd.DataFrame([{"Проблема": ""}])
    )


def test_find_selected_success_matches_by_string_name() -> None:
    selected = ptc_batch_results.find_selected_success(
        selected_name="2",
        successes=[_success("1"), _success("2")],
    )

    assert selected.name == "2"
