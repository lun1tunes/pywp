---
description: Comprehensive handoff prompt for AI agent working on pywp well trajectory planner
---

# AGENT HANDOFF PROMPT — `pywp` Well Trajectory Planner

**Project path:** `/home/lun1z/pywp`
**Python version:** >=3.10
**Last verified:** 2026-05-05

---

## 1. PROJECT OVERVIEW

`pywp` is a **well trajectory planner** for the Russian oil & gas upstream sector. It computes directional drilling trajectories from 3 control points (surface `S`, entry point `t1`, target `t3`) using a **J-profile model**: VERTICAL → BUILD1 → HOLD → BUILD2 → HORIZONTAL. The solver uses `scipy.optimize.least_squares` and `scipy.optimize.differential_evolution` (DE) hybrid optimization.

The app is a **Streamlit multi-page application** (v1.54.0) with:
- `app.py` — landing page
- `pages/01_trajectory_constructor.py` — single-well PTC (Prototype Trajectory Constructor) with batch multi-well planning, anti-collision, reference well import
- `pages/02_single_well.py` — single-well planner with presets
- `pages/03_well_classification.py` — depth-based complexity classification

**UI language:** Russian (all labels, reports, diagnostics, session state keys use Russian terms).

---

## 2. DOMAIN VOCABULARY & OILFIELD TERMINOLOGY

You MUST understand and use these terms correctly:

### Trajectory Geometry
- **MD** — Measured Depth (измеренная глубина), along-hole depth in meters. Always monotonically increases.
- **TVD** — True Vertical Depth (истинная вертикальная глубина), vertical component of position. `TVD ≤ MD`.
- **INC / Inclination** (инклинация, угол склонения) — angle from vertical, `[0°, 90°]` in basic formulation. In this codebase `max_inc_deg` can go up to 95° (slightly upward horizontal).
- **AZI / Azimuth** (азимут) — horizontal direction from North, `[0°, 360°)`. MUST be normalized. Critical: crossing through North (0°/360°) requires special handling.
- **DLS** — Dogleg Severity (интенсивность искривления, ПИ), rate of change of wellbore direction, measured in `°/30m`. This is the PRIMARY curvature constraint in planning.
- **Dogleg Angle** (β, dogleg angle) — total angle change between two survey stations.
- **KOP** — Kick-Off Point (точка отрыва), where the wellbore begins to deviate from vertical. `kop_min_vertical_m` is the minimum vertical depth before KOP.
- **Build Rate (B)** — rate of inclination increase, `°/30m`.
- **Turn Rate (T)** — rate of azimuth change, `°/30m`.
- **Toolface Angle / TFO** — orientation of the dogleg plane relative to high-side or North.
- **Curve Length (CL)** — length of a curved segment in meters.

### Control Points
- **S / Surface / Wellhead** — surface location `(x, y, z=0)`.
- **t1** — Entry point into the reservoir (точка входа в пласт). Key constraint: the solver must reach this point with a specific inclination.
- **t3** — Final target point (целевая точка). The well lands at this location after the horizontal section.
- **Section Geometry** — horizontal/vertical offsets between control points: `s1_m`, `z1_m`, `ds_13_m`, `dz_13_m`.
- **t1_cross_m**, **t3_cross_m** — cross-product magnitude for coplanarity check.
- **t1_east_m**, **t1_north_m** — t1 offset components.

### Trajectory Model (J-Profile)
The planner constructs trajectories from these segments in order:
1. **VERTICAL** — straight down from surface.
2. **BUILD1** — curve from vertical to `inc_entry_deg`.
3. **HOLD** — tangent (straight inclined) section at constant inclination.
4. **BUILD2** — curve from hold inclination to horizontal (90°).
5. **HORIZONTAL** — horizontal section to target.

### Optimization & Solver
- **Objective modes:** `minimize_md`, `minimize_kop`, `anti_collision_avoidance`, `none`.
- **Turn Solver Modes:** `least_squares` (primary), `de_hybrid` (fallback for hard geometries).
- **Interpolation methods:** `rodrigues` (default, 3D rotation-based), `slerp` (spherical linear interpolation).
- **Optimization Outcome** — result metadata: status, objective value, theoretical lower bound, gap.
- **TurnSolveResult** — contains `ProfileParameters` + optimization metadata.
- **CandidateOptimizationEvaluation** — feasibility checker for solver candidates.
- **EndpointState** — `md_m`, `east_m`, `north_m`, `tvd_m`, `inc_deg`, `azi_deg`.

### Anti-Collision
- **SF** — Separation Factor (коэффициент разнесения), dimensionless. Critical safety metric.
- **Uncertainty tube / ellipse** — 3D uncertainty envelope around wellbore position.
- **Reference wells** — Actual (фактические, black) and Approved (утвержденные, red) wells used for clearance checks.
- **Separation Factor thresholds** — from Table 6.2 of industry manual.
- **Anti-collision stage** — workflow stage classification.

### Well Classification
- **Same direction** (`same_direction`) — t1 and t3 are in the same azimuth quadrant.
- **Reverse direction** (`reverse_direction`) — t1 and t3 are in opposite directions (harder geometry).
- **Complexity:** `ordinary` (обычная), `complex` (сложная), `very_complex` (очень сложная).
- **Depth Classification Rules** — `gv_m` (глубина вертикальная?), offset ranges, hold angle limits.

### Coordinate Systems (CRITICAL — recent major feature)
- **WGS84** — Global GPS standard, EPSG:4326.
- **СК-42 / Pulkovo 1942** — Legacy Soviet datum, EPSG:4284, ~3m accuracy.
- **Пулково 1995 / Pulkovo 1995** — Modernized Soviet datum, EPSG:4200, ~1m accuracy.
- **ГСК-2011 / GSK-2011** — Current Russian state system, EPSG:7681, <1m accuracy.
- **Gauss-Kruger zones** — 6-degree projected zones for Pulkovo 1942 (EPSG:28406–28420, zones 6–20, CMs 33°–75°E).
- **PNO-13 / PNO-16** — Local oilfield coordinate systems. These are **NOT authorities** — they require disambiguation:
  - If `easting > 1,000,000 m` → **ZONE**-based (uses standard Gauss-Kruger zone).
  - If `easting < 1,000,000 m` → **CM** (custom meridian).
- **False easting/northing** — shift from reference datum.
- **Central meridian** — longitude of the projection axis.
- **Datum transformation** — conversion between geodetic datums (requires `pyproj`).

### Surveying / MWD
- **MWD** — Measurement While Drilling.
- **Survey station** — discrete measurement point with `MD_m`, `INC_deg`, `AZI_deg`, `X_m`, `Y_m`, `Z_m`.
- **Minimum Curvature Method (MCM)** — standard method for calculating station positions from survey data.
- **Ratio factor (RF)** — MCM correction factor for dogleg angle.
- **Error ellipse / uncertainty** — ISCWSA-inspired first-order position uncertainty model.
- **Sigma_inc / sigma_azi** — assumed 1-sigma measurement errors.

### Eclipse / Industry Formats
- **WELLTRACK** — Eclipse reservoir simulator well trajectory format (`.INC` files).
- **WELLTRACK parser** — reads ASCII well trajectory files with encodings `utf-8`, `cp1251`, `latin-1`.
- **WelltrackRecord** — parsed row: well name, MD, INC, AZI, TVD, X, Y.
- **WelltrackPoint** — individual point with validation.

---

## 3. TECHNOLOGY STACK

| Layer | Technology | Version / Notes |
|-------|-----------|-----------------|
| Language | Python | >=3.10 |
| Data validation | Pydantic v2 | Frozen models, `model_copy(update=...)` for immutable updates |
| Numerics | NumPy, SciPy | `least_squares`, `differential_evolution`, `SLSQP`, linalg |
| Data | pandas | DataFrames for survey stations, welltracks |
| Visualization | Plotly | 2D/3D charts, go.Figure |
| UI | Streamlit | **Pinned to 1.54.0** — do NOT upgrade without testing |
| 3D viewer | Three.js | Custom component via `pywp/three_viewer.py` |
| Coordinate transforms | pyproj | Optional dependency; `HAS_PYPROJ` feature flag |
| Testing | pytest | `-m "not slow"` default; markers: `integration`, `slow` |
| Build | setuptools | `pyproject.toml` |

**CRITICAL PATTERN — Streamlit logging suppression:**
Every Streamlit page MUST configure logging BEFORE importing streamlit:
```python
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
import streamlit as st
```

---

## 4. ARCHITECTURE & MODULE MAP

### Core Package: `pywp/`

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `models.py` | 289 | Pydantic models: `Point3D`, `TrajectoryConfig`, `PlannerResult`. Literal types for optimization modes, solver modes, interpolation methods. |
| `planner_types.py` | 196 | Pure dataclasses: `SectionGeometry`, `ProfileParameters`, `PostEntrySection`, `TurnSearchSettings`, `OptimizationOutcome`, `TurnSolveResult`, `CandidateOptimizationEvaluation`, `EndpointState`. `PlanningError` exception. |
| `planner.py` | ~3700 | Main solver. J-profile + TURN optimization. scipy least_squares + DE hybrid. |
| `planner_geometry.py` | ~170 | Azimuth math, section geometry calculations, DLS computations. |
| `planner_config.py` | ~130 | Config helpers, optimization mode labels. |
| `planner_validation.py` | ~650 | Pre-flight validation of trajectory constraints. |
| `segments.py` | ~225 | Abstract `Segment` class + `VerticalSegment`, `BuildSegment`, `HoldSegment`, `HorizontalSegment`. Interpolation methods: `rodrigues` (default), `slerp`. |
| `mcm.py` | ~160 | Minimum Curvature Method: `dogleg_angle_rad`, `ratio_factor`, `wrap_azimuth_deg`. |
| `trajectory.py` | ~45 | Trajectory container type. |
| `classification.py` | ~290 | Depth-based well complexity classification. |

### Coordinate Systems (RECENT — fully production-ready)

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `coordinate_systems.py` | 490 | `CoordinateSystem` enum (EPSG codes), `LocalCoordinateSystem` dataclass, `CoordinateTransformer` (pyproj wrapper), `LocalCoordinateTransformer` (false easting/northing + rotation + scale), disambiguation: `disambiguate_pno13/16`, `define_pno_13/16_system`, `get_pulkovo_zone`. |
| `coordinate_integration.py` | ~540 | Streamlit UI integration: `render_crs_sidebar()`, `get_selected_crs()`, `should_auto_convert()`, `transform_point_to_crs()`, `transform_stations_to_crs()`, `apply_crs_to_well_view()`, `format_coordinates_for_display()`, `get_crs_display_suffix()`. **Default CRS = PNO_16_ZONE**. |
| `geo_point.py` | ~80 | `GeoPoint` (extends Point3D with CRS), `KnownCRSs` convenience class. |

### Anti-Collision & Uncertainty

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `anticollision.py` | ~1780 | `AntiCollisionSample`, `AntiCollisionWell`, separation factor computation, clearance checks, corridor analysis. |
| `anticollision_optimization.py` | ~730 | Anti-collision avoidance optimizer (reruns with adjusted targets). |
| `anticollision_recommendations.py` | ~1120 | Automated rerun recommendations. |
| `anticollision_rerun.py` | ~990 | Rerun orchestration. |
| `anticollision_stage.py` | ~30 | Stage constants. |
| `uncertainty.py` | ~1025 | `PlanningUncertaintyModel` (first-order ISCWSA-inspired), `build_uncertainty_overlay()`, `UncertaintyTubeMesh`, `WellUncertaintyOverlay`, `build_uncertainty_tube_mesh()`. |

### PTC Core & UI

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `ptc_core.py` | ~9000 | PTC page core logic. ALL functions are pure Python (no `run_page()`). Handles: WELLTRACK import, pad detection, batch planning, 3D viz, anti-collision analysis, reference well management, solver diagnostics. |
| `solver_diagnostics.py` | ~750 | `DiagnosticItem`, `ParsedSolverError`, error message parsing with regex. |
| `solver_diagnostics_ui.py` | ~40 | UI wrapper for diagnostics. |
| `ui_well_result.py` | ~580 | `SingleWellResultView` Pydantic model, `render_key_metrics()`, `render_result_plots()`, `render_result_tables()`. |
| `ui_well_panels.py` | ~180 | `render_plan_section_panel()`, `render_survey_table_with_download()`, `render_trajectory_dls_panel()`, `render_run_log_panel()`. |
| `ui_calc_params.py` | ~560 | Calculation parameter UI panels. |
| `ui_theme.py` | ~120 | CSS theming, `apply_page_style()`, `render_hero()`. |
| `ui_utils.py` | ~45 | `arrow_safe_text_dataframe()`, `dls_to_pi()`, `format_distance()`. |
| `visualization.py` | ~1230 | Plotly figure builders for 2D/3D trajectory plots. |
| `plotly_config.py` | ~25 | Plotly global config. |
| `plot_axes.py` | ~90 | Axis label helpers. |

### Welltrack & Batch

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `eclipse_welltrack.py` | ~400 | WELLTRACK parser for Eclipse `.INC` files. Encodings: `utf-8`, `cp1251`, `latin-1`. |
| `welltrack_batch.py` | ~1700 | Batch planning with `SuccessfulWellPlan`, `BatchPlannerResult`. Multiprocessing support. |
| `welltrack_quality.py` | ~80 | Welltrack data quality checks. |
| `well_pad.py` | ~280 | Pad detection and surface layout grouping. |
| `pad_optimization.py` | ~500 | Pad-level optimization. |
| `reference_trajectories.py` | ~500 | Reference trajectory helpers. |
| `actual_fund_analysis.py` | ~1200 | Actual vs planned fund depth analysis. |
| `analytical_precheck.py` | ~280 | Fast analytical feasibility pre-check for trajectories. |

### Infrastructure

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `__init__.py` | ~80 | Package exports. **Two separate try blocks**: `HAS_COORDINATE_SYSTEMS` (pyproj only) and `HAS_COORDINATE_INTEGRATION` (pyproj + streamlit). |
| `pydantic_base.py` | ~50 | `FrozenModel`, `FrozenArbitraryModel`, `coerce_model_like()`. |
| `constants.py` | ~12 | `DEG2RAD`, `RAD2DEG`, `SMALL = 1e-9`. |
| `three_viewer.py` | ~60 | Streamlit custom component wrapper for Three.js 3D viewer. |

---

## 5. FILE STRUCTURE

```
/home/lun1z/pywp/
├── app.py                          # Streamlit landing page
├── pyproject.toml                  # Package config (setuptools)
├── pytest.ini                      # Test config: -m "not slow" default
├── requirements.txt                # Runtime deps
├── requirements-dev.txt            # pytest, pytest-xdist
├── README.md                       # Russian README
├── manual_well_planning.md         # AI agent ontology reference (Russian)
├── streamlit-docs.md               # Streamlit API docs reference
├── trajectory_calculation_guide.pdf       # Industry manual (English)
├── trajectory_calculation_guide_detailed.pdf
├── Преобразования координат между Pulkovo ... .pdf  # CRS research (Russian)
│
├── pywp/                           # Core package
│   ├── __init__.py
│   ├── models.py
│   ├── planner.py
│   ├── planner_types.py
│   ├── planner_geometry.py
│   ├── planner_config.py
│   ├── planner_validation.py
│   ├── segments.py
│   ├── mcm.py
│   ├── trajectory.py
│   ├── classification.py
│   ├── constants.py
│   ├── pydantic_base.py
│   ├── coordinate_systems.py       # CRS enum, transformers, disambiguation
│   ├── coordinate_integration.py     # Streamlit CRS sidebar + transforms
│   ├── geo_point.py
│   ├── ptc_core.py                 # ~9K lines, PTC page pure functions
│   ├── solver_diagnostics.py
│   ├── solver_diagnostics_ui.py
│   ├── ui_well_result.py           # SingleWellResultView + renderers
│   ├── ui_well_panels.py
│   ├── ui_calc_params.py
│   ├── ui_theme.py
│   ├── ui_utils.py
│   ├── visualization.py
│   ├── plotly_config.py
│   ├── plot_axes.py
│   ├── anticollision.py
│   ├── anticollision_optimization.py
│   ├── anticollision_recommendations.py
│   ├── anticollision_rerun.py
│   ├── anticollision_stage.py
│   ├── uncertainty.py
│   ├── eclipse_welltrack.py
│   ├── welltrack_batch.py
│   ├── welltrack_quality.py
│   ├── well_pad.py
│   ├── pad_optimization.py
│   ├── reference_trajectories.py
│   ├── actual_fund_analysis.py
│   ├── analytical_precheck.py
│   ├── three_viewer.py
│   └── three_viewer_assets/        # Three.js component assets
│
├── pages/                          # Streamlit multi-page app
│   ├── 01_trajectory_constructor.py    # PTC: batch planner (uses ptc_core)
│   ├── 02_single_well.py
│   └── 03_well_classification.py
│
├── tests/                          # ~40 test files, ~400 tests total
│   ├── test_data/                  # WELLTRACKS4.INC, sample data
│   ├── test_coordinate_systems.py
│   ├── test_coordinate_integration.py
│   ├── test_planner.py
│   ├── test_welltrack_batch.py
│   ├── test_anticollision.py
│   ├── test_uncertainty.py
│   ├── test_visualization.py
│   ├── test_ptc_core.py (test_welltrack_import_page.py — renamed legacy)
│   └── ... (see full list in tests/)
│
├── docs/
│   ├── coordinate_systems.md       # CRS module documentation
│   └── coordinate_integration.md   # CRS UI integration documentation
│
└── scripts/
    ├── check_streamlit_pages.py
    ├── generate_reference_stress_welltracks.py
    └── run_tests.py
```

---

## 6. CRITICAL ARCHITECTURAL DECISIONS

### 6.1 Immutable Pydantic v2 Models
- All domain models extend `FrozenModel` or `FrozenArbitraryModel`.
- Use `model_copy(update={...})` for mutations. Never mutate in place.
- `coerce_model_like()` helper converts dicts/Mappings to Pydantic models before validation.

### 6.2 PTC Core Architecture
- `ptc_core.py` is a **pure Python module** with NO `run_page()` function.
- `pages/01_trajectory_constructor.py` is the Streamlit page that imports `pywp.ptc_core as wt` and calls functions.
- This separation exists because Streamlit's module reloads break pickling for multiprocessing.

### 6.3 Multiprocessing Pattern
- Batch planning uses `multiprocessing.Pool`.
- Worker entry-point `_evaluate_record_from_dicts()` accepts plain dicts (not Pydantic models) to avoid `PicklingError` from Streamlit module reloads.
- Inside the worker, dicts are converted back to models and `_evaluate_record_standalone()` is called.

### 6.4 Coordinate System Integration
- **Default CRS:** `CoordinateSystem.PNO_16_ZONE` (user requirement).
- **Two feature flags:** `HAS_COORDINATE_SYSTEMS` (pyproj available) and `HAS_COORDINATE_INTEGRATION` (pyproj + streamlit available).
- `pywp/__init__.py` has **two separate `try/except` blocks** so that missing Streamlit does NOT disable core coordinate system functionality.
- `apply_crs_to_well_view()` transforms all coordinates (surface, t1, t3, stations) while **preserving Z (TVD)**.
- PNO-13/PNO-16 are **placeholder systems** — they do NOT have EPSG codes. Direct pyproj transformation is blocked for them via `_can_transform_directly()`. Actual field-specific parameters (false easting, central meridian) must be configured per-project.
- `_transform_xy()` handles all four combinations: geo→geo, projected→projected, geo→projected, projected→geo.
- `format_coordinates_for_display()` uses **locale-independent** space-thousands separator for Russian UI.

### 6.5 Session State Management
- Session state keys use **Russian labels** for UI state and **English prefixes** for internal state.
- Key prefixes: `wt_` (welltrack/PTC), `ptc_` (PTC-specific), `trajectory_crs_` (coordinate system).
- Critical session state keys:
  - `wt_records`, `wt_records_original` — imported welltrack records
  - `wt_successes`, `wt_summary_rows` — batch planning results
  - `wt_selected_names` — selected wells for calculation
  - `wt_reference_wells`, `wt_reference_actual_wells`, `wt_reference_approved_wells`
  - `trajectory_crs_selected`, `trajectory_crs_selectbox`, `trajectory_crs_auto_convert`
  - `wt_3d_render_mode`, `wt_3d_backend`
  - `wt_results_all_view_mode` — "Траектории" / "Anti-collision"

### 6.6 Testing Strategy
- **Default run:** `pytest -q` (excludes `@slow` tests).
- **Markers:** `integration` (e2e with real planner), `slow` (expensive geometries).
- **Test profiles:** `unit`, `fast`, `integration`, `slow`, `full` via `scripts/run_tests.py`.
- **Known flaky tests:** `well_02` from `WELLTRACKS3.INC` has solver failures (expected).
- **PTC page tests** use `AppTest.from_file("pages/01_trajectory_constructor.py")` (Streamlit testing framework).

### 6.7 WELLTRACK Import
- Parses Eclipse `.INC` files with regex.
- Supports aliases: "s"→wellhead, "entry"→t1, "target"→t3.
- Point order: wellhead → t1 → t3.
- Encoding fallback chain: `utf-8` → `cp1251` → `latin-1`.

---

## 7. RECENT CHANGES & KNOWN ISSUES

### Recently Completed (May 2026)
1. **Full coordinate system integration** — Pulkovo 1942/1995, GSK-2011, WGS84, PNO-13/16 with disambiguation, UI sidebar, automatic transformation in reports/graphs/inclinometry.
2. **Production-ready `coordinate_integration.py`** — actual pyproj-based transformations, graceful fallbacks, Russian diagnostic messages, locale-independent formatting.
3. **Split `__init__.py` try blocks** — coordinate systems and coordinate integration are independent.
4. **All tests pass** — 397 passed, 2 skipped (known legacy UI differences).

### Known Limitations
- **PNO-13/PNO-16 transformations** require field-specific parameter configuration (false easting, central meridian). Current implementation returns original coordinates with warning for these placeholder systems.
- **pyproj is optional** — without it, coordinates display with renamed columns but no actual datum transformation.
- **Streamlit pinned to 1.54.0** — upgrading may break `AppTest` or session state behavior.
- **Two skipped tests** in `test_welltrack_import_page.py` — relate to legacy UI widgets from removed `02_welltrack_import.py` page.

---

## 8. INVESTIGATION INSTRUCTIONS FOR NEW AGENT

When you start working on this project, follow this order:

### Step 1: Verify Environment
```bash
cd /home/lun1z/pywp
source .venv/bin/activate
python -c "import pywp; print('OK:', pywp.__all__)"
python -c "from pywp.coordinate_systems import HAS_PYPROJ; print('HAS_PYPROJ:', HAS_PYPROJ)"
python -c "from pywp import HAS_COORDINATE_SYSTEMS, HAS_COORDINATE_INTEGRATION; print(HAS_COORDINATE_SYSTEMS, HAS_COORDINATE_INTEGRATION)"
pytest tests/test_coordinate_integration.py -q
pytest tests/test_coordinate_systems.py -q
```

### Step 2: Understand the Domain Model
Read in this order:
1. `pywp/models.py` — `Point3D`, `TrajectoryConfig`, literals
2. `pywp/planner_types.py` — `SectionGeometry`, `ProfileParameters`, `EndpointState`
3. `pywp/segments.py` — segment types and interpolation
4. `pywp/mcm.py` — minimum curvature math
5. `pywp/classification.py` — complexity rules
6. `manual_well_planning.md` — full domain ontology

### Step 3: Understand Coordinate Systems (if working on CRS features)
1. `pywp/coordinate_systems.py` — enum, transformers, disambiguation
2. `pywp/coordinate_integration.py` — UI integration, transformation functions
3. `docs/coordinate_systems.md` and `docs/coordinate_integration.md`
4. `tests/test_coordinate_systems.py`, `tests/test_coordinate_integration.py`
5. Research PDF: `Преобразования координат между Pulkovo ... .pdf`

### Step 4: Understand PTC / Batch Flow (if working on UI or batch)
1. `pages/01_trajectory_constructor.py` — page structure, calls `render_crs_sidebar()`
2. `pywp/ptc_core.py` — search for functions referenced in the page
3. `pywp/ui_well_result.py` — `SingleWellResultView`, rendering functions
4. `pywp/welltrack_batch.py` — batch planning flow
5. `pywp/anticollision.py` — separation factor logic

### Step 5: Run Relevant Tests Before Changes
```bash
# Coordinate systems
pytest tests/test_coordinate_systems.py tests/test_coordinate_integration.py -v --tb=short

# Planner
pytest tests/test_planner.py -v --tb=short -m "not slow"

# Full suite (takes ~3 minutes)
pytest tests/ -q
```

### Step 6: Code Search Strategy
- Use `grep_search` or `code_search` tool to find functions by name.
- Session state keys: search for `st.session_state[` in `ptc_core.py` and `pages/`.
- UI strings: most labels are in Russian — search with both Russian and English terms.
- Pydantic validators: search for `@field_validator` and `@model_validator`.

---

## 9. CODING STANDARDS

- **Type hints:** All public functions must have full type annotations.
- **Frozen models:** Never mutate Pydantic models in place. Always use `model_copy(update=...)`.
- **Imports:** `from __future__ import annotations` at top. No relative imports in `pywp/`.
- **Constants:** UPPER_SNAKE_CASE in `constants.py` or module-level.
- **Russian UI:** All user-facing strings in Streamlit are Russian. Internal/code strings are English.
- **Error messages:** Use Russian for user-facing diagnostics in Streamlit; English for internal exceptions.
- **`__all__` exports:** Every module defines `__all__` explicitly.
- **Docstrings:** Google-style or plain descriptive. Include Args/Returns for public APIs.
- **Testing:** Add tests for new features. Run `pytest` before committing. Do NOT delete or weaken existing tests.

---

## 10. KEY CONTACT POINTS

- **Package root:** `pywp/__init__.py` — check exports and feature flags first.
- **Models:** `pywp/models.py` — `Point3D`, `TrajectoryConfig`, all literal types.
- **Planner core:** `pywp/planner.py` — `TrajectoryPlanner` class.
- **PTC core:** `pywp/ptc_core.py` — all batch/UI pure functions.
- **CRS:** `pywp/coordinate_systems.py` + `pywp/coordinate_integration.py`.
- **Tests:** `tests/test_coordinate_integration.py` — good example of test patterns.

---

*End of handoff prompt. Good luck, agent. Verify everything before changing it.*
