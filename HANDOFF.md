# HANDOFF: pywp / PTC

## 1. Что это за проект

`pywp` — это инженерный планировщик траекторий скважин на Python со Streamlit-интерфейсом. Репозиторий уже давно вышел за рамки “одного солвера”: сейчас это полноценный рабочий инструмент со следующими слоями:

- batch-конструктор траекторий с импортом целей, редактированием, пересчётом и anti-collision;
- sandbox для расчёта одной скважины;
- классификация траекторий по правилам сложности;
- пересчёт координат между CRS;
- импорт проектных и фактических референсов;
- анализ фактического фонда;
- 3D-визуализация и 3D-редактирование целей;
- подготовка рекомендаций и повторных прогонов для разведения конфликтов.

Технологический стек:

- Python `>=3.10`
- Streamlit `1.54.0`
- NumPy / Pandas / SciPy
- Pydantic v2
- Plotly
- PyProj
- кастомный локальный 3D-компонент на Three.js

Важно: это не backend + frontend в классическом смысле. Основная бизнес-логика живёт в Python-процессе Streamlit, а состояние UI держится в `st.session_state`.

## 2. Быстрая ориентация по репозиторию

На текущий момент в проекте:

- `72` Python-модуля в `pywp/`
- `56` test-модулей в `tests/`

Ключевые директории и файлы:

| Путь | Назначение |
| --- | --- |
| `app.py` | минимальная главная страница Streamlit |
| `pages/01_trajectory_constructor.py` | главный batch/PTC UI |
| `pages/02_single_well.py` | расчёт одной скважины |
| `pages/03_well_classification.py` | классификация по ГВ/отходу/hold |
| `pages/04_crs_calculator.py` | калькулятор CRS |
| `pywp/planner.py` | главное расчётное ядро траектории |
| `pywp/welltrack_batch.py` | batch-оркестратор по нескольким скважинам |
| `pywp/anticollision.py` | расчёт антиколлизии и пересечений |
| `pywp/anticollision_rerun.py` | инкрементальный anti-collision, prepared overrides, cluster rerun |
| `pywp/pilot_wells.py` | логика пилотов, ЗБС, окон зарезки |
| `pywp/multi_horizontal.py` | multi-target / multi-horizontal extension |
| `pywp/ptc_core.py` | огромный центральный UI/orchestration-модуль PTC |
| `pywp/ptc_page.py` | тонкая сборка главной PTC-страницы из fragments |
| `pywp/ptc_page_run.py` | блок запуска batch-расчёта |
| `pywp/ptc_target_import.py` / `pywp/ptc_target_import_dev.py` | импорт целей из WELLTRACK, таблицы и `.dev` |
| `pywp/eclipse_welltrack.py` | парсер WELLTRACK и табличного target-формата |
| `pywp/reference_trajectories.py` | импорт фактических/утверждённых референсов |
| `pywp/uncertainty.py` | модель неопределённости, включая ISCWSA MWD |
| `pywp/three_viewer.py` | обёртка кастомного Streamlit-компонента для 3D |
| `pywp/ptc_three_payload.py` | оптимизация 3D payload |
| `pywp/ptc_three_overrides.py` | сборка 3D overlay и редактируемых payload |
| `pywp/well_pad.py` | обнаружение кустов и расчёт pad layout |
| `pywp/coordinate_systems.py` / `pywp/coordinate_integration.py` | CRS и интеграция в UI |
| `tests/` | unit + integration + Streamlit AppTest покрытие |
| `docs.md` | техническая документация верхнего уровня |
| `manual_well_planning.md` | методологические заметки по планированию |
| `docs/coordinate_systems.md`, `docs/coordinate_integration.md` | отдельная документация по CRS |

Отдельно стоит понимать масштаб `ptc_core.py`: это почти `10k` строк, и он является историческим центром UI-сценариев. Большая часть “магии страницы” всё ещё проходит через него.

## 3. Пользовательские поверхности продукта

### 3.1 Главная рабочая поверхность: PTC

Вход:

- `pages/01_trajectory_constructor.py`
- далее `pywp/ptc_page.py`
- затем большая часть действий уходит в `pywp/ptc_core.py` и соседние `ptc_*` модули

Основные секции страницы:

1. импорт целей;
2. кусты и расчёт устьев;
3. референсные скважины;
4. запуск расчёта;
5. результаты, anti-collision, 3D и сопутствующие представления.

### 3.2 Single Well

`pages/02_single_well.py` — отдельный sandbox, где удобно:

- быстро воспроизводить геометрические кейсы;
- проверять solver без batch-слоя;
- смотреть диагностику ошибок;
- работать с одной траекторией и одним 3D-view без всей сложности PTC.

### 3.3 Classification

`pages/03_well_classification.py` — вспомогательная страница с бизнес-правилами:

- same/reverse direction;
- ordinary/complex/very complex;
- интерполяция ограничений по ГВ.

### 3.4 CRS Calculator

`pages/04_crs_calculator.py` — отдельный инструмент для пересчёта координат:

- одна точка;
- batch-таблица;
- поддержка WGS84 в decimal и DMS;
- отображение безопасных ограничений по доступным преобразованиям.

## 4. Главная архитектурная идея

Проект не разделён на “чистый API слой” и “тонкий UI”. На практике это гибрид:

- расчётные ядра (`planner.py`, `anticollision.py`, `uncertainty.py`, `well_pad.py`, `coordinate_systems.py`) написаны как обычные Python-модули;
- orchestration, state management и часть бизнес-правил сидят прямо в Streamlit-слое;
- `ptc_core.py` объединяет импорт, состояние, подготовку batch-расчёта, рендер таблиц и много специфичных сценариев.

Если коротко:

- “тяжёлая математика” — отдельные pure-ish модули;
- “что и когда пересчитать / как показать / как пережить rerun” — Streamlit-слой;
- “проект живой”, поэтому в нём много инженерных компромиссов в пользу UX и устойчивости.

## 5. Доменные сущности и соглашения

### 5.1 Базовые объекты

Ключевые модели:

- `Point3D` — XYZ-точка;
- `TrajectoryConfig` — расчётная конфигурация;
- `PlannerResult` — результат планирования;
- `WelltrackPoint` / `WelltrackRecord` — импортированные цели;
- `ImportedTrajectoryWell` — референсная фактическая/утверждённая траектория;
- `SuccessfulWellPlan` — успешный результат batch-планировщика;
- `AntiCollisionWell`, `AntiCollisionZone`, `AntiCollisionCorridor` — сущности anti-collision.

### 5.2 Координаты

Принятые соглашения в расчётном ядре:

- `X` / `Y` — плановые координаты;
- `Z` — вертикальная координата в принятой проектом системе, обычно положительная вниз;
- `MD` — measured depth;
- `INC` / `AZI` — инклинация и азимут;
- DLS внутри ядра в основном хранится как `deg/30m`.

UI местами показывает “ПИ” в пересчитанном формате, поэтому важно не путать представление для человека и внутренние единицы ядра.

### 5.3 Типы целевых записей

Проект поддерживает несколько разных геометрий входа:

- обычная цель: `S`, `t1`, `t3`;
- multi-target / multi-horizontal: расширенные последовательности целей;
- pilot wells: имена с суффиксом `_PL`;
- ЗБС: имена с `_ZBS`, а также часть `_2` кейсов;
- target-only боковые стволы без `S`;
- `.dev` импорт, из которого может родиться как обычная цель, так и более сложный профиль.

### 5.4 Критичное правило по MD

Во всех серьёзных расчётных ветках предполагается строго возрастающий `MD`. Это обязательная инварианта и для WELLTRACK, и для референсных траекторий, и для side-track / ZBS сценариев.

## 6. Импорт и подготовка данных

### 6.1 WELLTRACK и табличные цели

Основной парсер — `pywp/eclipse_welltrack.py`.

Поддерживаются:

- классический текстовый WELLTRACK;
- табличный ввод `Wellname / Point / X / Y / Z`;
- распознавание кодировок (`utf-8`, `cp1251`, `latin-1`);
- нормализация point aliases;
- валидация MD и структуры точек.

### 6.2 `.dev` импорт

Логика — `pywp/ptc_target_import_dev.py`.

Есть два режима:

- простой `.dev`, где траектория уже фактически сводится к `S / t1 / t3`;
- аналитический `.dev`, где из готовой траектории извлекаются KOP, entry, build/horizontal-участки и формируется `DevTargetImportSummary`.

Особенность:

- если `.dev` импортирован как простая цель, дальше он живёт как обычная target-запись и использует общие параметры расчёта;
- для части импортов возможна фиксация `t1` inclination по скважине.

### 6.3 Импорт референсов

`pywp/reference_trajectories.py` обслуживает:

- фактические скважины (`actual`);
- утверждённые проектные (`approved`);
- текстовые/табличные источники;
- WELLTRACK и `.dev`;
- дальнейшее построение survey-таблиц, DLS и azimuth.

Эти данные затем используются:

- для визуального сравнения;
- для anti-collision;
- для анализа фактического фонда;
- для построения глубинных кластеров и depth-based KOP heuristics.

### 6.4 Анализ фактического фонда

`pywp/actual_fund_analysis.py` умеет:

- реконструировать survey по фактическим скважинам;
- находить KOP;
- выделять BUILD / HOLD / HORIZONTAL зоны;
- считать horizontal entry;
- строить метрики по кустам и depth clusters;
- формировать `ActualFundKopDepthFunction`.

Это важная часть проекта, потому что он не только строит проектные траектории, но и опирается на фактический фонд как на инженерный референс.

## 7. Архитектура PTC-страницы

### 7.1 Тонкий shell + толстый core

`pywp/ptc_page.py` — это тонкий shell:

- вызывает `wt._init_state()`;
- выставляет базовые defaults;
- рендерит fragments по секциям;
- показывает результаты и run log;
- связывает импорт, reference, run и results.

### 7.2 Главный state namespace

Практически всё состояние страницы живёт в `st.session_state` с префиксом `wt_`.

Примеры ключей:

- `wt_records`, `wt_records_original`
- `wt_selected_names`
- `wt_summary_rows`, `wt_successes`
- `wt_last_error`, `wt_last_runtime_s`
- `wt_pad_configs`
- `wt_anticollision_last_run`
- `wt_anticollision_analysis_cache`
- `wt_prepared_well_overrides`
- `wt_edit_targets_pending_names`
- `wt_three_viewer_nonce`

Логика инициализации сосредоточена в `wt._init_state()` внутри `pywp/ptc_core.py`.

### 7.3 Важная инженерная особенность state management

В проекте много versioned/defaulted состояния. Например:

- `wt_ui_defaults_version`
- `force_ptc_defaults()`
- сигнатуры последнего запуска и последних calc params

Это сделано не “для красоты”, а чтобы Streamlit-переезды UI не ломали старую `session_state` после рефакторингов.

## 8. Расчётное ядро траектории

### 8.1 Центральный entrypoint

Основной entrypoint — `TrajectoryPlanner.plan()` в `pywp/planner.py`.

Схема работы:

1. валидация `TrajectoryConfig`;
2. подготовка геометрии цели через `_build_section_geometry`;
3. классификация траектории как same/reverse direction;
4. определение случая с нулевым азимутальным поворотом;
5. подбор параметров профиля через unified solver;
6. построение выходной инклинометрии методом minimum curvature;
7. расчёт DLS и возврат `PlannerResult`.

### 8.2 Что именно ищет солвер

По сути солвер подбирает геометрию траектории под систему ограничений:

- KOP;
- build sections;
- hold / post-entry geometry;
- азимутальный поворот;
- max inclination;
- DLS limits;
- допуски по lateral/vertical target miss;
- optionally — оптимизацию по MD, KOP или anti-collision clearance.

Внутри используются:

- `least_squares`
- `minimize`
- `differential_evolution`

Иными словами, это не один фиксированный closed-form алгоритм, а многосценарный оптимизационный солвер с fallback-логикой и рестартами.

### 8.3 Геометрическая модель

На инженерном уровне профиль оперирует секциями вида:

- вертикальный участок до KOP;
- BUILD1;
- HOLD / turn-совместимая секция;
- BUILD2;
- HORIZONTAL / post-entry section;
- для multi-target — дополнительные extension-секции.

Ключевые геометрические утилиты:

- `_horizontal_offset`
- `_mid_azimuth_deg`
- `_normalize_azimuth_deg`
- `_required_dls_for_t1_reach`
- `_radius_from_dls`
- `_dls_from_radius`

### 8.4 Minimum Curvature

После нахождения параметров профиля итоговая survey-таблица строится через minimum curvature:

- `minimum_curvature_increment`
- `compute_positions_min_curv`
- `add_dls`

Это важно: solver работает с параметризованной траекторией, а итоговая выходная инклинометрия вычисляется уже как инженерный survey по станциям.

### 8.5 Same / Reverse direction и классификация

`pywp/classification.py` хранит бизнес-правила классификации:

- таблицу опорных глубин;
- окно reverse direction;
- границы ordinary / complex / very complex;
- интерполяцию лимитов по `gv_m`.

Эта классификация влияет не только на UX-страницу, но и на инженерную интерпретацию сценариев.

### 8.6 Основные расчётные параметры

Критичные поля `TrajectoryConfig`:

- `md_step_m` — шаг выходной survey;
- `md_step_control_m` — более мелкий контрольный шаг для расчёта;
- `lateral_tolerance_m`, `vertical_tolerance_m`;
- `entry_inc_target_deg`, `entry_inc_tolerance_deg`;
- `max_inc_deg`;
- `dls_build_max_deg_per_30m`;
- `dls_build2_max_deg_per_30m`;
- `dls_horizontal_max_deg_per_30m`;
- `kop_min_vertical_m`;
- `use_fixed_kop`;
- `min_hold_inc_deg`;
- `optimization_mode`;
- `turn_solver_mode`;
- `turn_solver_max_restarts`;
- `max_total_md_postcheck_m`;
- `min_structural_segment_m`;
- `j_profile_policy`.

Очень важная тонкость:

- `max_total_md_postcheck_m` — это пользовательская post-check валидация, а не жёсткое ограничение внутри поиска решения.

### 8.7 J-profile

Поддержка J-профиля встроена прямо в solver и управляется через:

- `j_profile_policy = off / propose / prefer`

Проект также поддерживает legacy-совместимость через `offer_j_profile`, но фактическое “истинное” поле сейчас именно `j_profile_policy`.

### 8.8 Multi-target и multi-horizontal

`TrajectoryPlanner.plan_multi_target()`:

- строит базовую траекторию до первых двух целей;
- затем последовательно достраивает расширения до следующих целей;
- использует текущую конечную станцию как новую точку старта;
- подбирает направление на следующий target / next-next target;
- добавляет дополнительные участки в survey.

Отдельный тяжёлый слой — `pywp/multi_horizontal.py`, где:

- поддерживаются последовательности `1_t1/1_t3`, `2_t1/2_t3`, ...;
- проверяются extended stations;
- строятся `HORIZONTAL_BUILDn`, `HORIZONTALn`;
- контролируются DLS и точность попадания при extension.

### 8.9 Пилоты, ЗБС и sidetrack

`pywp/pilot_wells.py` — один из самых важных доменных модулей. Он:

- отличает обычные скважины от пилотов и ЗБС;
- извлекает parent-child отношение;
- валидирует target-only ZBS сценарии;
- считает окна зарезки;
- строит pilot/sidetrack планы.

Для side-track используется отдельный `SidetrackPlanner` из `pywp/sidetrack_solver.py`.

Его инженерная особенность:

- от окна зарезки до `t1` строится гладкий cubic Bezier build;
- дальше идёт прямой горизонтальный участок до `t3`;
- для кандидатов перебираются пары контрольных длин (`lead_m`, `tail_m`);
- выбирается решение, удовлетворяющее DLS и target miss.

Это заметно отличается от обычного unified planner и является отдельной веткой расчёта.

## 9. Инженерный расчёт anti-collision и пересечений

### 9.1 Что считается “пересечением” в этом проекте

Проект считает не просто геометрическое пересечение двух centerlines. Он работает с неопределённостью траекторий и анализирует пересечение/сближение uncertainty envelopes.

То есть ключевая сущность — это не “линиї пересеклись”, а:

- насколько близки осевые линии;
- какова суммарная неопределённость;
- где формируется corridor overlap;
- какой separation factor остаётся между двумя объектами.

### 9.2 Модель неопределённости

`pywp/uncertainty.py` поддерживает два режима:

- полноценная модель на основе digitized ISCWSA MWD error model;
- быстрая proxy-модель первого порядка для горячих оптимизационных циклов.

Конфигурация задаётся через `PlanningUncertaintyModel`:

- `sigma_inc_deg`
- `sigma_azi_deg`
- `sigma_lateral_drift_m_per_1000m`
- `confidence_scale`
- `sample_step_m`
- display-параметры эллипсов
- `iscwsa_tool_code`

Для UI по умолчанию используется preset семейства MWD poor/unknown magnetic.

### 9.3 Сборка anti-collision wells

Перед попарным анализом каждый успешный план превращается в `AntiCollisionWell`:

- station samples;
- covariance по станциям;
- uncertainty overlay;
- sample arrays;
- metadata по `t1`/`t3`;
- metadata по sidetrack parent/window.

Референсные скважины проходят тот же путь, но с отдельной маркировкой `is_reference_only`.

### 9.4 Попарный анализ

Главная идея в `pywp/anticollision.py`:

1. сначала грубый prefilter пар по envelope / lateral bounds;
2. затем сканирование по sampled stations;
3. затем локальное уточнение в проблемных местах;
4. формирование zones / corridors / report events.

Ключевые параметры:

- `DEFINITIVE_SCAN_STEP_M = 10.0`
- `DEFINITIVE_LOCAL_REFINE_STEP_M = 5.0`
- trigger на локальное уточнение при небольшом запасе по `SF`
- лимиты на число sample pairs и geometry rings

### 9.5 Separation factor и overlap

В anti-collision optimization логика устроена так:

- для кандидатной траектории и reference path строятся непрерывные polyline segments;
- для пар сегментов ищется closest probe;
- оценивается combined radius upper bound на основе ковариаций и `confidence_scale`;
- вычисляется `SF = centerline_distance / combined_radius`;
- overlap depth оценивается как превышение суммарного радиуса над фактической дистанцией.

Интерпретация:

- `SF < 1` означает фактический конфликт по модели;
- чем больше `SF`, тем больше запас;
- overlap depth нужен как более наглядная “глубина конфликта”, а не только бинарный статус.

### 9.6 Corridor и zone model

Модель результатов разделена на:

- `AntiCollisionZone` — локальная hotspot-точка/зона;
- `AntiCollisionCorridor` — интервал вдоль траектории;
- `AntiCollisionReportEvent` / groups — агрегированное представление для UI и рекомендаций.

Именно corridor-модель особенно важна в инженерном UX: она лучше описывает длинные участки близкого прохождения, чем одиночная минимальная точка.

### 9.7 Incremental anti-collision

Один из главных ускорителей проекта — инкрементальная anti-collision схема:

- cache по well signatures;
- cache по pair signatures;
- повторное использование уже собранных `AntiCollisionWell`;
- повторное использование уже рассчитанных pair corridors/zones.

Оркестрация лежит в `pywp/anticollision_rerun.py`:

- `build_incremental_anti_collision_analysis_for_successes`
- `build_anti_collision_wells_for_successes`
- pair/well reuse statistics

Это особенно важно после локального редактирования одной-двух скважин, когда пересчитывать весь антиколлижен “с нуля” было бы слишком дорого.

### 9.8 Dynamic cluster execution и prepared overrides

Поверх definitive anti-collision в проекте есть слой активного разрешения конфликтов:

- кластеризация рекомендаций;
- определение “moving wells”;
- подготовка override-контекстов;
- staged rerun конфликтных кластеров.

Здесь используется `AntiCollisionOptimizationContext`, который передаётся обратно в planner и меняет objective функции так, чтобы новая траектория:

- достигала target;
- соблюдала инженерные ограничения;
- увеличивала clearance относительно reference paths;
- по возможности сохраняла baseline KOP/build1/build2.

Это один из самых сильных инженерных элементов проекта: anti-collision не только диагностируется, но и подмешивается обратно в планирование как оптимизационное ограничение/цель.

## 10. Batch-расчёт

### 10.1 Оркестратор

Главная batch-точка — `WelltrackBatchPlanner` в `pywp/welltrack_batch.py`.

Он отвечает за:

- проход по выбранным скважинам;
- учёт pad layout;
- pilot/ZBS ветвления;
- prepared overrides;
- запуск planner/sidetrack planner;
- объединение summary/results;
- запуск anti-collision на успешных скважинах.

### 10.2 Порядок и состояние

Batch-слой должен учитывать:

- какие скважины вообще видимы для расчёта;
- какие требуют пересчёта после edit;
- какие имеют parent/pilot зависимости;
- какие должны использовать override config;
- какие reference wells входят в anti-collision scope.

Именно здесь соединяются:

- импорт,
- расчёт,
- anti-collision,
- UI summary,
- рекомендации.

## 11. Кусты и расчёт устьев

`pywp/well_pad.py` решает отдельную инженерную задачу: если несколько скважин стартуют из одной точки, их нужно грамотно развести по кусту.

Что умеет модуль:

- обнаруживать pad-группы по общим surface coordinates;
- вычислять auto pad azimuth;
- оценивать направление walking/skidding через PCA облака target midpoints;
- упорядочивать скважины по имени / глубине / projection;
- применять spacing и anchor rules.

Это не только UI-фича. Изменение устьев меняет геометрию всего последующего расчёта.

## 12. 3D слой и редактирование целей

### 12.1 Архитектура 3D

3D реализован через локальный Streamlit component:

- `pywp/three_viewer.py`
- assets в `pywp/three_viewer_assets/`

Компонент принимает JSON payload со сценой:

- линии;
- точки;
- mesh-объекты;
- подписи;
- legend tree;
- anti-collision overlays;
- edit channel.

### 12.2 Оптимизация payload

`pywp/ptc_three_payload.py` выполняет оптимизацию перед рендером:

- merge линий с одинаковыми style keys;
- merge point collections;
- merge mesh buffers;
- обрезание числа labels;
- decimation hover payload;
- расчёт merged bounds.

Это критично. Без этого 3D payload быстро становится слишком тяжёлым для Streamlit + браузера.

### 12.3 Редактирование

`pywp/ptc_three_overrides.py` и `pywp/ptc_edit_targets.py` обеспечивают:

- 3D edit event -> нормализация;
- запись изменений в records;
- инвалидацию старых результатов;
- подсветку изменённых скважин;
- перевод фокуса на “Все скважины / Anti-collision” при необходимости.

Важная тонкость:

- для части простых `.dev`-скважин, где surface XY исторически совпадает с `t1`, при редактировании `t1` может синхронно сдвигаться и surface XY, чтобы сохранить вертикальную логику от устья до первой цели.

## 13. Streamlit: как здесь сделана производительность

Это один из ключевых разделов handoff. В данном проекте производительность определяется не только скоростью математики, но и качеством управления rerun-моделью Streamlit.

### 13.1 Fragments

Главная PTC-страница нарезана на `@st.fragment`:

- импорт целей;
- layout кустов;
- overview;
- raw records;
- reference section;
- run section;
- results section.

Зачем это сделано:

- локальные действия в секции не всегда должны перерисовывать весь экран;
- выбор скважин, добавление куста, локальные UI-изменения дешевле прогонять fragment-rerun;
- тяжёлые результаты и anti-collision можно оставить нетронутыми до момента явного полного обновления.

Прямой пример — `pywp/ptc_page_run.py`:

- для select-all / add-pad / replace-with-pad используется `_rerun_fragment()`;
- после реального batch-run вызывается уже полный `_rerun_app()`, потому что результаты живут за пределами fragment.

Это очень важная архитектурная договорённость.

### 13.2 Forms

Проект активно использует `st.form(...)`.

Основной смысл:

- собрать несколько widget-изменений в одну транзакцию;
- не запускать тяжёлый перерасчёт на каждый ввод;
- отделить “редактирование параметров” от “commit действия”.

Формы есть в:

- импорте источников;
- run section PTC;
- single well;
- вспомогательных UI-блоках.

Полезные паттерны:

- `clear_on_submit=False` — не терять введённые значения;
- `enter_to_submit=False` там, где случайный Enter не должен запускать расчёт;
- отдельные submit buttons на изменение selection и на реальный запуск расчёта.

### 13.3 Session State как главный store

Производительность здесь во многом достигается через то, что дорого посчитанное состояние переиспользуется в `st.session_state`:

- результаты batch;
- last run log;
- anti-collision cache;
- pad config;
- prepared overrides;
- reference imports;
- подписи стейта и pending selections.

Фактически `session_state` — это здесь и UI-store, и cache, и workflow memory.

### 13.4 Pending-state паттерн

Очень важный местный приём:

- пользователь жмёт кнопку в fragment;
- изменение сначала пишется в `wt_pending_*`;
- после rerun это pending-состояние аккуратно синхронизируется в реальное active state.

Такой подход уменьшает хаос от Streamlit rerun order и помогает не терять selections.

### 13.5 Signature-based invalidation

Проект не полагается только на “надеюсь, Streamlit сам поймёт”.

Используются явные сигнатуры:

- `wt_last_calc_param_signature`
- сигнатуры ручных override-конфигов
- well signatures для anti-collision cache
- pair signatures для incremental pair reuse

Это даёт контролируемую инвалидацию без тотального пересчёта.

### 13.6 Streamlit cache используется точечно

В проекте почти нет слепого использования `st.cache_*`.

По сути явный `@st.cache_data(show_spinner=False)` здесь используется точечно для парсинга WELLTRACK текста. В остальном проект в основном предпочитает:

- ручные caches в `session_state`;
- `lru_cache` для статических текстовых asset-ов 3D viewer;
- identity/digest based short-lived cache для сериализованного 3D payload.

Причина простая:

- ядро активно работает с mutable DataFrame и сложным session workflow;
- у Streamlit cache в таких случаях легко появляются трудноотлаживаемые stale-state эффекты.

### 13.7 3D performance tricks

В 3D слое есть несколько важных оптимизаций:

- `lru_cache(maxsize=1)` на чтение HTML/JS asset-ов;
- digest runtime-а по mtimes/stats файлов;
- short cache последних сериализованных payload;
- ограничение числа labels;
- decimation hover points;
- merge однотипных geometry blocks.

Именно это удерживает custom component в рабочем состоянии на больших сценах.

### 13.8 Multiprocessing

Batch и anti-collision умеют multiprocessing, но очень консервативно.

Логика:

- до `3` выбранных скважин multiprocessing автоматически отключается;
- от `4` до `7` — обычно `2` процесса;
- от `8` и выше — `4` процесса.

Это реализовано в `pywp/ptc_page_run.py`.

Отдельный модуль `pywp/parallel.py` подбирает безопасный start method:

- `spawn` на Windows/macOS;
- `forkserver` на Linux;
- fallback обратно в `spawn`, если контекст недоступен.

Это критично из-за сочетания Streamlit + multiprocessing. Неаккуратный `fork` здесь очень легко приводит к нестабильности, зависаниям или поломке STDIN/runtime context.

### 13.9 Почему не всё распараллелено

Важно понимать локальную философию:

- для малых batch-ов последовательный расчёт часто быстрее, чем spin-up процессов;
- serial path проще и устойчивее;
- распараллеливание используют только там, где выигрыш окупает overhead.

### 13.10 Что сильнее всего влияет на реальную отзывчивость

На практике самые важные факторы производительности здесь такие:

1. размер и частота rerun;
2. объём 3D payload;
3. количество anti-collision pair checks;
4. процент reuse well/pair cache;
5. число reference wells, попавших в collision scope;
6. изоляция тяжёлых действий в forms/fragments.

## 14. Точки риска и технический долг

### 14.1 `ptc_core.py` как монолит

Это самая большая техническая точка риска.

Проблемы:

- много разнородной логики в одном файле;
- высокая связанность по `st.session_state`;
- трудно безопасно менять локальные UI-сценарии без понимания глобального контекста;
- приватные helpers `wt._...` уже фактически стали внутренним API.

Практический вывод:

- перед изменением PTC UI всегда читать не только локальный модуль, но и связанные `ptc_*` обвязки;
- не предполагать, что изменение “маленькое”, если оно касается ключей `wt_*`.

### 14.2 Доменные ветвления

Обычные скважины, пилоты, ZBS, `_2`, multi-horizontal, simple `.dev`, analytical `.dev`, reference wells, actual wells — всё это разные ветки логики.

Любая правка импорта или edit-flow почти автоматически имеет blast radius на несколько доменных сценариев.

### 14.3 DLS/PI и единицы

Одна из самых частых потенциальных ловушек:

- UI показывает часть параметров в одном привычном человеку виде;
- solver работает в `deg/30m`;
- вспомогательные утилиты (`dls_to_pi` и обратные) используются во многих местах.

Перед изменением формулы или отображения всегда проверять, в каких единицах реально живёт величина в расчётном ядре.

### 14.4 Streamlit stale-state bugs

Типовые классы багов здесь:

- old results after edit;
- stale prepared overrides;
- selection drift после rerun;
- “не тот” view mode после расчёта;
- ref/import state, переживший сценарий, в котором его уже надо было сбросить.

Обычно первыми смотреть:

- `wt_pending_*`;
- `wt_last_*signature*`;
- `wt_prepared_*`;
- `wt_anticollision_*cache*`;
- `wt_three_viewer_nonce`.

## 15. Тестовая стратегия

### 15.1 Что покрыто

Тесты покрывают:

- solver;
- segments / MCM / uncertainty;
- anti-collision;
- batch planner;
- pad layout;
- CRS;
- импорт WELLTRACK / `.dev`;
- 3D payload;
- Streamlit pages;
- prepared rerun / anticollision resolution;
- UI state sync.

### 15.2 Streamlit AppTest

Проект активно использует `streamlit.testing.v1.AppTest`, что особенно важно для проверки:

- формы;
- fragments;
- session state;
- page-to-page UX сценариев;
- regression-эффектов после UI refactor.

### 15.3 Как запускать

Из `README.md`:

```bash
pytest -q
python scripts/run_tests.py unit
python scripts/run_tests.py fast
python scripts/run_tests.py integration
python scripts/run_tests.py slow
python scripts/run_tests.py full
```

Есть и полезные служебные скрипты:

- `scripts/check_streamlit_pages.py`
- `scripts/project_pack.py`
- генераторы reference/actual depth cluster fixtures

## 16. Как безопасно вносить изменения

Рекомендации для следующего инженера:

1. Если задача касается математики, сначала работай в чистом модуле (`planner.py`, `anticollision.py`, `uncertainty.py`), а не в UI.
2. Если задача касается поведения страницы, сначала найди state keys и rerun path.
3. Любая правка импорта должна быть проверена минимум на:
   - обычной скважине,
   - пилоте,
   - ZBS,
   - `.dev`,
   - multi-horizontal кейсе.
4. Любая правка anti-collision должна проверяться и на full rebuild, и на incremental reuse path.
5. Любая правка 3D должна проходить через payload size sanity check.
6. Любая правка multiprocessing должна считаться рискованной, пока не проверена на Linux/macOS/Windows strategy assumptions.

## 17. С чего читать код новому человеку

Оптимальный порядок вхождения:

1. `README.md`
2. `HANDOFF.md` и `docs.md`
3. `pages/01_trajectory_constructor.py`
4. `pywp/ptc_page.py`
5. `pywp/ptc_page_run.py`
6. `pywp/planner.py`
7. `pywp/welltrack_batch.py`
8. `pywp/anticollision.py`
9. `pywp/anticollision_rerun.py`
10. `pywp/pilot_wells.py`
11. `pywp/ptc_target_import.py`, `pywp/eclipse_welltrack.py`
12. `pywp/three_viewer.py`, `pywp/ptc_three_payload.py`

Если нужно быстро понять только инженерный solver, порядок лучше такой:

1. `pywp/models.py`
2. `pywp/planner.py`
3. `pywp/mcm.py`
4. `pywp/planner_geometry.py`
5. `pywp/planner_validation.py`
6. `pywp/multi_horizontal.py`
7. `pywp/sidetrack_solver.py`

Если нужно быстро понять только anti-collision:

1. `pywp/uncertainty.py`
2. `pywp/anticollision_optimization.py`
3. `pywp/anticollision.py`
4. `pywp/anticollision_rerun.py`
5. `pywp/ptc_anticollision_view.py`

## 18. Итоговая инженерная оценка проекта

Сильные стороны проекта:

- мощное расчётное ядро с несколькими режимами геометрии;
- серьёзный anti-collision слой, а не декоративная проверка дистанций;
- хорошая доменная поддержка pilot/ZBS/multi-horizontal;
- зрелая Streamlit-практика: fragments, forms, signature invalidation, manual caches;
- хорошее покрытие тестами, включая AppTest.

Главные системные риски:

- `ptc_core.py` как монолит и скрытый внутренний API;
- сложность доменных ветвлений;
- высокая цена stale-state багов;
- необходимость очень бережно обращаться с производительностью rerun + 3D payload.

В целом это уже не “демо на Streamlit”, а насыщенное инженерное приложение, где успех правок почти всегда определяется не только формулой, но и тем, насколько аккуратно изменение встроено в state/rerun/performance модель всего интерфейса.
