from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DiagnosticItem:
    reason_ru: str
    action_ru: str


@dataclass(frozen=True)
class ParsedSolverError:
    title_ru: str
    items: tuple[DiagnosticItem, ...]
    raw_message: str


_RE_FLOAT = r"[-+]?\d+(?:\.\d+)?"


def _pi_text_from_dls_text(value_text: str) -> str:
    try:
        value = float(value_text)
    except (TypeError, ValueError):
        return value_text
    return f"{value / 3.0:.2f}"


def _replace_dls_terms_for_ui(text: str) -> str:
    source = str(text or "")
    if not source:
        return source

    def _convert_deg_30_to_deg_10(match: re.Match[str]) -> str:
        value_text = match.group(1)
        return f"{_pi_text_from_dls_text(value_text)} deg/10m"

    converted = re.sub(
        rf"({_RE_FLOAT})\s*deg/30m",
        _convert_deg_30_to_deg_10,
        source,
    )
    converted = converted.replace("deg/30m", "deg/10m")
    converted = converted.replace("DLS", "ПИ")
    return converted


def ui_error_text(raw_message: str) -> str:
    return _replace_dls_terms_for_ui(str(raw_message or "").strip())


def _split_error(raw_message: str) -> tuple[str, list[str]]:
    message = str(raw_message or "").strip()
    if not message:
        return "", []

    marker = "\nReasons and actions:\n"
    if marker in message:
        head, tail = message.split(marker, maxsplit=1)
        bullets = [
            line.strip()[2:].strip()
            for line in tail.splitlines()
            if line.strip().startswith("- ")
        ]
        return head.strip(), bullets
    return message, []


def _translate_title_ru(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return "Ошибка расчета траектории."
    if "No valid VERTICAL->BUILD1->HOLD->BUILD2->HORIZONTAL solution" in source:
        return "Не найдено допустимое решение профиля VERTICAL->BUILD1->HOLD->BUILD2->HORIZONTAL."
    if "No valid trajectory solution found within configured limits" in source:
        return "Не найдено допустимое решение траектории."
    if "With current global max INC the t1->t3 geometry is infeasible without overbend" in source:
        return "Геометрия t1->t3 небурима при текущем max INC без overbend."
    if "Entry INC target exceeds configured max INC" in source:
        return "Целевой INC входа выше допустимого max INC."
    if "Failed to hit t1 within tolerance" in source:
        return "Точка t1 не достигнута в заданном допуске."
    if "Failed to hit t3 within tolerance" in source:
        return "Точка t3 не достигнута в заданном допуске."
    if "DLS limit exceeded on segment" in source:
        return "Превышен лимит ПИ на одном из участков."
    return "Ошибка расчета траектории."


def _item_from_text(line: str) -> DiagnosticItem:
    text = str(line or "").strip()

    match = re.search(
        rf"With current global max INC.*Required straight INC is ({_RE_FLOAT}) deg, max INC is ({_RE_FLOAT}) deg",
        text,
    )
    if match:
        inc_req, inc_max = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=(
                f"Геометрия t1->t3 небурима без overbend: нужен INC {inc_req} deg, "
                f"допустимый max INC {inc_max} deg."
            ),
            action_ru=(
                "Увеличьте max INC, либо сделайте t3 глубже и/или ближе к t1 по горизонтальной проекции."
            ),
        )

    match = re.search(
        rf"Entry INC target exceeds configured max INC.*entry_inc_target=({_RE_FLOAT}) deg, max_inc=({_RE_FLOAT}) deg",
        text,
    )
    if match:
        inc_target, inc_max = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=(
                f"Целевой INC входа {inc_target} deg выше max INC {inc_max} deg."
            ),
            action_ru="Уменьшите entry INC target или увеличьте max INC.",
        )

    match = re.search(
        rf"Failed to hit t[13] within tolerance.*Miss=({_RE_FLOAT}) m, tolerance=({_RE_FLOAT}) m",
        text,
    )
    if match:
        miss, tol = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=f"Цель не достигнута в допуске: промах {miss} м при допуске {tol} м.",
            action_ru=(
                "Увеличьте допустимое число рестартов решателя; "
                "также можно ослабить допуск или скорректировать геометрию целей."
            ),
        )

    match = re.search(
        rf"BUILD DLS upper bound is insufficient.*max ({_RE_FLOAT}) deg/30m, required about ({_RE_FLOAT}) deg/30m",
        text,
    )
    if match:
        dls_max, dls_req = match.group(1), match.group(2)
        pi_max = _pi_text_from_dls_text(dls_max)
        pi_req = _pi_text_from_dls_text(dls_req)
        return DiagnosticItem(
            reason_ru=(
                f"Ограничение BUILD по ПИ недостаточно для достижения t1: "
                f"доступно max {pi_max} deg/10m, требуется примерно {pi_req} deg/10m."
            ),
            action_ru=(
                "Увеличьте max ПИ BUILD, либо уменьшите горизонтальный отход до t1 "
                "и/или увеличьте TVD t1."
            ),
        )

    match = re.search(
        rf"DLS limit exceeded on segment ([A-Z0-9_]+):\s*({_RE_FLOAT})\s*>\s*({_RE_FLOAT})",
        text,
    )
    if match:
        segment_name, dls_actual, dls_limit = match.group(1), match.group(2), match.group(3)
        pi_actual = _pi_text_from_dls_text(dls_actual)
        pi_limit = _pi_text_from_dls_text(dls_limit)
        return DiagnosticItem(
            reason_ru=(
                f"Превышен лимит ПИ на сегменте {segment_name}: "
                f"{pi_actual} > {pi_limit} deg/10m."
            ),
            action_ru="Снизьте фактический ПИ на этом сегменте или увеличьте допустимый лимит ПИ.",
        )

    match = re.search(
        rf"Minimum VERTICAL before KOP is too deep.*kop_min_vertical=({_RE_FLOAT}) m, t1 TVD=({_RE_FLOAT}) m",
        text,
    )
    if match:
        kop, tvd = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=(
                f"Минимальный VERTICAL до KOP слишком большой: "
                f"kop_min_vertical={kop} м при t1 TVD={tvd} м."
            ),
            action_ru="Уменьшите параметр «Мин VERTICAL до KOP, м» либо углубите t1.",
        )

    match = re.search(
        rf"t1->t3 geometry requires INC ({_RE_FLOAT}) deg, above max INC ({_RE_FLOAT}) deg",
        text,
    )
    if match:
        inc_req, inc_max = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=(
                f"Для геометрии t1->t3 требуется INC {inc_req} deg, "
                f"что выше max INC {inc_max} deg."
            ),
            action_ru=(
                "Сделайте t3 глубже и/или ближе к t1 по горизонтальной проекции, "
                "либо увеличьте max INC."
            ),
        )

    match = re.search(
        rf"Post-entry t1->t3 connection is not feasible with HORIZONTAL DLS limit ({_RE_FLOAT}) deg/30m; requires about ({_RE_FLOAT}) deg/30m",
        text,
    )
    if match:
        dls_lim, dls_req = match.group(1), match.group(2)
        pi_lim = _pi_text_from_dls_text(dls_lim)
        pi_req = _pi_text_from_dls_text(dls_req)
        return DiagnosticItem(
            reason_ru=(
                f"Участок после входа в пласт не бурим при HORIZONTAL ПИ limit {pi_lim} deg/10m: "
                f"нужно примерно {pi_req} deg/10m."
            ),
            action_ru="Увеличьте лимит HORIZONTAL ПИ или переместите t3 ближе к t1 по разрезу.",
        )

    if "Post-entry t1->t3 connection is infeasible even with high DLS scan up to 30 deg/30m" in text:
        pi_scan_max = _pi_text_from_dls_text("30")
        return DiagnosticItem(
            reason_ru=(
                f"Участок после входа в пласт небурим даже при высоком ПИ (проверка до {pi_scan_max} deg/10m)."
            ),
            action_ru=(
                "Измените геометрию целей: увеличьте TVD t3 и/или сократите отход t1->t3; "
                "при необходимости смягчите entry INC target/max INC."
            ),
        )

    match = re.search(
        rf"Total MD limit is too restrictive.*minimum required MD is about ({_RE_FLOAT}) m, limit is ({_RE_FLOAT}) m",
        text,
    )
    if match:
        md_req, md_lim = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=(
                f"Ограничение по общей длине ствола слишком жесткое: "
                f"минимально нужно ~{md_req} м, установлен лимит {md_lim} м."
            ),
            action_ru="Увеличьте max_total_md_m или упростите геометрию целей.",
        )

    if "BUILD DLS interval is empty after constraints" in text:
        return DiagnosticItem(
            reason_ru="Допустимый интервал BUILD ПИ пустой после применения всех ограничений.",
            action_ru="Проверьте и разведите min/max ПИ BUILD и лимиты ПИ по сегментам.",
        )

    match = re.search(
        rf"Total MD exceeds configured post-check limit.*Calculated total MD=({_RE_FLOAT}) m, limit=({_RE_FLOAT}) m",
        text,
    )
    if match:
        md_actual, md_limit = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=(
                f"Итоговая длина скважины по MD превышает заданный порог: "
                f"получено {md_actual} м при лимите {md_limit} м."
            ),
            action_ru=(
                "Скважина получается слишком длинной для текущего лимита MD. "
                "Увеличьте порог «Макс итоговая MD (постпроверка), м» или упростите геометрию целей."
            ),
        )

    match = re.search(
        rf"Solver endpoint miss to t1 after optimization is ({_RE_FLOAT}) m \(tolerance ({_RE_FLOAT}) m\)",
        text,
    )
    if match:
        miss, tol = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=f"После оптимизации решателя промах по t1 составил {miss} м (допуск {tol} м).",
            action_ru=(
                "Увеличьте допустимое число рестартов решателя, ослабьте допуск по позиции "
                "или скорректируйте геометрию целей."
            ),
        )

    return DiagnosticItem(
        reason_ru=_replace_dls_terms_for_ui(text),
        action_ru="Проверьте ограничения профиля и геометрию целей.",
    )


def parse_solver_error(raw_message: str) -> ParsedSolverError:
    title_raw, bullets = _split_error(raw_message=raw_message)
    if bullets:
        items = tuple(_item_from_text(item) for item in bullets)
    else:
        items = tuple()
        if title_raw:
            items = (_item_from_text(title_raw),)
    return ParsedSolverError(
        title_ru=_translate_title_ru(title_raw),
        items=items,
        raw_message=str(raw_message or "").strip(),
    )


def diagnostics_rows_ru(raw_message: str) -> list[dict[str, str]]:
    parsed = parse_solver_error(raw_message=raw_message)
    rows = [
        {"Причина": item.reason_ru, "Что изменить": item.action_ru}
        for item in parsed.items
    ]
    return rows


def summarize_problem_ru(raw_message: str) -> str:
    parsed = parse_solver_error(raw_message=raw_message)
    if parsed.items:
        return parsed.items[0].reason_ru
    return parsed.title_ru
