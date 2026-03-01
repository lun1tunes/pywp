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
    if "No valid TURN solution found for non-coplanar targets" in source:
        return "Не найдено допустимое TURN-решение для некомпланарных целей."
    if "No valid reverse-direction profile found" in source:
        return "Не найдено допустимое решение профиля «цели в обратном направлении»."
    if "No valid reverse TURN solution found" in source:
        return "Не найдено допустимое reverse TURN-решение."
    if "With current global max INC the t1->t3 geometry is infeasible without overbend" in source:
        return "Геометрия t1->t3 небурима при текущем max INC без overbend."
    if "Entry INC target exceeds configured max INC" in source:
        return "Целевой INC входа выше допустимого max INC."
    if "Failed to hit t1 within tolerance" in source:
        return "Точка t1 не достигнута в заданном допуске."
    if "Failed to hit t3 within tolerance" in source:
        return "Точка t3 не достигнута в заданном допуске."
    if "DLS limit exceeded on segment" in source:
        return "Превышен лимит DLS на одном из участков."
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
                "Увеличьте глубину поиска солвера (TURN samples/starts), "
                "ослабьте допуск или скорректируйте геометрию целей."
            ),
        )

    match = re.search(
        rf"BUILD DLS upper bound is insufficient.*max ({_RE_FLOAT}) deg/30m, required about ({_RE_FLOAT}) deg/30m",
        text,
    )
    if match:
        dls_max, dls_req = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=(
                f"Ограничение BUILD DLS недостаточно для достижения t1: "
                f"доступно max {dls_max} deg/30m, требуется примерно {dls_req} deg/30m."
            ),
            action_ru=(
                "Увеличьте max DLS BUILD, либо уменьшите горизонтальный отход до t1 "
                "и/или увеличьте TVD t1."
            ),
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
        return DiagnosticItem(
            reason_ru=(
                f"Участок после входа в пласт не бурим при HORIZONTAL DLS limit {dls_lim} deg/30m: "
                f"нужно примерно {dls_req} deg/30m."
            ),
            action_ru="Увеличьте лимит HORIZONTAL DLS или переместите t3 ближе к t1 по разрезу.",
        )

    if "Post-entry t1->t3 connection is infeasible even with high DLS scan up to 30 deg/30m" in text:
        return DiagnosticItem(
            reason_ru=(
                "Участок после входа в пласт небурим даже при высоком DLS (проверка до 30 deg/30m)."
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
            reason_ru="Допустимый интервал BUILD DLS пустой после применения всех ограничений.",
            action_ru="Проверьте и разведите min/max DLS BUILD и лимиты DLS по сегментам.",
        )

    match = re.search(
        rf"Reverse mode selected by classification: t1 offset=({_RE_FLOAT}) m is inside reverse range \[({_RE_FLOAT}), ({_RE_FLOAT})\] m",
        text,
    )
    if match:
        offset, left, right = match.group(1), match.group(2), match.group(3)
        return DiagnosticItem(
            reason_ru=(
                f"По классификации выбран reverse-режим: отход t1={offset} м попадает "
                f"в диапазон [{left}, {right}] м."
            ),
            action_ru=(
                "Если нужен профиль «цели в одном направлении», увеличьте отход до t1 "
                "выше верхней границы reverse-диапазона."
            ),
        )

    match = re.search(
        rf"To force same-direction trajectory.*offset above about ({_RE_FLOAT}) m",
        text,
    )
    if match:
        target = match.group(1)
        return DiagnosticItem(
            reason_ru="Текущая цель t1 слишком близка для same-direction классификации.",
            action_ru=f"Увеличьте горизонтальный отход до t1 выше ~{target} м.",
        )

    if "Reverse INC bounds are too narrow" in text:
        return DiagnosticItem(
            reason_ru="Диапазон reverse INC слишком узкий для поиска решения.",
            action_ru="Расширьте границы reverse INC: уменьшите min и/или увеличьте max.",
        )

    if "If reverse profile stays infeasible" in text:
        return DiagnosticItem(
            reason_ru="Reverse-профиль не сходится в текущих ограничениях.",
            action_ru="Увеличьте BUILD max DLS, поднимите reverse_inc_max_deg или уменьшите kop_min_vertical_m.",
        )

    match = re.search(
        rf"TURN endpoint miss to t1 after optimization is ({_RE_FLOAT}) m \(tolerance ({_RE_FLOAT}) m\)",
        text,
    )
    if match:
        miss, tol = match.group(1), match.group(2)
        return DiagnosticItem(
            reason_ru=f"После оптимизации TURN промах по t1 составил {miss} м (допуск {tol} м).",
            action_ru=(
                "Увеличьте TURN QMC samples / TURN local starts, ослабьте допуск по позиции "
                "или скорректируйте геометрию целей."
            ),
        )

    return DiagnosticItem(
        reason_ru=text,
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
