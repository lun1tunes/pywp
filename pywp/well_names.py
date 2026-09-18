from __future__ import annotations

__all__ = [
    "ALT_BRANCH_SUFFIX",
    "PILOT_SUFFIX",
    "ZBS_SUFFIX",
    "is_alt_branch_name",
    "is_pilot_name",
    "is_zbs_name",
    "parent_name_for_pilot",
    "parent_name_for_zbs",
    "pilot_name_for_parent",
    "well_name_key",
]

PILOT_SUFFIX = "_PL"
ZBS_SUFFIX = "_ZBS"
ALT_BRANCH_SUFFIX = "_2"


def _parent_for_optional_underscore_suffix(
    name: object,
    *,
    suffix: str,
) -> str | None:
    text = str(name).strip()
    folded = text.casefold()
    bare_suffix = str(suffix).strip().lstrip("_").casefold()
    separated_suffix = f"_{bare_suffix}"
    if folded.endswith(separated_suffix):
        parent = text[: -len(separated_suffix)]
    elif folded.endswith(bare_suffix):
        parent = text[: -len(bare_suffix)]
    else:
        return None
    return parent if parent.strip() else None


def is_pilot_name(name: object) -> bool:
    return _parent_for_optional_underscore_suffix(
        name,
        suffix=PILOT_SUFFIX,
    ) is not None


def is_zbs_name(name: object) -> bool:
    return _parent_for_optional_underscore_suffix(
        name,
        suffix=ZBS_SUFFIX,
    ) is not None


def is_alt_branch_name(name: object) -> bool:
    text = str(name).strip()
    return len(text) > len(ALT_BRANCH_SUFFIX) and text.endswith(ALT_BRANCH_SUFFIX)


def parent_name_for_pilot(name: object) -> str:
    text = str(name).strip()
    return (
        _parent_for_optional_underscore_suffix(text, suffix=PILOT_SUFFIX)
        or text
    )


def parent_name_for_zbs(name: object) -> str:
    text = str(name).strip()
    zbs_parent = _parent_for_optional_underscore_suffix(text, suffix=ZBS_SUFFIX)
    if zbs_parent is not None:
        return zbs_parent
    if is_alt_branch_name(text):
        return text[: -len(ALT_BRANCH_SUFFIX)]
    return text


def pilot_name_for_parent(name: object) -> str:
    return f"{str(name).strip()}{PILOT_SUFFIX}"


def well_name_key(name: object) -> str:
    """Return a case-insensitive key with compact suffixes canonicalized."""

    text = str(name).strip()
    pilot_parent = _parent_for_optional_underscore_suffix(
        text,
        suffix=PILOT_SUFFIX,
    )
    if pilot_parent is not None:
        return f"{pilot_parent.strip().casefold()}{PILOT_SUFFIX.casefold()}"
    zbs_parent = _parent_for_optional_underscore_suffix(
        text,
        suffix=ZBS_SUFFIX,
    )
    if zbs_parent is not None:
        return f"{zbs_parent.strip().casefold()}{ZBS_SUFFIX.casefold()}"
    return text.casefold()
