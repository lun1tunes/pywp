from __future__ import annotations


ANTI_COLLISION_STAGE_EARLY_KOP_BUILD1 = "early_kop_build1"
ANTI_COLLISION_STAGE_LATE_TRAJECTORY = "late_trajectory"


def anti_collision_stage_from_context(context: object | None) -> str | None:
    if context is None:
        return None
    prefer_lower_kop = bool(getattr(context, "prefer_lower_kop", False))
    prefer_higher_build1 = bool(getattr(context, "prefer_higher_build1", False))
    prefer_keep_kop = bool(getattr(context, "prefer_keep_kop", False))
    prefer_keep_build1 = bool(getattr(context, "prefer_keep_build1", False))
    prefer_adjust_build2 = bool(getattr(context, "prefer_adjust_build2", False))
    if prefer_lower_kop or prefer_higher_build1:
        return ANTI_COLLISION_STAGE_EARLY_KOP_BUILD1
    if prefer_keep_kop or prefer_keep_build1 or prefer_adjust_build2:
        return ANTI_COLLISION_STAGE_LATE_TRAJECTORY
    return None
