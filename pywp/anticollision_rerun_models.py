from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pywp.anticollision_optimization import AntiCollisionOptimizationContext


@dataclass(frozen=True)
class ReferenceWindow:
    well_name: str
    md_start_m: float
    md_end_m: float

    def merged_with(self, other: "ReferenceWindow") -> "ReferenceWindow":
        return ReferenceWindow(
            well_name=str(self.well_name),
            md_start_m=min(float(self.md_start_m), float(other.md_start_m)),
            md_end_m=max(float(self.md_end_m), float(other.md_end_m)),
        )

    def as_tuple(self) -> tuple[float, float]:
        return float(self.md_start_m), float(self.md_end_m)


@dataclass(frozen=True)
class PreparedOverride:
    update_fields: Mapping[str, object]
    source: str
    reason: str
    optimization_context: AntiCollisionOptimizationContext | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "update_fields": dict(self.update_fields),
            "source": str(self.source),
            "reason": str(self.reason),
            "optimization_context": self.optimization_context,
        }


@dataclass
class TrajectoryOverrideSpec:
    well_name: str
    candidate_md_start_m: float
    candidate_md_end_m: float
    reference_windows: dict[str, ReferenceWindow] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    prefer_lower_kop: bool = False
    prefer_higher_build1: bool = False
    prefer_keep_kop: bool = False
    prefer_keep_build1: bool = False
    prefer_adjust_build2: bool = False

    def expand_candidate_window(self, md_start_m: float, md_end_m: float) -> None:
        self.candidate_md_start_m = min(
            float(self.candidate_md_start_m),
            float(md_start_m),
        )
        self.candidate_md_end_m = max(
            float(self.candidate_md_end_m),
            float(md_end_m),
        )

    def add_reference_window(
        self,
        reference_name: str,
        md_start_m: float,
        md_end_m: float,
    ) -> None:
        window = ReferenceWindow(
            well_name=str(reference_name),
            md_start_m=float(md_start_m),
            md_end_m=float(md_end_m),
        )
        current = self.reference_windows.get(str(reference_name))
        self.reference_windows[str(reference_name)] = (
            window if current is None else current.merged_with(window)
        )

    def add_reason(self, reason: str) -> None:
        self.reasons.append(str(reason))
