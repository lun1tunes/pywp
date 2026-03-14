from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

ModelT = TypeVar("ModelT", bound="FrozenModel")


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    def validated_copy(self: ModelT, **update: Any) -> ModelT:
        payload = self.model_dump(mode="python")
        payload.update(update)
        return self.__class__.model_validate(payload)


class FrozenArbitraryModel(FrozenModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )
