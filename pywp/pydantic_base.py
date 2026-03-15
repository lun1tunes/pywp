from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

ModelT = TypeVar("ModelT", bound="FrozenModel")
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def validated_copy(self: ModelT, **update: Any) -> ModelT:
        payload = self.model_dump(mode="python", round_trip=True)
        payload.update(update)
        return self.__class__.model_validate(payload)


class FrozenArbitraryModel(FrozenModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )


def coerce_model_like(value: Any, model_cls: type[BaseModelT]) -> BaseModelT:
    if isinstance(value, model_cls):
        return value
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="python", round_trip=True)
        return model_cls.model_validate(payload)
    if isinstance(value, Mapping):
        return model_cls.model_validate(dict(value))
    return model_cls.model_validate(value, from_attributes=True)
