from __future__ import annotations

from enum import StrEnum
from os import environ, getenv
from pathlib import Path

from dotenv import get_key
from pydantic import BaseModel, Field, model_validator


class OptionalEnumMixin:
    def __bool__(self) -> bool:
        return self is not getattr(type(self), "NONE")


class Sector(OptionalEnumMixin, StrEnum):
    Industrial = "Industrial"
    Servicios = "Servicios"
    Comercio = "Comercio"
    Tecnologia = "Tecnología"
    Otro = "Otro"
    NONE = "-"


class Size(OptionalEnumMixin, StrEnum):
    Micro = "Micro"
    Pequeña = "Pequeña"
    Mediana = "Mediana"
    Grande = "Grande"
    NONE = "-"


class InputParameters(BaseModel):
    sector: Sector = Field(default=Sector.NONE)
    size: Size = Field(default=Size.NONE)
    province: str | None = Field(default=None)

    responses_separator: str = ";"

    @model_validator(mode="after")
    def validate_exactly_two_filters(self) -> InputParameters:
        values = [self.sector, self.size, self.province]
        if sum(bool(v) for v in values) != 2:
            raise ValueError("Exactly two of 'sector', 'size', 'province' must be set")
        return self

    @property
    def filter_key(self) -> str:
        """
        Equivalent to AggregateFilter.__str__()
        """
        parts = [str(v) for v in (self.sector, self.size, self.province) if v]
        return "_".join(parts)
