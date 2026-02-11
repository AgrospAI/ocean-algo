from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Sector(StrEnum):
    Industrial = "Industrial"
    Servicios = "Servicios"
    Comercial = "Comercial"
    Tecnologia = "Tecnología"
    Otro = "Otro"


class Size(StrEnum):
    PYME = "PYME"
    Mediana = "Mediana"
    Micro = "Micro"
    Pequeña = "Pequeña"


class AggregateAPI(BaseModel):
    url: str


class AggregateFilter(BaseModel):
    sector: Sector | None = Field(default=None)
    size: Size | None = Field(default=None)
    province: str | None = Field(default=None)

    def model_post_init(self, _: Any):
        # Ensure exactly two fields are set
        values = [self.sector, self.size, self.province]
        if sum(v is not None for v in values) != 2:
            raise ValueError("Exactly two of 'sector', 'size', 'province' must be set")

    def __str__(self) -> str:
        # Pick only the non-None fields
        parts = [str(v) for v in (self.sector, self.size, self.province) if v]
        return "_".join(parts)


class InputParameters(BaseModel):
    aggregate_filter: AggregateFilter
    aggregate_api: AggregateAPI
    responses_separator: str = ";"
