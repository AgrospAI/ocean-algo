from __future__ import annotations

from dotenv import load_dotenv, get_key
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Sector(StrEnum):
    Industrial = "Industrial"
    Servicios = "Servicios"
    Comercio = "Comercio"
    Tecnologia = "Tecnología"
    Otro = "Otro"


class Size(StrEnum):
    Micro = "Micro"
    Pequeña = "Pequeña"
    Mediana = "Mediana"
    Grande = "Grande"


def get_url() -> str | None:
    return get_key("/data/transformations/algorithm", "S3_WRAPPER_URL")


class InputParameters(BaseModel):
    sector: Sector | None = Field(default=None)
    size: Size | None = Field(default=None)
    province: str | None = Field(default=None)

    url: str | None = Field(init=False, default=get_url())

    responses_separator: str = ";"

    @model_validator(mode="after")
    def validate_url_populated(self) -> InputParameters:
        if not self.url:
            raise ValueError("URL not populated from environment file")
        return self

    @model_validator(mode="after")
    def validate_exactly_two_filters(self) -> InputParameters:
        values = [self.sector, self.size, self.province]
        if sum(v is not None for v in values) != 2:
            raise ValueError("Exactly two of 'sector', 'size', 'province' must be set")
        return self

    @property
    def filter_key(self) -> str:
        """
        Equivalent to AggregateFilter.__str__()
        """
        parts = [str(v) for v in (self.sector, self.size, self.province) if v]
        return "_".join(parts)
