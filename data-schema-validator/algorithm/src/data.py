from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class OptionalEnumMixin:
    def __bool__(self) -> bool:
        return self is not getattr(type(self), "NONE")


class Format(OptionalEnumMixin, StrEnum):
    json = "application/json"
    xml = "application/xml"
    jsonld = "application/ld+json"
    n_triples = "application/n-triples"
    rdf_xml = "application/rdf+xml"
    turtle = "text/turtle"
    csvw = "text/csv"
    NONE = "-"


class InputParameters(BaseModel):
    format: Format = Field(default=Format.NONE)

    conformsTo: HttpUrl | None = Field(
        default=None,
        description="A valid URL pointing to the validation schema (e.g., SHACL, XSD, JSON Schema).",
    )

    @field_validator("conformsTo", mode="before")
    @classmethod
    def handle_empty_string(cls, v):
        if v == "":
            return None
        if v == "-":
            return None
        return v
