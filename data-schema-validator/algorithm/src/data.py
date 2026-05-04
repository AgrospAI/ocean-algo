from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class OptionalEnumMixin:
    def __bool__(self) -> bool:
        return self is not getattr(type(self), "NONE")


class Format(OptionalEnumMixin, StrEnum):
    Json = "application/json"
    Xml = "application/xml"
    JsonLd = "application/ld+json"
    NTriples = "application/n-triples"
    RdfXml = "application/rdf+xml"
    Turtle = "text/turtle"
    Csvw = "text/csv"
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
