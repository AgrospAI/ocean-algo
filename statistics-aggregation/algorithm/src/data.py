from __future__ import annotations

from dotenv import get_key
from pydantic import BaseModel, Field, model_validator


def get_url() -> str | None:
    return get_key("/data/transformations/algorithm", "S3_WRAPPER_URL")


class InputParameters(BaseModel):
    url: str | None = Field(init=False, default=get_url())

    @model_validator(mode="after")
    def validate_url_populated(self) -> InputParameters:
        if not self.url:
            raise ValueError("URL not populated from environment file")
        return self
