from typing import Any
from pydantic import BaseModel, field_validator

class InputParameters(BaseModel):
    refcat: str

    @field_validator("refcat", mode="before")
    @classmethod
    def is_valid_refcat(cls, v: Any) -> str | None:
        if v is None:
            return v
        stripped = str(v).strip()
        if stripped and not stripped.isalnum():
            raise ValueError(f"'{stripped}' must be alpha-numeric")
        return stripped