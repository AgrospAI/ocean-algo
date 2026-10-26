from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class InputParameters(BaseModel):
    title: str = Field("Profiling Report")
    sensitive: bool = Field(False)
    # Auto-detect by default so a compute request that carries no parameters
    # at all (the frontend may not prompt for them) still produces a report.
    auto_detect_timeseries_column: bool = Field(True)
    timeseries_columns_name: list[str] = Field(default_factory=list)

    @field_validator("timeseries_columns_name", mode="before")
    @classmethod
    def parse_stringified_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @model_validator(mode="after")
    def validate_timeseries(self) -> Self:
        # If manual mode was requested but no columns were supplied, fall back to
        # auto-detection instead of failing: the algorithm still yields a report.
        if not self.auto_detect_timeseries_column and not self.timeseries_columns_name:
            self.auto_detect_timeseries_column = True

        return self
