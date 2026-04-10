from typing import Self

from pydantic import BaseModel, Field, model_validator


class InputParameters(BaseModel):
    title: str = Field("Profiling Report")
    sensitive: bool = Field(False)
    auto_detect_timeseries_column: bool = Field(False)
    timeseries_columns_name: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timeseries(self) -> Self:
        if not self.auto_detect_timeseries_column:
            assert len(self.timeseries_columns_name) != 0, (
                "If column auto-detect is not set, you must provide timeseries columns"
            )

        return self
