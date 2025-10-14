from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PeriodicityT = Literal["minutes", "hours", "days", "weeks", "months", "years"]


@dataclass
class InputParameters:
    data_separator: str | None = None
    data_target_column: str | None = None
    data_datetime_column: str | None = None
    data_splits: int | None = 2
    data_lags: int | None = 3
    data_periodicity: list[PeriodicityT] | None = None
    data_is_zipped: bool = False

    model_name: str = "AdaBoostRegressor"
    model_params: dict[str, any] | None = None

    metrics: list[str] = field(default_factory=lambda: ["neg_mean_squared_error"])
