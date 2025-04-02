from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Periodicity(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

    @property
    def value(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, value: str) -> "Periodicity":
        if value not in cls._value2member_map_:
            raise ValueError(f"Invalid periodicity: {value}")
        return cls(value)

    def __repr__(self) -> str:
        return f"Periodicity('{self.value}')"


@dataclass(frozen=True)
class ColumnNames:
    datetime: str
    target: str
    categorical: List[str]
    numeric: List[str]


@dataclass
class ModelParameters:
    name: str = "AdaBoostRegressor"
    parameters: dict[str, any] | None = None
    metrics: List[str] = field(default_factory=lambda: ["neg_mean_squared_error"])


@dataclass
class DatasetParameters:
    separator: str | None = None
    target_column: str | None = None
    datetime_column: str | None = None
    split: float | None = 0.7
    lags: int | None = 3
    periodicity: List[Periodicity] | None = None


@dataclass
class InputParameters:
    model: ModelParameters
    dataset: DatasetParameters
