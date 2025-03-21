from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Periodicity(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

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


@dataclass(frozen=True)
class InputParameters:
    separator: Optional[str] = None
    target_column: Optional[str] = None
    datetime_column: Optional[str] = None
    split: Optional[float] = None
    lags: Optional[int] = None
    periodicity: Optional[List[Periodicity]] = None
