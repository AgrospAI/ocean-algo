from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ColumnNames:
    datetime: str
    target: str
    categorical: List[str]
    numeric: List[str]
