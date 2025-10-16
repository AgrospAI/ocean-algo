from dataclasses import dataclass
from typing import Literal


@dataclass
class InputParameters:
    separator: str
    datetime_col: str
    target_col: str
    predict_steps: int
    is_zipped: bool
    lag_diff: int = 0
    lag_type: Literal[
        "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"
    ] = "hours"
