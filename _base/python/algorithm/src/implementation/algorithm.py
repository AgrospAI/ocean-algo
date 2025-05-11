from logging import getLogger
from pathlib import Path
from typing import Any, Optional, TypeVar, Value
import pathlib
from oceanprotocol_job_details.ocean import JobDetails

T = TypeVar("T")

logger = getLogger(__name__)

_ResultType = Any


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[_ResultType] = None

    def _validate_input(self) -> None:
        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> "Algorithm":
        raise NotImplementedError()

        # self._validate_input()
        # self.results = "ALGO RESULTS"
        # return self

    def save_result(self, path: Path) -> None:
        raise NotImplementedError()

        # with(path.open("w", encoding="utf-8") as f):
        #     try:
        #         f.write(self.results)
        #         logger.info(f"Saved results to {path}")
        #     except:
        #         logger.error(f"Failed to save results to {path}")
