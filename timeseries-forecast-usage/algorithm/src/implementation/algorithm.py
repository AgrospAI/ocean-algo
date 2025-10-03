from logging import getLogger
from pathlib import Path
from typing import Any, Self, TypeVar

from oceanprotocol_job_details.ocean import JobDetails

from implementation.data import InputParameters

logger = getLogger(__name__)
_ResultType = TypeVar("_ResultType", bound=Any)


class Algorithm:
    def __init__(self, job_details: JobDetails[InputParameters]) -> None:
        self._job_details = job_details
        self._results: _ResultType | None = None

    @property
    def results(self) -> _ResultType:
        if self._results is None:
            raise ValueError("No results available. Please run the algorithm first.")
        return self._results

    def _validate_input(self) -> None:
        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> Self:
        raise NotImplementedError()

        # self._validate_input()
        # self._results = "ALGO RESULTS"
        # return self

    def save_results(self, path: Path) -> None:
        raise NotImplementedError()

        # with(path.open("w", encoding="utf-8") as f):
        #     try:
        #         f.write(self._results)
        #         logger.info(f"Saved results to {path}")
        #     except:
        #         logger.error(f"Failed to save results to {path}")
