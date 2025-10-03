from logging import getLogger
from pathlib import Path
from typing import Self

import pandas as pd
from oceanprotocol_job_details.ocean import JobDetails

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails) -> None:
        self._job_details = job_details
        self._results: pd.DataFrame | None = None

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            assert False, "Run algorithm first"
        return self._results

    def _validate_input(self) -> None:
        if not self._job_details.ddos:
            logger.warning("No DDOs found")
            raise ValueError("No DDOs found")

        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> Self:
        self._validate_input()

        filename = self._job_details.files[0].input_files[0]

        df = pd.read_csv(filename)
        self._results = df.describe(include="all")

        logger.info(f"Descriptive statistics for {filename}: \n {self._results}")

        return self

    def save_result(self, path: Path) -> None:
        self.results.to_csv(path)
