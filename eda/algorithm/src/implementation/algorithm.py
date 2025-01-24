from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from pandas import read_csv
from ydata_profiling import ProfileReport

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[Any] = None

    def _validate_input(self) -> "Algorithm":
        if not self._job_details.dids or len(self._job_details.dids) == 0:
            logger.warning("No DIDs found")
            raise ValueError("No DIDs found")

        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> "Algorithm":

        self._validate_input()

        first_did = self._job_details.dids[0]
        filename = self._job_details.files[first_did][0]

        df = read_csv(filename, sep=None, engine="python")

        self.results = ProfileReport(df, title="Profiling Report", sensitive=False)
        logger.info(f"Generated profiling report for {filename}")

        return self

    def save_result(self, path: Path) -> None:
        self.results.to_file(path)
