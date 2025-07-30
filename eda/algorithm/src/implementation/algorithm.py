from logging import getLogger
from pathlib import Path
from typing import Optional, Self

from oceanprotocol_job_details.job_details import JobDetails
from pandas import read_csv
from ydata_profiling import ProfileReport

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[ProfileReport] = None

    def _validate_input(self) -> Self:
        assert self._job_details.files, "No files found"

    def run(self) -> Self:
        """The algorithm entrypoint. This method does the following:

        1. Loads the input data from the first given file.
        1. Generate the summary report.

        """

        # Validates the given JobDetails instance
        self._validate_input()

        # Get input filepath
        input_path = self._job_details.files[0].input_files[0]
        df = read_csv(input_path, sep=None, engine="python")

        self.results = ProfileReport(df, title="Profiling Report", sensitive=False)
        logger.info(f"Generated profiling report for {input_path.name}")

        return self

    def save_result(self, path: Path) -> None:
        self.results.to_file(path)
