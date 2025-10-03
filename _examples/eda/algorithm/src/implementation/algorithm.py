from logging import getLogger
from pathlib import Path
from typing import Self

from oceanprotocol_job_details.ocean import JobDetails
from pandas import read_csv
from ydata_profiling import ProfileReport

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self._results: ProfileReport | None = None

    @property
    def results(self) -> ProfileReport:
        if self._results is None:
            assert False, "Run the algorithm first"
        return self._results

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

        self._results = ProfileReport(df, title="Profiling Report", sensitive=False)
        logger.info(f"Generated profiling report for {input_path.name}")

        return self

    def save_result(self, path: Path) -> None:
        if path.is_dir():
            path /= "profiling_report.html"
        self.results.to_file(path)
