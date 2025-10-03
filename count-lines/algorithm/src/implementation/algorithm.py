import subprocess
from logging import getLogger
from pathlib import Path
from typing import Self

from oceanprotocol_job_details.ocean import JobDetails

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails) -> None:
        self._job_details = job_details
        self._results: int = 0

    @property
    def results(self) -> int:
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

        self._results = int(subprocess.check_output(["wc", "-l", filename]).split()[0])
        logger.info(f"Number of non-blank lines found {self._results}")

        return self

    def save_result(self, path: Path) -> None:
        with open(path, "w") as f:
            f.write(str(self.results))
