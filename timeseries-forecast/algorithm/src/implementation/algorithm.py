from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from oceanprotocol_job_details.dataclasses.job_details import JobDetails

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
        raise NotImplementedError()

        # self._validate_input()
        # return self

    def save_result(self, path: Path) -> None:
        raise NotImplementedError()
