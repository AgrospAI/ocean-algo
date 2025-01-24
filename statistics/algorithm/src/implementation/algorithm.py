from logging import getLogger
from pathlib import Path
from typing import Optional

import pandas as pd

from oceanprotocol_job_details.dataclasses.job_details import JobDetails

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[pd.DataFrame] = None

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
        
        df = pd.read_csv(filename)
        self.results = df.describe(include="all")
        
        logger.info(f"Descriptive statistics for {filename}: \n {self.results}")
        
        return self
    

    def save_result(self, path: Path) -> None:
        if self.results is None:
            raise ValueError("No results")
        self.results.to_csv(path)