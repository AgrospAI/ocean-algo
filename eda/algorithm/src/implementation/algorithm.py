import logging
from pathlib import Path

import pandas as pd
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from pandas_profiling import ProfileReport

logger = logging.getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self._profile = None

    def run(self) -> "Algorithm":
        first_did = self._job_details.dids[0]
        filename = self._job_details.files[first_did][0]

        df = pd.read_csv(filename, sep=None)

        self._profile = ProfileReport(df, title="Profiling Report", sensitive=False)
        logger.info(f"Generated profiling report for {filename}")

        return self

    def save_result(self, path: Path) -> None:
        self._profile.to_file(path)
