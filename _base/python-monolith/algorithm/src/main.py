import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from oceanprotocol_job_details.dataclasses.constants import Paths
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from orjson import dumps

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_ResultType = Any


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[_ResultType] = None

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


def main():
    # Load the current job details from the environment variables
    job_details: JobDetails = OceanProtocolJobDetails().load()

    logger.info("Starting compute job with the following input information:")
    logger.info(dumps({k: str(v) for k, v in asdict(job_details).items()}))

    algorithm = Algorithm(job_details)

    try:
        algorithm.run()
    except Exception as e:
        logger.exception(f"An error occurred while running the algorithm: {e}")

    try:
        algorithm.save_result(Paths.OUTPUTS)
    except Exception as e:
        logger.exception(f"An error occurred while saving the results: {e}")


if __name__ == "__main__":
    main()
