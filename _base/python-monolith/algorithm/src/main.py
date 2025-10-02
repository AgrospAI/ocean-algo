import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypeVar

from oceanprotocol_job_details.config import config
from oceanprotocol_job_details.ocean import JobDetails
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_ResultType = TypeVar("_ResultType", bound=Any)


@dataclass
class InputParameters:
    name: str
    age: int


class Algorithm:
    def __init__(self, job_details: JobDetails[InputParameters]):
        self._job_details = job_details
        self._results: _ResultType | None = None

    @property
    def results(self) -> _ResultType:
        if self._results is None:
            raise ValueError("No results available. Please run the algorithm first.")
        return self._results

    def _validate_input(self) -> Self:
        if not self._job_details.dids or len(self._job_details.dids) == 0:
            logger.warning("No DIDs found")
            raise ValueError("No DIDs found")

        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> Self:
        raise NotImplementedError()

        self._validate_input()

    def save_results(self, path: Path) -> None:
        raise NotImplementedError()


def main():
    # Load the current job details from the environment variables
    job_details = OceanProtocolJobDetails(InputParameters).load()

    algorithm = Algorithm(job_details)
    try:
        algorithm.run()
    except Exception as e:
        logger.exception(f"An error occurred while running the algorithm: {e}")

    try:
        algorithm.save_result(config.path_outputs)
    except Exception as e:
        logger.exception(f"An error occurred while saving the results: {e}")


if __name__ == "__main__":
    main()
