import os
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from oceanprotocol_job_details.dataclasses.constants import Paths
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from orjson import dumps

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_ResultType = str


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: _ResultType = ""

    def _explore(self, start_path: Path) -> str:
        result = ""
        for root, _, files in os.walk(start_path):
            level = root.replace(start_path, "").count(os.sep)
            indent = " " * 4 * (level)
            result += f"{indent}{os.path.basename(root)}/\n"
            subindent = " " * 4 * (level + 1)
            for f in files:
                result += f"{subindent}{f}\n"
        return result

    def run(self) -> "Algorithm":
        """Inspects the structure of the input DIDs, DDOs.

        :return: executed algorithm
        :rtype: Algorithm
        """

        for path, value in asdict(Paths).items():
            if not isinstance(value, Path):
                continue

            logger.info(f"Listing files in {path}:")
            self._explore(value)

        return self

    def save_result(self, path: Path) -> None:
        with open(path / "exploration.txt", "w") as file:
            file.write(self.results)


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
