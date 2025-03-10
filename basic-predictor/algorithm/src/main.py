import sys
import os

# Append current directory to the path
sys.path.append("/algorithm")

for root, dirs, files in os.walk("/algorithm"):
    level = root.replace("..", "").count(os.sep)
    indent = " " * 4 * (level)
    print("{}{}/".format(indent, os.path.basename(root)))
    subindent = " " * 4 * (level + 1)
    for f in files:
        print("{}{}".format(subindent, f))


import logging
from dataclasses import asdict
from pathlib import Path

import orjson
from oceanprotocol_job_details.dataclasses.constants import Paths
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails

from implementation.algorithm import Algorithm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Load the current job details from the environment variables
    job_details: JobDetails = OceanProtocolJobDetails().load()

    logger.info("Starting compute job with the following input information:")
    logger.info(orjson.dumps({k: str(v) for k, v in asdict(job_details).items()}))

    algorithm = Algorithm(job_details)

    try:
        algorithm.run()
    except Exception as e:
        logger.exception(f"An error occurred while running the algorithm: {e}")

    try:
        algorithm.save_result(Path.joinpath(Path("/"), Paths.OUTPUTS))
    except Exception as e:
        logger.exception(f"An error occurred while saving the results: {e}")


if __name__ == "__main__":
    main()
