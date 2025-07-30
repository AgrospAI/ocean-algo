# =========
# Append the path of the algorithm to the sys.path
#
# This is needed because THIS .py file is downloaded from the given URL, whilst the rest of the implementation is not.
# The rest of the implementation, can (and should) be added in two ways:
# 1. ADD/COPY the implementation source code in the Dockerfile provided later to the dataspace.
# 2. Mounted as a volume for quick development iterations.
#
# This step is not needed if this file contains the whole implementation of your algorithm, in which case
# you could use the `python-monolith` version.
import sys
from pathlib import Path

sys.path.append("/algorithm/src")
# ======

import logging

from oceanprotocol_job_details.config import config
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails

from implementation.algorithm import Algorithm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    job_details = OceanProtocolJobDetails().load()

    logger.info("Starting compute job with the following input information:")
    algorithm = Algorithm(job_details)

    try:
        algorithm.run()
    except Exception as e:
        logger.exception(f"An error occurred while running the algorithm: {e}")

    try:
        algorithm.save_result(Path(config.path_outputs) / "profiling_report.html")
    except Exception as e:
        logger.exception(f"An error occurred while saving the results: {e}")


if __name__ == "__main__":
    main()
