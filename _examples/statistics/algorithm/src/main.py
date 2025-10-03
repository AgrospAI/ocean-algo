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

sys.path.append("/algorithm/src")
# ======


import logging

from implementation.algorithm import Algorithm
from oceanprotocol_job_details.ocean import JobDetails

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Load the current job details from the environment variables
    job_details: JobDetails = JobDetails().load()

    Algorithm(job_details).run().save_result(
        job_details.paths.path_outputs / "result.csv"
    )


if __name__ == "__main__":
    main()
