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
from pathlib import Path
from types import SimpleNamespace

from implementation.algorithm import GenericCsvPdfReportAlgorithm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LocalJobDetails:
    def __init__(self, files):
        self.files = files


def main():
    # Specify your CSV file path here
    csv_path = Path("../_data/inputs/CEP-2021-S1-ENVCOMFORT.csv").resolve()
    job_details = LocalJobDetails([str(csv_path)])

    algorithm = GenericCsvPdfReportAlgorithm(job_details)

    try:
        algorithm.run()
    except Exception as e:
        logger.exception(f"An error occurred while running the algorithm: {e}")

if __name__ == "__main__":
    main()
