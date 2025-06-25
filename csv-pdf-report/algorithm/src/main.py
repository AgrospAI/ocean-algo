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
import logging

sys.path.append("/algorithm/src")
# ======

from implementation.algorithm import Algorithm
from implementation.data import InputParameters
from oceanprotocol_job_details.config import config
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Get job details based on environment
        job_details = OceanProtocolJobDetails(InputParameters).load()
        output_path = Path(config.path_outputs)
        temporal_path = output_path/"temp"
        temporal_path.mkdir(parents=True, exist_ok=True)

        # Initialize and run algorithm
        algorithm = Algorithm(job_details)
        algorithm.run(temporal_path)

        # Save results
        algorithm.save_result(output_path)
        logger.info(f"Results saved to {output_path}")
            
    except Exception as e:
        logger.exception(f"An error occurred while running the algorithm: {e}")
        raise

if __name__ == "__main__":
    main()
