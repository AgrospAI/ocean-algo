import json
import logging
from dataclasses import asdict

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
    logger.info(
        json.dumps(
            {k: str(v) for k, v in asdict(job_details).items()},
            sort_keys=True,
            indent=4,
        )
    )

    Algorithm(job_details).run().save_result(Paths.OUTPUTS / "result")


if __name__ == "__main__":
    main()
