import logging

from implementation.algorithm import Algorithm
from oceanprotocol_job_details.config import config
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from oceanprotocol_job_details.ocean import JobDetails

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Load the current job details from the environment variables
    job_details: JobDetails = OceanProtocolJobDetails().load()

    Algorithm(job_details).run().save_result(config.path_outputs / "result")


if __name__ == "__main__":
    main()
