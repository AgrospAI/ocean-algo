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
    job_details: JobDetails = JobDetails.load()

    Algorithm(job_details).run().save_result(job_details.paths.outputs / "result")


if __name__ == "__main__":
    main()
