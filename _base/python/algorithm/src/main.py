import logging

from implementation.algorithm import Algorithm
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    # Load the current job details from the environment variables
    job_details: JobDetails = OceanProtocolJobDetails().load()

    Algorithm(job_details)  # .run()


if __name__ == "__main__":
    main()
