import logging

from oceanprotocol_job_details.dataclasses.job_details import JobDetails


logger = logging.getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details

        logger.info("Instantiating the algorithm with details:")
        logger.info(self._job_details)

    def run(self) -> None:
        raise NotImplementedError()
