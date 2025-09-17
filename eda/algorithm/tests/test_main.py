from pathlib import Path
from typing import Optional

from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from oceanprotocol_job_details.config import config
from pytest import fixture, raises
from src.implementation.algorithm import Algorithm

job_details: Optional[OceanProtocolJobDetails]
algorithm: Optional[Algorithm]


@fixture(scope="session", autouse=True)
def setup():
    """Setup code that will run before the first test in this module."""

    global job_details, algorithm

    job_details = OceanProtocolJobDetails().load()
    algorithm = Algorithm(job_details)

    yield

    print("End of testing session ...")


def test_details():
    assert job_details is not None


def test_main_results():
    algorithm.run()
    assert algorithm.results is not None


def test_output():
    algorithm.save_result(Path(config.path_outputs) / "profiling_report.html")
    assert (Path(config.path_outputs) / "profiling_report.html").exists()
