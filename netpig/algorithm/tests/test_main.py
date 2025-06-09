import sys

# Append relative src directory to path
sys.path.append("src")

from typing import Optional

from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from pytest import fixture, raises
from src.implementation.algorithm import Algorithm
from pathlib import Path
from oceanprotocol_job_details.config import config

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


def test_main():
        algorithm.run()


def test_main_results():
    assert algorithm.results is not None


def test_output(tmp_path):
    print(Path(config.path_outputs))
    algorithm.save_result(Path(config.path_outputs))
