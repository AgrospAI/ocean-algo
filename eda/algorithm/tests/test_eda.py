from pathlib import Path
from typing import Optional

from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from pytest import fixture
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


def test_eda():
    assert algorithm.run()


def test_eda_results():
    assert algorithm.results is not None


def test_output(tmp_path):
    algorithm.save_result(tmp_path / "output.html")

    assert Path(tmp_path / "output.html").exists()
