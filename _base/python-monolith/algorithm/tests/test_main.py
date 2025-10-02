import sys
from dataclasses import dataclass

# Append relative src directory to path
sys.path.append("src")

from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from pytest import fixture, raises

from src.main import Algorithm

job_details: OceanProtocolJobDetails | None
algorithm: Algorithm | None


@dataclass
class InputParameters:
    name: str
    age: int


@fixture(scope="session", autouse=True)
def setup():
    """Setup code that will run before the first test in this module."""

    global job_details, algorithm

    job_details = OceanProtocolJobDetails(InputParameters).load()
    algorithm = Algorithm(job_details)

    yield

    print("End of testing session ...")


def test_details():
    assert job_details is not None


def test_main():
    with raises(NotImplementedError):
        algorithm.run()


def test_main_results():
    with raises(ValueError):
        algorithm.results


def test_output(tmp_path):
    with raises(NotImplementedError):
        algorithm.save_results(tmp_path)
