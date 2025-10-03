from oceanprotocol_job_details.ocean import JobDetails
from pytest import fixture
from src.implementation.algorithm import Algorithm

job_details: JobDetails | None
algorithm: Algorithm | None


@fixture(scope="session", autouse=True)
def setup():
    """Setup code that will run before the first test in this module."""

    global job_details, algorithm

    job_details = JobDetails.load()
    algorithm = Algorithm(job_details)

    yield

    print("End of testing session ...")


def test_details():
    assert job_details is not None


def test_main():
    assert algorithm.run() is not None


def test_main_results():
    assert algorithm.results is not None


def test_output(tmp_path):
    path = tmp_path / "output"

    algorithm.save_result(path)
    assert path.exists()
