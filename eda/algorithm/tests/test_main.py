from typing import Optional

import pytest
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from src.implementation.algorithm import Algorithm

job_details: Optional[OceanProtocolJobDetails]


@pytest.fixture(scope="session", autouse=True)
def setup():
    """Setup code that will run before the first test in this module."""

    global job_details
    job_details = OceanProtocolJobDetails().load()

    yield

    print("End of testing session ...")


# Add actual tests on real algorithm implementations


def test_details():
    global job_details
    assert job_details is not None


def test_main():
    global job_details
    with pytest.raises(NotImplementedError):
        Algorithm(job_details).run()
