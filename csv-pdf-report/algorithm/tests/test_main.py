import sys
from pathlib import Path
import pandas as pd
import pytest
from pytest import fixture, raises
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

sys.path.append("src")

from typing import Optional
from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from implementation.algorithm import Algorithm
from implementation.data import InputParameters

job_details: Optional[OceanProtocolJobDetails]
algorithm: Optional[Algorithm]

@fixture(scope="session", autouse=True)
def setup():
    global job_details, algorithm
    job_details = OceanProtocolJobDetails(InputParameters).load()
    algorithm = Algorithm(job_details)
    yield

def test_details():
    assert job_details is not None

def test_algorithm_run(tmp_path):
    result = algorithm.run(tmp_path)
    assert result is not None
    assert result.results is not None

def test_output(tmp_path):
    algorithm.save_result(tmp_path)
    assert tmp_path.exists()
