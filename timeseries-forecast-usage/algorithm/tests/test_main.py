import os
import pandas as pd
from src.algorithm import algorithm


def test_algorithm():
    algorithm()
    assert isinstance(algorithm.result, pd.DataFrame)
    assert os.listdir(algorithm.job_details.paths.outputs), "Did not save output"
