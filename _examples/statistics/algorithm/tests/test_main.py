import pandas as pd
import os
from src.algorithm import algorithm


def test_main():
    algorithm()

    assert isinstance(algorithm.result, pd.DataFrame)
    assert os.listdir(algorithm.job_details.paths.outputs)
