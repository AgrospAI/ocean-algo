import os
from src.algorithm import algorithm


def test_main():
    algorithm()
    
    assert os.listdir(algorithm.job_details.paths.outputs)

