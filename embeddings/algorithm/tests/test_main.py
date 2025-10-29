import os

from src.algorithm import algorithm


def test_main():
    algorithm()

    in_files = os.listdir(algorithm.job_details.paths.inputs)
    out_files = os.listdir(algorithm.job_details.paths.outputs)

    assert len(out_files) == len(in_files)
