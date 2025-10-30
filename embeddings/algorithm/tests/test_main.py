import os

from src.algorithm import algorithm


def test_main():
    algorithm()

    out_files = os.listdir(algorithm.job_details.paths.outputs)

    # Flat input
    in_files = [
        os.listdir(algorithm.job_details.paths.inputs / f)
        for f in os.listdir(algorithm.job_details.paths.inputs)
        if os.path.isdir(algorithm.job_details.paths.inputs / f)
    ]
    in_files = [item for sublist in in_files for item in sublist]

    assert len(out_files) == len(in_files)
