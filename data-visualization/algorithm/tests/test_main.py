from src.algorithm import algorithm


def test_main():
    algorithm()

    assert len(algorithm.job_details.paths.outputs.glob("*")), "There are no outputs"
