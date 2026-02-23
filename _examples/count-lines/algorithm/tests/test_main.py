from src.algorithm import algorithm


def test_main():
    algorithm()

    assert algorithm.result == 11
    assert len(list(algorithm.job_details.paths.outputs.glob("*"))) == 1
