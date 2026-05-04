from src.algorithm import algorithm


def test_main():
    algorithm()

    assert len(list(algorithm.job_details.paths.outputs.glob("*"))), "No outputs"
