import pandas as pd

from src.algorithm import algorithm


def test_main():
    algorithm()

    assert algorithm.job_details.paths.outputs.iterdir()

    assert isinstance(algorithm.result, list)

    res0 = algorithm.result[0]

    assert isinstance(res0, tuple)

    did, result = res0

    assert isinstance(did, str)
    assert isinstance(result, pd.DataFrame)
