import pandas as pd
from ocean_runner import Algorithm, Config
from pytest import fixture

from src.implementation import InputParameters, run, save_data, validate


@fixture(scope="session", autouse=True)
def algorithm():
    algorithm = Algorithm(
        Config(custom_input=InputParameters),
    )

    yield algorithm


def test_validation(algorithm):
    assert algorithm.validate(validate)


def test_main(algorithm):
    algorithm.run(run)


def test_output(algorithm, tmp_path):
    algorithm.save_results(save_data, override_path=tmp_path)


def test_main_result(algorithm):
    assert algorithm.result is not None
    assert isinstance(algorithm.result, pd.DataFrame)
