import cloudpickle
from ocean_runner import Algorithm, Config
from pytest import fixture, raises

from src.implementation.algorithm import load_data, run, save_results
from src.implementation.data import InputParameters


@fixture(scope="session", autouse=True)
def algorithm():
    algorithm = Algorithm(Config(InputParameters))

    yield algorithm


@fixture()
def model_path(algorithm):
    yield algorithm.job_details.paths.outputs / "model.pkl"


@fixture()
def model(algorithm, model_path):
    algorithm.save_results(save_results)

    with open(model_path, "rb") as f:
        yield cloudpickle.load(f)


def test_details(algorithm):
    assert algorithm.job_details is not None


def test_main(algorithm):
    assert algorithm.run(run) is not None


def test_main_results(algorithm):
    assert algorithm.result is not None


def test_output(algorithm, model_path):
    assert (algorithm.job_details.paths.outputs / "scores.csv").exists()
    assert (model_path).exists()


def test_predict(algorithm, model):
    df = load_data(algorithm)
    predictions = model.predict(df)

    assert predictions is not None


def test_wrong_predict_data(model):
    with raises((ValueError, KeyError)):
        model.predict({"test": ["data"]})
