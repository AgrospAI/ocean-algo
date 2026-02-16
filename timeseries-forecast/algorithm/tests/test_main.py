from pytest import fixture, raises
from src.algorithm import algorithm, load_data


@fixture()
def model_path():
    yield algorithm.job_details.paths.outputs / "model.pkl"


@fixture()
def model(model_path):
    with open(model_path, "rb") as f:
        import cloudpickle

        yield cloudpickle.load(f)


def test_main():
    algorithm()

    assert algorithm.result is not None
    assert (algorithm.job_details.paths.outputs / "scores.csv").exists()


def test_predict(model):
    df = load_data(algorithm)
    predictions = model.predict(df)

    assert predictions is not None


def test_wrong_predict_data(model):
    with raises((ValueError, KeyError)):
        model.predict({"test": ["data"]})
