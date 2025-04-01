import sys
from pathlib import Path

from oceanprotocol_job_details.ocean import JobDetails

# Append relative src directory to path
sys.path.append("src")

import warnings
from typing import Optional

from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from pytest import fixture, mark
from src.implementation.algorithm import Algorithm
from src.implementation.data import InputParameters

warnings.filterwarnings("ignore", category=FutureWarning)

job_details: JobDetails[InputParameters]
algorithm: Optional[Algorithm]


@fixture(scope="session", autouse=True)
def setup():
    """Setup code that will run before the first test in this module."""

    global job_details, algorithm

    job_details = OceanProtocolJobDetails(InputParameters).load()
    algorithm = Algorithm(job_details)

    yield

    print("End of testing session ...")


def test_details():
    assert job_details is not None


def test_main():
    assert algorithm.run() is not None


def test_main_results():
    assert algorithm.results is not None


def test_output(tmp_path):
    """Test that the algorithm output contains the necessary files to make predictions
    and that it's possible to load the model and make predictions.
    """

    tmp = Path(tmp_path)
    algorithm.save_result(tmp)

    assert (tmp / "parameters.json").exists()
    assert (tmp / "scores.csv").exists()
    assert (tmp / "timeseries_features.pkl").exists()
    assert (tmp / "model.pkl").exists()
    assert (tmp / "plot.png").exists()


@mark.filterwarnings("ignore::FutureWarning")
@mark.filterwarnings("error")
def test_can_predict(tmp_path):
    import pandas as pd

    def load_model(path: Path):
        import cloudpickle

        with open(path, "rb") as f:
            return cloudpickle.load(f)

    tmp = Path(tmp_path)
    algorithm.save_result(tmp)

    # Load the pipelines
    ts_features = load_model(tmp / "timeseries_features.pkl")
    model = load_model(tmp / "model.pkl")

    # Load the data
    df: pd.DataFrame = algorithm._df

    try:
        df = ts_features.transform(df)
    except Exception as e:
        raise Exception("Error transforming the data") from e

    # Hardcoded because the end-user should know :)
    target_column = "Sales"
    X_df = df.drop(columns=[target_column])

    # Make predictions
    try:
        predictions = model.predict(X_df.to_numpy())
    except Exception as e:
        raise Exception("Error predicting") from e

    assert predictions is not None
