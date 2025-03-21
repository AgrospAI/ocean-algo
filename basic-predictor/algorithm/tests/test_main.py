import sys

# Append relative src directory to path
sys.path.append("src")

from pathlib import Path
from typing import Optional

from oceanprotocol_job_details.job_details import OceanProtocolJobDetails
from pytest import fixture, mark
from src.implementation.algorithm import Algorithm

job_details: Optional[OceanProtocolJobDetails]
algorithm: Optional[Algorithm]


@fixture(scope="session", autouse=True)
def setup():
    """Setup code that will run before the first test in this module."""

    global job_details, algorithm

    job_details = OceanProtocolJobDetails().load()
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
    tmp = Path(tmp_path)

    algorithm.save_result(tmp_path)

    assert (tmp / "pipe.pkl").exists()
    assert (tmp / "scores.csv").exists()
    assert (tmp / "parameters.json").exists()


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
    model = load_model(tmp / "pipe.pkl")

    # Load the data
    df: pd.DataFrame = algorithm._df

    # Hardcoded because the end-user should know :)
    target_column = "species"
    X_df = df.drop(columns=[target_column])

    # Make predictions
    try:
        predictions = model.predict(X_df)
    except Exception as e:
        raise Exception("Error predicting") from e

    assert predictions is not None
