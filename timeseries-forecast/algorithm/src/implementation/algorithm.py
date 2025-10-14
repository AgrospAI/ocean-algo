from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cloudpickle
import pandas as pd
from ocean_runner import Algorithm
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils import all_estimators

from . import estimators
from .data import InputParameters
from .window import WindowGenerator


@dataclass(frozen=True)
class ResultType:
    window: WindowGenerator
    pipeline: Pipeline
    metrics: dict[str, float]


def load_data(algo: Algorithm) -> pd.DataFrame:
    _, filepath = next(algo.job_details.next_path())
    input_parameters: InputParameters = algo.job_details.input_parameters
    return pd.read_csv(
        filepath,
        index_col=0,
        sep=input_parameters.data_separator,
        compression=("zip" if input_parameters.data_is_zipped else "infer"),
    )


def get_model(algorithm: Algorithm) -> BaseEstimator:
    """Returns an untrained instance of the specified scikit-learn model."""

    input_parameters: InputParameters = algorithm.job_details.input_parameters

    model = input_parameters.model_name
    algorithm.logger.info(f"Creating model: {model}")

    estimators = {estimator[0]: estimator[1] for estimator in all_estimators()}
    if model not in estimators:
        raise ValueError(f"Model {model} not found in scikit-learn estimators")

    return estimators[model](**input_parameters.model_params)


def run(algorithm: Algorithm) -> ResultType:
    """The algorithm entry point. This method does the following:

    1. Load the input data from the given files.
    1. Preprocess the data using a scikit-learn pipeline.
    1. Train the model using the preprocessed data.
    1. Evaluate the model using the test data.
    """

    input_parameters: InputParameters = algorithm.job_details.input_parameters

    algorithm.logger.info(f"Training with {input_parameters.data_lags} lags")

    # Loads the input data from the given files
    df = load_data(algorithm)
    algorithm.logger.info(f"Data shape: {df.shape}")

    # Window generator in charge of splitting the data and preprocessing it
    window = WindowGenerator(df, algorithm.job_details.input_parameters)

    # Get the scikit-learn model
    model = get_model(algorithm)
    full_pipeline = window.build_full_pipeline(model)
    cv, X_train, X_test, y_train, y_test = window.split()

    full_pipeline.fit(X_train, y_train)

    evaluation_results = {}
    for metric in input_parameters.metrics:
        try:
            evaluation_results[metric] = cross_val_score(
                full_pipeline, X_test, y_test, cv=cv, scoring=metric
            )
        except Exception as e:
            algorithm.logger.error(f"Error computing {metric}: {e}")

    return ResultType(window=window, pipeline=full_pipeline, metrics=evaluation_results)


def save_results(results: ResultType, base_path: Path, algorithm: Algorithm) -> None:
    if not results:
        return

    def store(path: Path, store_function: Callable[[any], None]):
        with open(path, "wb") as f:
            try:
                store_function(f)
            except Exception as e:
                algorithm.logger.info(f"Error in the saving process: {e}")

    cloudpickle.register_pickle_by_value(estimators)
    store(
        base_path / "model.pkl",
        lambda f: cloudpickle.dump(results.pipeline, f),
    )
    store(
        base_path / "scores.csv",
        lambda f: pd.DataFrame(results.metrics).to_csv(f),
    )
