from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Self

import cloudpickle
import orjson
import pandas as pd
from oceanprotocol_job_details.ocean import JobDetails
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils import all_estimators

from implementation import estimators
from implementation.data import InputParameters
from implementation.store.impl.fs_store import FileSystemStore
from implementation.window import WindowGenerator

logger = getLogger(__name__)


@dataclass(frozen=True)
class ResultType:
    window_pipeline: Pipeline
    model: BaseEstimator
    metrics: dict[str, float]


@dataclass
class Algorithm:
    job_details: JobDetails[InputParameters]
    _results: ResultType | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        assert self.job_details.files, "No files found"
        assert self.job_details.input_parameters, "No input parameters found"

    def run(self) -> Self:
        """The algorithm entry point. This method does the following:

        1. Load the input data from the given files.
        1. Preprocess the data using a scikit-learn pipeline.
        1. Train the model using the preprocessed data.
        1. Evaluate the model using the test data.

        """

        # Loads the input data from the given files
        df = self._df
        logger.info(f"Data shape: {df.shape}")
        logger.debug(f"Data head: \n{df.head()}")

        # Window generator in charge of splitting the data and preprocessing it
        self.window = WindowGenerator(df, self.job_details.input_parameters)
        X_train, X_test, y_train, y_test = self.window.preprocess()

        # Get the scikit-learn model
        model = self._model
        self.window.train(X_train, y_train, model)
        evaluation_results = self.window.evaluate(
            model,
            X_test,
            y_test,
            self.job_details.input_parameters.model.metrics,
        )

        self.results = ResultType(
            window_pipeline=self.window.timeseries_pipeline,
            model=model,
            metrics=evaluation_results,
        )

        return self

    def save_result(self, path: Path) -> None:
        """Save the trained model pipeline to output"""

        timeseries_pipeline_path = path / "timeseries_features.pkl"
        model_pipeline_path = path / "model.pkl"
        score_path = path / "scores.csv"
        parameters_path = path / "parameters.json"
        plotting_path = path / "plot.png"

        fs_store = FileSystemStore()

        fs_store.store(
            parameters_path,
            lambda f: f.write(orjson.dumps(self.job_details.input_parameters)),
        )

        if self.results:

            cloudpickle.register_pickle_by_value(estimators)

            fs_store.store(
                # === Save timeseries preprocessing pipeline ===
                timeseries_pipeline_path,
                lambda f: cloudpickle.dump(self.results.window_pipeline, f),
            ).store(
                # === Save algorithm resulting pipeline ===
                model_pipeline_path,
                lambda f: cloudpickle.dump(self.results.model, f),
            ).store(
                # === Save scores to CSV ===
                score_path,
                lambda f: pd.DataFrame(self.results.metrics, index=[0]).to_csv(
                    f, index=False
                ),
            ).store(
                # === Save periodicity plot ===
                plotting_path,
                lambda f: self.window.save_figure(plotting_path),
            )

    @property
    def _df(self) -> pd.DataFrame:
        # Right now we only support passing one DID with one file.
        try:
            filepath = self.job_details.files[0].input_files[0]
        except IndexError:
            logger.error("No input files found")
            raise ValueError("No input files found")

        logger.info(f"Getting input data from file: {filepath}")
        return pd.read_csv(
            filepath,
            sep=self.job_details.input_parameters.dataset.separator,
            index_col=0,
        )

    @cached_property
    def _model(self) -> BaseEstimator:
        """Returns an untrained instance of the specified scikit-learn model."""

        model = self.job_details.input_parameters.model
        logger.info(f"Creating model: {model}")

        estimators = {estimator[0]: estimator[1] for estimator in all_estimators()}
        if model.name not in estimators:
            raise ValueError(f"Model {model} not found in scikit-learn estimators")

        return estimators[model.name](**model.parameters)
