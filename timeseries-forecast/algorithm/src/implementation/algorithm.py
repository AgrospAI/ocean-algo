from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

import orjson
import pandas as pd
from implementation import estimators
from implementation.data import InputParameters
from implementation.window import WindowGenerator
from oceanprotocol_job_details.ocean import JobDetails
from sklearn.utils import all_estimators

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails[InputParameters]) -> None:
        self._job_details: JobDetails[InputParameters] = job_details
        self.results: Optional[Any] = None

    def _validate_input(self) -> None:
        assert self._job_details.files, "No files found"
        assert self._job_details.input_parameters, "No input parameters found"

    def run(self) -> "Algorithm":
        """The algorithm entry point. This method does the following:

        1. Load the input data from the given files.
        1. Preprocess the data using a scikit-learn pipeline.
        1. Train the model using the preprocessed data.
        1. Evaluate the model using the test data.

        """

        # Validates the given JobDetails instance
        self._validate_input()

        # Loads the input data from the given files
        df = self._df
        logger.info(f"Data shape: {df.shape}")
        logger.debug(f"Data head: \n{df.head()}")

        # Window generator in charge of splitting the data and preprocessing it
        self.window = WindowGenerator(df, self._job_details.input_parameters)
        X_train, X_test, y_train, y_test = self.window.preprocess()

        # Get the scikit-learn model
        model = self._model
        self.window.train(X_train, y_train, model)
        evaluation_results = self.window.evaluate(
            model,
            X_test,
            y_test,
            self._job_details.input_parameters.model.metrics,
        )

        self.results = (
            self.window.timeseries_pipeline,
            model,
            evaluation_results,
        )
        return self

    def save_result(self, path: Path) -> None:
        """Save the trained model pipeline to output"""

        timeseries_pipeline_path = path / "timeseries_features.pkl"
        model_pipeline_path = path / "model.pkl"
        score_path = path / "scores.csv"
        parameters_path = path / "parameters.json"
        plotting_path = path / "plot.png"

        # === Save algorithm run parameters ===
        with open(parameters_path, "wb") as f:
            try:
                f.write(orjson.dumps(self._job_details.input_parameters))
            except Exception as e:
                logger.exception(f"Error saving algorithm parameters: {e}")

        if self.results:
            import cloudpickle  # type: ignore

            ts_pipe, pipe, scores = self.results
            cloudpickle.register_pickle_by_value(estimators)

            # === Save timeseries preprocessing pipeline ===
            with open(timeseries_pipeline_path, "wb") as f:
                try:
                    cloudpickle.dump(ts_pipe, f)
                    logger.info(f"Saved model to {timeseries_pipeline_path}")
                except Exception as e:
                    logger.exception(f"Error saving model: {e}")

            # === Save algorithm resulting pipeline ===
            with open(model_pipeline_path, "wb") as f:
                try:
                    cloudpickle.dump(pipe, f)
                    logger.info(f"Saved model to {model_pipeline_path}")
                except Exception as e:
                    logger.exception(f"Error saving model: {e}")

            # === Save scores to CSV ===
            try:
                scores = pd.DataFrame(scores, index=[0])
                scores.to_csv(score_path, index=False)
            except Exception as e:
                logger.exception(f"Error saving scores: {e}")

            # === Save periodicity plot ===
            try:
                self.window.save_figure(plotting_path)
            except Exception as e:
                logger.exception(f"Error saving periodicity plot: {e}")

    @property
    def _df(self) -> pd.DataFrame:
        # Right now we only support passing one DID with one file.
        try:
            filepath = self._job_details.files.files[0].input_files[0]
        except IndexError:
            logger.error("No input files found")
            raise ValueError("No input files found")

        logger.info(f"Getting input data from file: {filepath}")
        return pd.read_csv(
            filepath,
            sep=self._job_details.input_parameters.dataset.separator,
            index_col=0,
        )

    @cached_property
    def _model(self) -> Any:
        """Returns an untrained instance of the specified scikit-learn model."""

        model = self._job_details.input_parameters.model
        logger.info(f"Creating model: {model}")

        estimators = {estimator[0]: estimator[1] for estimator in all_estimators()}
        if model.name not in estimators:
            raise ValueError(f"Model {model} not found in scikit-learn estimators")

        return estimators[model.name](**model.parameters)
