from logging import getLogger
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from implementation import estimators
from implementation.utils import get
from implementation.window import WindowGenerator
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from orjson import JSONDecodeError, dumps, loads
from sklearn.ensemble import AdaBoostRegressor

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[Any] = None

    def _validate_input(self) -> "Algorithm":
        if not self._job_details.dids or len(self._job_details.dids) == 0:
            logger.warning("No DIDs found")
            raise ValueError("No DIDs found")

        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> "Algorithm":
        self._validate_input()

        df = self._df
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Data head: \n{df.head()}")

        self.window = WindowGenerator(
            df=df,
            target_column="Sales",
            datetime_column="Date",
        )

        X_train, X_test, y_train, y_test = self.window.preprocess()
        model = AdaBoostRegressor(n_estimators=100, learning_rate=0.05)

        self.window.train(X_train, y_train, model)

        evaluation_results = self.window.evaluate(
            model,
            X_test,
            y_test,
            ["neg_mean_squared_error"],
        )

        self.results = (
            self.window.timeseries_pipeline,
            model,
            evaluation_results,
        )

        logger.info(f"Resulting metrics: {evaluation_results}")

        return self

    def save_result(self, path: Path) -> None:
        """Save the trained model pipeline to output"""

        timeseries_pipeline_path = path / "timeseries_features.pkl"
        model_pipeline_path = path / "model.pkl"
        score_path = path / "scores.csv"
        parameters_path = path / "parameters.json"

        def check_steps(pipeline):
            for name, step in ts_pipe.named_steps.items():
                if hasattr(step, "fit") and not hasattr(step, "transform"):
                    logger.warning(f"⚠️ Warning: {name} might not be fitted!")

        # === Save algorithm run parameters ===
        with open(parameters_path, "wb") as f:
            try:
                f.write(dumps(self._job_details.parameters))
            except Exception as e:
                logger.exception(f"Error saving algorithm parameters: {e}")

        if self.results:
            import cloudpickle

            ts_pipe, pipe, scores = self.results
            cloudpickle.register_pickle_by_value(estimators)

            # === Save timeseries preprocessing pipeline ===
            check_steps(ts_pipe)
            with open(timeseries_pipeline_path, "wb") as f:
                try:
                    cloudpickle.dump(ts_pipe, f)
                    logger.info(f"Saved model to {timeseries_pipeline_path}")
                except Exception as e:
                    logger.exception(f"Error saving model: {e}")

            # === Save algorithm resulting pipeline ===
            check_steps(pipe)
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

    @property
    def _df(self) -> pd.DataFrame:
        filepath = self._job_details.files[list(self._job_details.files.keys())[0]][0]
        self._dataset_info = get(self._job_details.parameters, "dataset")

        if isinstance(self._dataset_info, str):
            try:
                self._dataset_info = loads(self._dataset_info)
            except JSONDecodeError as e:
                logger.error(f"Model info {self._dataset_info}")
                logger.error(f"Error decoding dataset info: {e}")

        separator = get(self._dataset_info, "separator", None)

        logger.info(f"Getting input data from file: {filepath}")
        return pd.read_csv(filepath, sep=separator, index_col=0)
