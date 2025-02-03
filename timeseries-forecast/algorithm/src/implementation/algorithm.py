from logging import getLogger
from pathlib import Path
from typing import Any, Optional

import orjson
import pandas as pd
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from sklearn.ensemble import AdaBoostRegressor

from implementation import estimators, utils
from implementation.window import WindowGenerator

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

        self.window = WindowGenerator(df, "", 3)

        model = AdaBoostRegressor(n_estimators=100, learning_rate=0.05)

        self.results = self.window.train(model)
        return self

    def save_result(self, path: Path) -> None:
        """Save the trained model pipeline to output"""

        pipeline_path = path / "pipe.pkl"
        score_path = path / "scores.csv"
        parameters_path = path / "parameters.json"

        if self.results:
            import cloudpickle

            pipe, scores = self.results
            cloudpickle.register_pickle_by_value(estimators)

            try:
                with open(pipeline_path, "wb") as f:
                    f.write(cloudpickle.dumps(pipe))
                logger.info(f"Saved model to {path}")
            except Exception as e:
                logger.exception(f"Error saving model: {e}")

            try:
                # Save scores into csv
                scores = pd.DataFrame(scores, index=[0])
                scores.to_csv(score_path, index=False)
            except Exception as e:
                logger.exception(f"Error saving scores: {e}")

        try:
            # Save algorithm parameters
            with open(parameters_path, "wb") as f:
                f.write(orjson.dumps(self._job_details.parameters))
        except Exception as e:
            logger.exception(f"Error saving algorithm parameters: {e}")

    @property
    def _df(self) -> pd.DataFrame:
        filepath = self._job_details.files[list(self._job_details.files.keys())[0]][0]
        self._dataset_info, _ = utils.get(self._job_details.parameters, "dataset")
        separator, _ = utils.get(self._dataset_info, "separator", None)

        logger.info(f"Getting input data from file: {filepath}")
        return pd.read_csv(filepath, sep=separator)
