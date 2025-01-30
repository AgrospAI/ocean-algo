from logging import getLogger
from pathlib import Path
from typing import Mapping, Optional, Tuple

import orjson
import pandas as pd
from implementation import estimators, utils
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import all_estimators

logger = getLogger(__name__)

_ResultType = Tuple[Pipeline, Mapping[str, float]]


class Algorithm:

    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[_ResultType] = None

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
        logger.info(f"Loaded data with shape: {df.shape}")

        # Fitting the model & preprocessing
        X_train, X_test, y_train, y_test = self._split(df)

        preprocessor = clone(self._preprocessor)
        preprocessor.fit(X_train)

        # Ensure the preprocessed data retains feature names
        X_train_transformed = preprocessor.transform(X_train)

        predictor = clone(self._predictor)
        predictor.fit(X_train_transformed, y_train)

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("predictor", predictor),
            ]
        )

        self.results = (pipeline, None)
        pipeline.predict(X_train)

        score = self._scores(pipeline, X_test, y_test)
        self.results = (pipeline, score)

        return self

    def save_result(self, path: Path) -> None:
        """Save the trained model pipeline to output"""

        if self.results:
            import cloudpickle

            pipe, scores = self.results
            cloudpickle.register_pickle_by_value(estimators)

            try:
                with open(path / "pipe.pkl", "wb") as f:
                    f.write(cloudpickle.dumps(pipe))
                logger.info(f"Saved model to {path}")
            except Exception as e:
                logger.exception(f"Error saving model: {e}")

            try:
                # Save scores into csv
                scores = pd.DataFrame(scores, index=[0])
                scores.to_csv(path / "scores.csv", index=False)
            except Exception as e:
                logger.exception(f"Error saving scores: {e}")

        try:
            # Save algorithm parameters
            with open(path / "parameters.json", "wb") as f:
                f.write(orjson.dumps(self._job_details.parameters))
        except Exception as e:
            logger.exception(f"Error saving algorithm parameters: {e}")

    @property
    def _preprocessor(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    estimators.Imputer(categorical_columns=self._categorical_features),
                ),
                (
                    "encoding",
                    ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(),
                                self._categorical_features,
                            ),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("preprocessing", StandardScaler()),
            ]
        )

    @property
    def _predictor(self):
        model_info, _ = utils.get(self._job_details.parameters, "model")
        self._model_info = model_info

        model_name, _ = utils.get(model_info, "name")
        model_params, _ = utils.get(model_info, "params", {})

        logger.info(f"Creating model: {model_name} with params: {model_params}")

        estimators = {est[0]: est[1] for est in all_estimators()}
        if model_name in estimators:
            return estimators[model_name](**model_params)

        raise ValueError(f"Unknown scikit-learn model: {model_name}")

    def _split(self, df: pd.DataFrame) -> list:
        target_column, _ = utils.get(self._dataset_info, "target_column")
        if type(target_column) is not str:
            raise ValueError("Target column must be a single string")

        random_state, _ = utils.get(self._dataset_info, "random_state", 42)
        split, _ = utils.get(self._dataset_info, "split", 0.7)
        stratify, _ = utils.get(self._dataset_info, "stratify", False)

        X, y = df.drop(columns=[target_column]), df[target_column]

        # Get numerical and categorical columns
        self._categorical_features = X.select_dtypes(include=["object"]).columns

        return train_test_split(
            X,
            y,
            stratify=y if stratify else None,
            test_size=split,
            random_state=random_state,
        )

    @property
    def _df(self) -> pd.DataFrame:
        filepath = self._job_details.files[list(self._job_details.files.keys())[0]][0]
        self._dataset_info, _ = utils.get(self._job_details.parameters, "dataset")
        separator, _ = utils.get(self._dataset_info, "separator", None)

        logger.info(f"Getting input data from file: {filepath}")
        return pd.read_csv(filepath, sep=separator)

    def _scores(self, pipe: Pipeline, X_test, y_test) -> Mapping[str, float]:
        metric_names, _ = utils.get(self._model_info, "metrics", [])
        scores = {}
        for metric in metric_names:
            name, params = metric, {}
            if type(metric) is dict:
                name, _ = utils.get(metric, "name")
                params, _ = utils.get(metric, "params", {})

            try:
                scorer = get_scorer(name)
            except ValueError:
                logger.warning(f"Metric `{name}` not found, skipping")
                scores[name] = "UNKNOWN"
                continue

            if params:
                try:
                    scorer = make_scorer(scorer._score_func, **params)
                except Exception as e:
                    logger.warning(
                        f"Error creating scorer for `{name}` with params: {params}:  {e}"
                    )
                    continue

            try:
                score = scorer(pipe, X_test, y_test)
            except Exception as e:
                logger.warning(f"Error calculating score for `{name}`: {e}")
                scores[name] = "ERROR"
                continue

            scores[name] = score
            logger.info(f"Score for `{name}`: {score}")

        return scores
