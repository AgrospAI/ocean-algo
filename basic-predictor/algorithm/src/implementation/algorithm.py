from logging import getLogger
from pathlib import Path
from typing import Mapping, Optional, Tuple

import pandas as pd
from implementation.estimators import Imputer
from implementation.utils import get
from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from orjson import JSONDecodeError, dumps, loads
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
        """ "
        Validate that the input data is correct, this means:
            1. There are given DIDs to train with.
            2. There are found files for those DIDs.
        """

        if not self._job_details.dids or len(self._job_details.dids) == 0:
            logger.warning("No DIDs found")
            raise ValueError("No DIDs found")

        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> "Algorithm":
        """
        Run the algorithm.

        This method performs:
            1. Loads the input data into a pandas.DataFrame.
            2. Splits the data into training and testing sets.
            3. Fits the preprocessing pipeline with the training data.
            4. Fits the predictor with the preprocessed training data, and combines them into a single sklearn Pipeline object.
            5. Evaluates the model with the testing data.
        """

        self._validate_input()

        #   1. Loads the input data into a pandas.DataFrame.
        df = self._df
        logger.info(f"Loaded data with shape: {df.shape}")
        logger.info(f"Dataset columns: {df.columns}")

        #   2. Splits the data into training and testing sets.
        X_train, X_test, y_train, y_test = self._split(df)

        #   3. Fits the preprocessing pipeline with the training data.
        preprocessor = clone(self._preprocessor)
        preprocessor.fit(X_train)

        # Ensure the preprocessed data retains feature names
        X_train_transformed = preprocessor.transform(X_train)

        #   4. Fits the predictor with the preprocessed training data, and combines them into a single sklearn Pipeline object.
        predictor = clone(self._predictor)
        predictor.fit(X_train_transformed, y_train)

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("predictor", predictor),
            ]
        )

        #   5. Evaluates the model with the testing data.
        try:
            pipeline.predict(X_train)
            score = self._scores(pipeline, X_test, y_test)
            self.results = (pipeline, score)
        except Exception as e:
            logger.exception(f"Error evaluating model: {e}")
            self.results = (pipeline, None)

        return self

    def save_result(self, path: Path) -> None:
        """Save the trained model pipeline to output"""

        pipeline_path = path / "pipe.pkl"
        score_path = path / "scores.csv"
        parameters_path = path / "parameters.json"

        # === Save algorithm run parameters ===
        with open(parameters_path, "wb") as f:
            try:
                f.write(dumps(self._job_details.parameters))
            except Exception as e:
                logger.exception(f"Error saving algorithm parameters: {e}")

        if self.results:
            import cloudpickle
            import implementation.estimators as estimators

            cloudpickle.register_pickle_by_value(estimators)

            # If only the pipeline is saved, save the pipeline
            try:
                pipe, scores = self.results
            except Exception:
                pipe = self.results
                scores = None

            # === Save algorithm resulting pipeline ===
            try:
                with open(pipeline_path, "wb") as f:
                    f.write(cloudpickle.dumps(pipe))
                logger.info(f"Saved model to {path}")
            except Exception as e:
                logger.exception(f"Error saving model: {e}")

            # === Save algorithm pipeline scores ===
            try:
                scores = pd.DataFrame(scores, index=[0])
                scores.to_csv(score_path, index=False)
            except Exception as e:
                logger.exception(f"Error saving scores: {e}")

    @property
    def _preprocessor(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    Imputer(
                        categorical_columns=self._categorical_features,
                    ),
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
        self._model_info = get(self._job_details.parameters, "model")

        if isinstance(self._model_info, str):
            try:
                self._model_info = loads(self._model_info)
            except JSONDecodeError as e:
                logger.error(f"Model info {self._model_info}")
                logger.error(f"Error decoding dataset info: {e}")

        model_name = get(self._model_info, "name")
        model_params = get(self._model_info, "params", {})
        logger.info(f"Creating model: {model_name} with params: {model_params}")

        estimators = {est[0]: est[1] for est in all_estimators()}
        if model_name in estimators:
            return estimators[model_name](**model_params)

        raise ValueError(f"Unknown scikit-learn model: {model_name}")

    def _split(self, df: pd.DataFrame) -> list:
        target_column = get(self._dataset_info, "target_column")
        if not isinstance(target_column, str):
            if isinstance(target_column, list) and len(target_column) == 1:
                target_column = target_column[0]
            else:
                raise ValueError("Target column must be a single string")

        random_state = get(self._dataset_info, "random_state", 42)
        split = get(self._dataset_info, "split", 0.7)
        stratify = get(self._dataset_info, "stratify", False)

        try:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        except KeyError as e:
            logger.error(f"Column {target_column} does not exist in dataset!!")
            raise e

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
        self._dataset_info = get(self._job_details.parameters, "dataset")

        if isinstance(self._dataset_info, str):
            try:
                self._dataset_info = loads(self._dataset_info)
            except JSONDecodeError as e:
                logger.error(f"Dataset info {self._dataset_info}")
                logger.error(f"Error decoding dataset info: {e}")

        separator = get(self._dataset_info, "separator", ",")

        logger.info(f"Getting input data from file: {filepath}")
        return pd.read_csv(filepath, sep=separator)

    def _scores(self, pipe: Pipeline, X_test, y_test) -> Mapping[str, float]:
        metric_names = get(self._model_info, "metrics", [])
        scores = {}
        for metric in metric_names:
            name, params = metric, {}
            if type(metric) is dict:
                name = get(metric, "name")
                params = get(metric, "params", {})

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
