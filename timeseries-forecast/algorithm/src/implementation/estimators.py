from logging import getLogger
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

logger = getLogger(__name__)

_Strategy = Literal["most_frequent", "mean", "median"]


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_columns: Sequence[str],
        numeric_columns: Sequence[str],
        skewness_threshold: float = 0.5,
    ):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.skewness_threshold = skewness_threshold

    def _strategy(self, col: str) -> _Strategy:
        # If value is categorical, fill with most frequent value (mode)
        if col in self.categorical_columns:
            return "mode"
        if col not in self.skewness:
            logger.warning(f"Column {col} not found in skewness")
            return "mean"

        return (
            "mean" if self.skewness.get(col, 0) < self.skewness_threshold else "median"
        )

    def fit(self, X, y=None):
        # Calculate skewness of df
        X = pd.DataFrame(X)
        self.skewness = X[self.numeric_columns].skew().abs().T

        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in X.columns:
            X[col] = X[col].fillna(getattr(X[col], self._strategy(col)))

        logger.info("Imputation transformation done")
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class ColumnTransformerWithNames(ColumnTransformer):
    """Wraps ColumnTransformer to return a DataFrame with correct column names."""

    def transform(self, X):
        X_transformed = super().transform(X)

        column_names = self.get_feature_names_out()
        logger.info(f"Column transformation done with columns {column_names}")
        return pd.DataFrame(X_transformed, columns=column_names, index=X.index)

    def get_feature_names_out(self, input_features=None):
        column_names = super().get_feature_names_out(input_features)
        return ["".join(name.split("__")[1:]) for name in column_names]


class Log(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        res = pd.DataFrame()

        for col in X.columns:
            if X.dtypes[col] is np.number:
                min_value = X[col].min()
                if min_value <= 0:
                    logger.warning(f"Column {col} has negative values")
                    continue

                res[col] = X[col]
                res[f"{col}_log"] = np.log(X[col])
            else:
                res[col] = X[col]

        logger.info("Logarithm transformation done")

        return res


class Lagger(BaseEstimator, TransformerMixin):
    def __init__(self, lag: int):
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for i in range(self.lag):
            for col in X.columns:
                X[f"{col}_lag_{i + 1}"] = X[col].shift(i + 1)

        logger.info("Lagging done")

        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class LogDifference(BaseEstimator, TransformerMixin):
    """
    Calculates the difference between the logarithmic values of the target column and the previous values.

    https://stackoverflow.com/questions/63517126/any-way-to-predict-monthly-time-series-with-scikit-learn-in-python
    """

    def __init__(self, lag: int, target_column: str):
        self.lag = lag
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ensure column exists
        log_column = f"{self.target_column}_log"
        if log_column not in X.columns:
            logger.warning(f"Column {log_column} not found")
            return X

        for i in range(1, self.lag + 1):
            lagged_column = f"{log_column}_lag_{i}"
            if lagged_column not in X.columns:
                logger.warning(
                    f"Expected lagged column '{lagged_column}', but it was not found."
                )
                continue

            # Calculate the difference from previous logarithmic values
            X[f"{log_column}_diff_{i}"] = X[log_column] - X[lagged_column]

        logger.info("Logarithmic difference done")

        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class Stationary(BaseEstimator, TransformerMixin):
    """Adds stationary information to the dataset"""

    def __init__(self, datetime_column: str):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_column] = pd.to_datetime(X[self.datetime_column])
        X = X.set_index(self.datetime_column)

        X_diff = X.diff().dropna()
        X_diff.columns = [f"{col}_diff" for col in X_diff.columns]

        logger.info("Stationary transformation done")
        return X_diff
