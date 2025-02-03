from logging import getLogger
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

logger = getLogger(__name__)


class Imputer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        categorical_columns: Sequence[str],
        skewness_threshold: float = 0.5,
    ):
        self.categorical_columns = categorical_columns
        self.skewness_threshold = skewness_threshold
        self._imputers = {}

    def fit(self, X, y=None):
        # Analyze the columns and fill the missing values with the proper strategy
        skewness = pd.DataFrame(X.skew().abs()).T

        for col in X.columns:
            # If value is categorical, fill with most frequent value (mode)
            if col in self.categorical_columns:
                imputer = SimpleImputer(strategy="most_frequent")
            elif skewness[col][0] < self.skewness_threshold:
                # If the column is normally distributed, fill with mean
                imputer = SimpleImputer(strategy="mean")
            else:
                # If the column is skewed, fill with median
                imputer = SimpleImputer(strategy="median")

            logger.info(f"Fitting `{imputer.strategy}` imputer for column {col}")
            imputer.fit(X[col].values.reshape(-1, 1))
            self._imputers[col] = imputer

        return self

    def transform(self, X):
        X = X.copy()
        for col, imputer in self._imputers.items():
            X[col] = imputer.transform(X[col].values.reshape(-1, 1)).ravel()
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class Log(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in X.columns:
            # Check if col is numeric and any is negative
            if X.dtypes[col] in [np.float64, np.int64] and X[col].min() > 0:
                X[f"{col}_log"] = np.log(X[col])

        return X


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

        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class LogDifference(BaseEstimator, TransformerMixin):
    """
    Calculates the difference between the logarithmic values of the target column and the previous values.

    https://stackoverflow.com/questions/63517126/any-way-to-predict-monthly-time-series-with-scikit-learn-in-python
    """

    def __init__(self, lags: int, target_column: str):
        self.lags = lags
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for i in range(self.lags):
            # Calculate the difference from previous logarithmic values
            X[f"{self.target_column}_diff_{i + 1}"] = (
                X[self.target_column] - X[f"{self.target_column}_{i + 1}"]
            )

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
        X = X.diff().dropna()
        return X
