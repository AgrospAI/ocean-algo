from logging import getLogger
from typing import Sequence

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
