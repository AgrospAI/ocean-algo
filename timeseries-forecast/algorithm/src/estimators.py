from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame, Series, to_datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from .data import PeriodicityT


@dataclass
class Imputer(BaseEstimator, TransformerMixin):
    """Imputes missing values based on a strategy for each column that is decided by it's characteristics."""

    cat_cols: list[str]
    threshold: float = field(default=0.5)
    _num_cols: list[str] = field(default_factory=lambda: [])

    def fit(self, X, y=None):
        X = DataFrame(X)
        self._num_cols.extend(X.select_dtypes(include="number").columns.to_list())
        self.skewness = X[self._num_cols].skew().abs().T
        return self

    def transform(self, X):
        X = DataFrame(X)

        for col in set(self.cat_cols + self._num_cols):
            if col in self.cat_cols:
                value = (
                    X[col].mode(dropna=True).iloc[0]
                    if not X[col].mode(dropna=True).empty
                    else np.nan
                )
            else:
                skew_value = self.skewness.get(col, 0)

                if isinstance(skew_value, Series):
                    skew_value = skew_value.iloc[1]

                strat = "mean" if skew_value < self.threshold else "median"
                value = getattr(X[col], strat)()

            X[col] = X[col].fillna(value)

        return DataFrame(X, columns=X.columns, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return input_features


class ColumnTransformerWithNames(ColumnTransformer):
    """Wraps ColumnTransformer to return a DataFrame with correct column names."""

    def transform(self, X):
        X_transformed = super().transform(X)
        column_names = self.get_feature_names_out()
        return DataFrame(X_transformed, columns=column_names, index=X.index)

    def fit_transform(self, X, y=None):
        X_t = super().fit_transform(X, y)
        return DataFrame(X_t, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features=None):
        return [
            "".join(name.split("__")[1:])
            for name in super().get_feature_names_out(input_features)
        ]


@dataclass
class Periodicity(BaseEstimator, TransformerMixin):

    datetime_col: str
    target_col: str
    periodicity: list[PeriodicityT]

    lags: int = field(default=3)

    _n_drop: int = field(default=0, init=False)

    def fit(self, X, y=None):
        self._n_drop = self.lags
        return self

    def transform(self, X) -> DataFrame:
        X = DataFrame(X).copy()

        # Always ensure datetime column is datetime
        X[self.datetime_col] = to_datetime(X[self.datetime_col])

        # If target_col not in X, skip lag/log features (e.g. in prediction mode)
        if self.target_col in X.columns:
            X[f"log_{self.target_col}"] = np.log(X[self.target_col])

            for i in range(self.lags):
                X[f"{self.target_col}_lag_{i + 1}"] = X[self.target_col].shift(i + 1)
                X[f"log_{self.target_col}_lag_{i + 1}"] = np.log(
                    X[f"{self.target_col}_lag_{i + 1}"]
                )
                X[f"log_diff_{i + 1}"] = (
                    X[f"log_{self.target_col}"]
                    - X[f"log_{self.target_col}_lag_{i + 1}"]
                )

            X.dropna(inplace=True)

        # Generate periodic features (works both for training and prediction)
        ts_sec = X[self.datetime_col].astype(np.int64) // 10**9
        periods = {
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
            "weeks": 7 * 86400,
            "months": 30.4368 * 86400,
            "years": 365.25 * 86400,
        }

        for name in self.periodicity:
            per = periods[name]
            X[f"{name}_sin"] = np.sin(2 * np.pi * ts_sec / per)
            X[f"{name}_cos"] = np.cos(2 * np.pi * ts_sec / per)

        return X.set_index(self.datetime_col)
