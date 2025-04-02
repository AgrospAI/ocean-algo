from dataclasses import dataclass
from logging import getLogger
from typing import Self, Sequence

from numpy import cos, log, pi, sin
from pandas import DataFrame, Timestamp, to_datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

logger = getLogger(__name__)


class Imputer(BaseEstimator, TransformerMixin):
    """Imputes missing values based on a strategy for each column that is decided by it's characteristics."""

    def __init__(
        self,
        datetime_column: str,
        categorical_columns: Sequence[str],
        numeric_columns: Sequence[str],
        threshold: float = 0.5,
    ) -> None:
        self.datetime_column = datetime_column
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.threshold = threshold

    def _strategy(self, col: str) -> str:
        # If value is categorical, fill with most frequent value (mode)
        if col in self.categorical_columns:
            return "mode"
        if col not in self.skewness:
            logger.warning(f"Column {col} not found in skewness")
            return "mode"
        return "mean" if self.skewness.get(col, 0) < self.threshold else "median"

    def fit(self, X, y=None):
        X = DataFrame(X)
        self.skewness = X[self.numeric_columns].skew().abs().T
        return self

    def transform(self, X):
        X = DataFrame(X) if not isinstance(X, DataFrame) else X
        for col in set([x for x in self.categorical_columns + self.numeric_columns]):
            strat = self._strategy(col)
            value = getattr(X[col], strat)()
            X[col] = X[col].fillna(value)
            logger.info(f"Imputed column {col} with: {value} [strategy: {strat}]")

        logger.info("Imputation transformation done")
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class ColumnTransformerWithNames(ColumnTransformer):
    """Wraps ColumnTransformer to return a DataFrame with correct column names."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = super().transform(X)

        column_names = self.get_feature_names_out()
        logger.info(f"Column transformation done with columns {column_names}")
        return DataFrame(X_transformed, columns=column_names, index=X.index)

    def get_feature_names_out(self, input_features=None):
        column_names = super().get_feature_names_out(input_features)
        return ["".join(name.split("__")[1:]) for name in column_names]


# https://stackoverflow.com/questions/63517126/any-way-to-predict-monthly-time-series-with-scikit-learn-in-python
@dataclass(frozen=True)
class Periodicity(BaseEstimator, TransformerMixin):
    """Processes the datetime column to extract extra features such as its' variance and mean to help forecasting processes."""

    datetime_column: str
    """The name of the datetime column in the DataFrame."""

    target_column: str
    """The name of the target column in the DataFrame."""

    periodicity: Sequence[str]
    """The periodicity to calculate (day, week, month, year)."""

    lags: int = 3
    """The number of lags to calculate (steps into the past)."""

    def fit(self, X, y=None) -> Self:
        return self

    def transform(self, X) -> DataFrame:
        X = DataFrame(X) if not isinstance(X, DataFrame) else X
        X = X.copy()

        # Calculates some extra features based on the target column

        # Logarithm of the target column
        # Needs that the target column is positive (MinMax before)
        X[f"log_{self.target_column}"] = log(X[self.target_column])

        # Check past values
        for i in range(self.lags):
            # Add previous values of the target column
            X[f"{self.target_column}_lag_{i + 1}"] = X[self.target_column].shift(i + 1)

            X.dropna(inplace=True)

            # Add logarithm values of lags
            X[f"log_{self.target_column}_lag_{i + 1}"] = log(
                X[f"{self.target_column}_lag_{i + 1}"]
            )

            X[f"log_diff_{i + 1}"] = (
                X[f"log_{self.target_column}"]
                - X[f"log_{self.target_column}_lag_{i + 1}"]
            )

        rate = lambda timestamp, period: timestamp * 2 * pi / period  # noqa

        day_s = 24 * 60 * 60
        periods = {
            "day": 1,
            "week": 7,
            "month": 30.4368,
            "year": 365.25,
        }

        try:
            # Also, add some periodicity features
            X[self.datetime_column] = to_datetime(X[self.datetime_column])
            timestamp_s = X[self.datetime_column].map(Timestamp.timestamp)

            try:
                for name in self.periodicity:
                    period = periods[name] * day_s
                    X[f"{name}_sin"] = timestamp_s.apply(lambda x: sin(rate(x, period)))
                    X[f"{name}_cos"] = timestamp_s.apply(lambda x: cos(rate(x, period)))
            except ValueError:
                pass
        except Exception as e:
            logger.error(f"Error processing periodicity: {e}")

        logger.info("Periodicity processing done")
        return X.set_index(self.datetime_column)
