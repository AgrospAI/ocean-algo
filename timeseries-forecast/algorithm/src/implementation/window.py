from dataclasses import dataclass

from pandas import DataFrame
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .data import InputParameters
from .estimators import (
    ColumnTransformerWithNames,
    Imputer,
    Periodicity,
)


@dataclass
class WindowGenerator:
    df: DataFrame
    params: InputParameters

    def build_full_pipeline(self, model):
        """Build a full pipeline that includes timeseries features, preprocessing and model."""
        cat_cols = [
            c
            for c in self.df.select_dtypes(include="object").columns
            if c != self.params.data_datetime_column
        ]

        num_cols = [
            c
            for c in self.df.select_dtypes(include="number").columns
            if c not in self.params.data_target_column
        ]

        return Pipeline(
            [
                (
                    "periodicity",
                    Periodicity(
                        target_col=self.params.data_target_column,
                        datetime_col=self.params.data_datetime_column,
                        periodicity=self.params.data_periodicity,
                        lags=self.params.data_lags,
                    ),
                ),
                (
                    "imputer",
                    Imputer(
                        cat_cols=cat_cols,
                        threshold=0.5,
                    ),
                ),
                (
                    "encoder",
                    ColumnTransformerWithNames(
                        transformers=[
                            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                            ("num", MinMaxScaler((0, 1)), num_cols),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("model", model),
            ]
        )

    def split(self):
        X = self.df.drop(columns=[self.params.data_target_column])
        y = self.df[self.params.data_target_column]

        ts_cv = TimeSeriesSplit(n_splits=self.params.data_splits)
        train_idx, test_idx = list(ts_cv.split(X, y))[0]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        return ts_cv, X_train, X_test, y_train, y_test
