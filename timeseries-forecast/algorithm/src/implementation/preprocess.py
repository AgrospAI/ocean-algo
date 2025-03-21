from typing import List

from implementation.data import ColumnNames
from implementation.estimators import (
    ColumnTransformerWithNames,
    Imputer,
    Periodicity,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_timeseries_pipeline(
    column_names: ColumnNames,
    periodicity: List[str],
    lags: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "periodicity",
                Periodicity(
                    target_column=column_names.target,
                    datetime_column=column_names.datetime,
                    periodicity=periodicity,
                    lags=lags,
                ),
            )
        ]
    )


def get_prepocessing_pipeline(
    column_names: ColumnNames,
) -> Pipeline:
    categorical_columns = column_names.categorical
    categorical_columns.remove(column_names.datetime)

    numeric_columns = column_names.numeric
    numeric_columns.remove(column_names.target)

    return Pipeline(
        [
            (
                "imputer",
                Imputer(
                    datetime_column=column_names.datetime,
                    categorical_columns=categorical_columns,
                    numeric_columns=numeric_columns,
                ),
            ),
            (
                "encoder",
                ColumnTransformerWithNames(
                    transformers=[
                        ("cat", OneHotEncoder(), categorical_columns),
                        ("num", MinMaxScaler((0, 1)), numeric_columns),
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )
