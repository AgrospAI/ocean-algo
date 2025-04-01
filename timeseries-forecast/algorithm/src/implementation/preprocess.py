from typing import List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from implementation.data import ColumnNames
from implementation.estimators import (
    ColumnTransformerWithNames,
    Imputer,
    Periodicity,
)


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


def get_preprocessing_pipeline(
    column_names: ColumnNames,
) -> Pipeline:
    categorical_columns = column_names.categorical

    if column_names.datetime in categorical_columns:
        # Remove datetime column from categorical columns
        categorical_columns.remove(column_names.datetime)

    numeric_columns = column_names.numeric

    if column_names.target in numeric_columns:
        # Remove target column from numeric columns
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
