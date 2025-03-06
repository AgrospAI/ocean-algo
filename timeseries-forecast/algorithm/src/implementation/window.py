from typing import Optional, Sequence
import pandas as pd
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from implementation.estimators import Imputer, Lagger, Log, LogDifference
from logging import getLogger

logger = getLogger(__name__)


def split(df: pd.DataFrame, target_column: str, train_ratio: float) -> tuple:
    n = len(df.index)

    train_df = df.iloc[: int(n * train_ratio)]
    test_df = df.iloc[int(n * train_ratio) :]

    X_train, y_train = (
        train_df.drop(columns=target_column),
        train_df[target_column],
    )
    X_test, y_test = (
        test_df.drop(columns=target_column),
        test_df[target_column],
    )

    return X_train, y_train, X_test, y_test


def generic_pipeline(
    categorical_columns: Sequence[str], target_column: str, lag: int
) -> Pipeline:
    return Pipeline(
        [
            (
                ("imputer", Imputer(categorical_columns=categorical_columns)),
                (
                    "encoder",
                    ColumnTransformer(
                        transformers=[("cat", OneHotEncoder(), categorical_columns)],
                        remainder="passthrough",
                    ),
                ),
                ("scaler", StandardScaler()),
                ("log", Log()),
                ("lag", Lagger(lag=lag)),
                (
                    "log-diff",
                    LogDifference(lag=lag, target_column=f"{target_column}_log"),
                ),
            )
        ]
    )


def evaluate_model(model, X_test, y_test) -> float:
    # Evaluate the model with MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


class WindowGenerator:

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        lag: int,
        preprocessing_pipeline: Optional[Pipeline] = None,
        train_ratio: float = 0.7,
    ):
        self._lag = lag

        self.X_train, self.y_train, self.X_test, self.y_test = split(
            df,
            target_column,
            train_ratio,
        )

        # Fit the preprocessing pipeline
        if preprocessing_pipeline is None:
            categorical_columns = self.X_train.select_dtypes(include="object").columns
            self.preprocessing_pipeline = generic_pipeline(
                categorical_columns=categorical_columns,
                lag=lag,
            )
        else:
            self.preprocessing_pipeline = clone(preprocessing_pipeline)
        self.preprocessing_pipeline.fit(self.X_train)

    def train(self, model) -> tuple[Pipeline, dict[str, float]]:
        X_train = self.preprocessing_pipeline.fit_transform(self.X_train)

        # Train the given model on the training data
        model.fit(X_train, self.y_train)

        # Evaluate the model on the test data
        X_test = self.preprocessing_pipeline.transform(self.X_test)
        mse = evaluate_model(model, X_test, self.y_test)
        logger.info(f"Mean Squared Error: {mse}")

        predicting_pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessing_pipeline),
                ("predictor", model),
            ]
        )

        return predicting_pipeline, {"mse": mse}
