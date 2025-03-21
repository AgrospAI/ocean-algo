from dataclasses import dataclass
from logging import getLogger
from typing import List, Sequence

from implementation.data import ColumnNames
from implementation.preprocess import (
    get_prepocessing_pipeline,
    get_timeseries_pipeline,
)
from pandas import DataFrame, Series
from sklearn.base import TransformerMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

logger = getLogger(__name__)


@dataclass
class WindowGenerator:
    df: DataFrame
    target_column: str
    datetime_column: str
    lags: int = 3
    train_ratio: float = 0.7

    def __post_init__(
        self,
    ):
        self.column_names = ColumnNames(
            datetime=self.datetime_column,
            target=self.target_column,
            categorical=list(self.df.select_dtypes(include="object").columns),
            numeric=list(self.df.select_dtypes(include="number").columns),
        )

        # Timeseries features pipeline, to apply to the whole data
        self.timeseries_pipeline = get_timeseries_pipeline(
            self.column_names,
            self.lags,
        )

        # Preprocessing pipeline, to apply to the training features
        self.preprocessing_pipeline = get_prepocessing_pipeline(
            self.column_names,
        )

    def preprocess(
        self,
    ) -> List:
        """Preprocess the pipeline on the training features.

        1. Add time periodicity features to the data.
        1. Split the training and testing data.
        1. Train the preprocessing pipeline on the training data.
        """

        # Add time periodicity features to the training data
        self.df = self.timeseries_pipeline.fit_transform(self.df)
        logger.info(
            f"After timeseries feature adding data shape: {self.df.shape}, head: \n{self.df.head()}"
        )

        # Plot periodicity of the data
        self.inspect_timedata(self.df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.drop(columns=[self.target_column]),
            self.df[self.target_column],
            train_size=self.train_ratio,
        )
        logger.info(f"Train shape: {X_train.shape} - Test shape: {X_test.shape}")

        X_train = self.preprocessing_pipeline.fit_transform(X_train)
        X_test = self.preprocessing_pipeline.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: DataFrame,
        y_train: Series,
        model: TransformerMixin,
    ) -> Pipeline:
        logger.info("=== After preprocessing")
        logger.info(f"Shapes: X {X_train.shape} y {y_train.shape}")

        # Train the given model on the training data
        model.fit(X_train, y_train)

        return make_pipeline(self.preprocessing_pipeline, model)

    def evaluate(
        self,
        trained_model,
        X_test: DataFrame,
        y_true: Series,
        metrics: Sequence[str],
    ) -> float:
        y_pred = trained_model.predict(X_test.to_numpy())
        results = {}

        for metric in metrics:
            try:
                scorer = get_scorer(metric)
            except ValueError as e:
                logger.error(f"Error getting scorer: {e}")
                continue

            try:
                results[metric] = scorer._score_func(y_true, y_pred)
            except Exception as e:
                logger.error(f"Error calculating metric {metric}: {e}")
                continue

        return results

    def inspect_timedata(
        self,
        df: DataFrame,
        n_samples: int = 50,
    ) -> None:
        import matplotlib.pyplot as plt
        from seaborn import color_palette, lineplot

        col_template = "sample_{period}_{operation}"

        palette = color_palette("husl", 8)
        periods = [
            # "day",
            # "week",
            "month",
            "year",
        ]

        for i, period in enumerate(periods):
            cos = col_template.format(period=period, operation="cos")
            sin = col_template.format(period=period, operation="sin")
            f = lineplot(
                data=df[[cos, sin]][:n_samples],
                palette=palette[i * 2 : i * 2 + 2],
            ).get_figure()
            plt.xticks(rotation=90)
            plt.tight_layout()

        f.savefig(f"/data/outputs/periodicity_{n_samples}.png")
        logger.info("Periodicity plots saved")
