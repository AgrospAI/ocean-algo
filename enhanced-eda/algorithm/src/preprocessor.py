from dataclasses import dataclass
from logging import Logger
from typing import Protocol, Sequence

import pandas as pd


class Preprocessor(Protocol):
    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Sequence[str]]:
        """Preprocess the given data without changing the original

        Args:
            df (pd.DataFrame): data

        Returns:
            tuple[pd.DataFrame, Sequence[str]]: preprocessed data and the list of timeseries columns
        """


@dataclass(frozen=True)
class AutoDetectedTimeseriesPreprocessor(Preprocessor):
    logger: Logger

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Sequence[str]]:
        self.logger.debug("Auto-detecting columns...")

        df_copy = df.copy()
        columns = df_copy.select_dtypes(include=["object", "string"]).columns
        ts_columns: list[str] = []

        for col in columns:
            converted = pd.to_datetime(df_copy[col], dayfirst=True, errors="coerce")

            if converted.notna().sum() > (len(df_copy) * 0.5):
                df_copy[col] = converted
                self.logger.debug("Converted column '%s' to datetime", col)
                ts_columns.append(col)
            else:
                self.logger.debug("Skipping column '%s'; not clearly a date", col)

            success_ratio = converted.notna().mean()

            self.logger.debug(
                "Column '%s' datetime parse success: %.2f%%",
                col,
                success_ratio * 100,
            )

        if ts_columns:
            before_count = len(df_copy)
            # subset=ts_columns ensures we only drop if the DATE is missing
            df_copy = df_copy.dropna(subset=ts_columns, how="all")
            after_count = len(df_copy)

            if before_count != after_count:
                self.logger.info(
                    "Dropped %d rows with null timestamps", before_count - after_count
                )

        return (df_copy, ts_columns)


@dataclass(frozen=True)
class ManualTimeseriesPreprocessor(Preprocessor):
    """Preprocess the given timeseries for the given columns"""

    logger: Logger
    timeseries_columns: Sequence[str]

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Sequence[str]]:
        df_copy = df.copy()

        actual_ts_cols: list[str] = []

        for col in self.timeseries_columns:
            if col not in df_copy.columns:
                self.logger.warning(
                    "Skipping given column '%s' since it is not present in the data",
                    col,
                )
                continue

            converted = pd.to_datetime(df_copy[col], dayfirst=True, errors="coerce")
            self.logger.debug(
                "Parsing '%s' as day-first datetime, example: %s",
                col,
                converted.iloc[0],
            )

            if converted.notna().sum() == 0:
                self.logger.error("Column '%s' could not be parsed as datetime", col)
                continue

            df_copy[col] = converted
            actual_ts_cols.append(col)

        if actual_ts_cols:
            before_count = len(df_copy)
            # We use how="any" here because if a user MANUALLY specified
            # these columns, they likely expect data to exist in all of them.
            df_copy = df_copy.dropna(subset=actual_ts_cols, how="any")

            self.logger.info(
                "Dropped %d rows with NaT in manual columns",
                before_count - len(df_copy),
            )

        return (df_copy, actual_ts_cols)
