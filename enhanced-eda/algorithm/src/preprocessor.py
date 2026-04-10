from dataclasses import dataclass
from logging import Logger
from typing import ClassVar, Protocol, Sequence

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

    def datetime_likelihood(self, sample: pd.Series) -> float:
        s = sample.dropna().astype(str)

        if len(s) == 0:
            return 0.0

        score = 0

        # contains separators typical of dates
        score += s.str.contains(r"[-/:T ]").mean()
        # contains digits
        score += s.str.contains(r"\d").mean()
        # contains year-like patterns
        score += s.str.contains(r"\b(?:19|20)\d{2}\b").mean()
        # penalize alphanumeric IDs
        score -= s.str.contains(r"[A-Za-z]{2,}\d+").mean()

        return score / 3

    def is_probably_id(self, series: pd.Series) -> bool:
        s = series.dropna().astype(str)

        if len(s) == 0:
            return False

        return (
            s.str.match(r"^[A-Z0-9_-]+$").mean() > 0.9
            and s.str.contains(r"[A-Za-z]").mean() > 0.5
        )

    def try_parse(self, series: pd.Series) -> tuple[pd.Series | None, float]:
        strategies = ["%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%H:%M:%S"]

        best: pd.Series | None = None
        best_score: float = 0.0

        for fmt in strategies:
            try:
                parsed = pd.to_datetime(series, format=fmt, errors="coerce")
                score = parsed.notna().mean()

                if score > best_score:
                    best = parsed
                    best_score = score

            except Exception:
                continue

        if best_score < 0.85:
            import warnings

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Could not infer format")
                    parsed = pd.to_datetime(series, utc=True, errors="coerce")

                score = parsed.notna().mean()

                if score > best_score:
                    best = parsed
                    best_score = score

            except Exception:
                pass

        return best, best_score

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Sequence[str]]:
        self.logger.debug("Auto-detecting datetime columns...")

        df_copy = df.copy()
        columns = df_copy.select_dtypes(include=["object", "string"]).columns
        ts_columns: list[str] = []

        for col in columns:
            series = df_copy[col]

            # Skip obvious IDs
            if self.is_probably_id(series):
                self.logger.debug("Skipping '%s' (likely ID)", col)
                continue

            sample = series.dropna().astype(str).head(20)

            likelihood = self.datetime_likelihood(sample)

            self.logger.debug("Column '%s' datetime likelihood: %.2f", col, likelihood)

            if likelihood < 0.5:
                self.logger.debug("Skipping '%s' (low likelihood)", col)
                continue

            parsed, success_ratio = self.try_parse(series)

            if parsed is None:
                self.logger.debug("Skipping '%s' (no parse succeeded)", col)
                continue

            self.logger.debug(
                "Column '%s' parse success: %.2f%%",
                col,
                success_ratio * 100,
            )

            # Strong acceptance criteria
            if success_ratio > 0.85 and parsed.nunique() > 1:
                df_copy[col] = parsed
                ts_columns.append(col)

                self.logger.debug("Converted column '%s' to datetime", col)
            else:
                self.logger.debug("Skipping '%s' (low success or low variance)", col)

        self.logger.info("Auto-detected timeseries columns: %s", ts_columns)

        # Drop rows with all timestamps missing
        if ts_columns:
            before = len(df_copy)
            df_copy = df_copy.dropna(subset=ts_columns, how="all")
            after = len(df_copy)

            if before != after:
                self.logger.info("Dropped %d rows with null timestamps", before - after)

            for col in ts_columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")

        return df_copy, ts_columns


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
