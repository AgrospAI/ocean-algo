import logging
import zipfile
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import IO, Iterable, Sequence, Tuple

import pandas as pd
from ocean_runner import Algorithm, Config
from oceanprotocol_job_details.domain import DID
from ydata_profiling import ProfileReport

from .data import InputParameters
from .loader import try_read_df
from .preprocessor import (
    AutoDetectedTimeseriesPreprocessor,
    ManualTimeseriesPreprocessor,
    Preprocessor,
)

type ResultT = Tuple[DID, ProfileReport]
type ResultsT = Sequence[ResultT]

algorithm = Algorithm[InputParameters, ResultsT].create(
    Config(custom_input=InputParameters)
)

getLogger("asyncio").setLevel(logging.INFO)
getLogger("urllib3").setLevel(logging.INFO)
getLogger("matplotlib").setLevel(logging.INFO)
algorithm.logger.setLevel(logging.DEBUG)


def get_df_preprocessor(parameters: InputParameters) -> Preprocessor:
    return (
        AutoDetectedTimeseriesPreprocessor(logger=algorithm.logger)
        if parameters.auto_detect_timeseries_column
        else ManualTimeseriesPreprocessor(
            timeseries_columns=parameters.timeseries_columns_name,
            logger=algorithm.logger,
        )
    )


def generator(
    df: pd.DataFrame,
    parameters: InputParameters,
    timeseries_columns: Sequence[str],
) -> ProfileReport:
    report_factory = partial(
        ProfileReport,
        df=df,
        title=parameters.title,
        sensitive=parameters.sensitive,
        progress_bar=False,
        samples=None,
    )

    if timeseries_columns:
        valid_ts = [c for c in timeseries_columns if c in df.columns]

        if valid_ts:
            ts_col = valid_ts[0]

            df = df.sort_values(by=ts_col)

        type_schema = {col: "datetime" for col in valid_ts}

        report_factory: partial[ProfileReport] = partial(
            report_factory,
            df=df,
            title=f"{parameters.title} - Timeseries",
            type_schema=type_schema,
        )

    return report_factory()


def get_inputs() -> Iterable[tuple[str, IO[bytes]]]:

    def open_stream(did: str, path: Path) -> Iterable[tuple[str, IO[bytes]]]:
        if not zipfile.is_zipfile(path):
            with path.open("rb") as f:
                yield did, f
        else:
            try:
                with zipfile.ZipFile(path, "r") as zip_ref:
                    for idx, member in enumerate(
                        m for m in zip_ref.namelist() if not m.endswith("/")
                    ):
                        with zip_ref.open(member) as f:
                            yield f"{did}_{idx}", f
            except Exception as e:
                algorithm.logger.error("Failure reading input path %s: %s", path, e)

    for idx, (did, path) in enumerate(algorithm.job_details.inputs()):
        for id, io_stream in open_stream(f"{did}_{idx}", path):
            yield (id, io_stream)


@algorithm.run
def run(_) -> ResultsT:
    parameters = algorithm.job_details.input_parameters

    def process_input(did: str, content: IO[bytes]) -> ResultT | None:
        df = try_read_df(algorithm.logger, content)

        if df is None:
            return

        preprocessor = get_df_preprocessor(parameters)
        preprocessed_df, timeseries_columns = preprocessor.preprocess(df)
        algorithm.logger.debug("Completed preprocess using %s", preprocessor.__class__)

        for col in preprocessed_df.columns:
            types = preprocessed_df[col].map(type).value_counts()

            if len(types) > 1:
                algorithm.logger.warning("Column '%s' has mixed types: %s", col, types)

            if col not in timeseries_columns:
                if preprocessed_df[col].map(type).nunique() > 1:
                    preprocessed_df[col] = preprocessed_df[col].astype(str)

        return (did, generator(preprocessed_df, parameters, timeseries_columns))

    results: list[ResultT] = []
    for did, fp in get_inputs():
        result = process_input(did, fp)

        if result is not None:
            results.append(result)

    return results


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    for did, report in result:
        report.to_file(base / f"{did}.html")
