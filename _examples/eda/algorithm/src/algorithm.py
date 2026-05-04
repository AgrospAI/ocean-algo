from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
from ocean_runner import Algorithm, Config
from ydata_profiling import ProfileReport

from .data import InputParameters

type ResultT = Tuple[str, ProfileReport]
type ResultsT = Sequence[ResultT]
algorithm = Algorithm[InputParameters, ResultT].create(
    Config(custom_input=InputParameters)
)


@algorithm.run
def run(_) -> ResultsT:

    def generate_profile_report(df: pd.DataFrame) -> ProfileReport:
        parameters = algorithm.job_details.input_parameters
        return ProfileReport(df, title=parameters.title, sensitive=parameters.sensitive)

    def process_input(
        did: str,
        file_path: Path,
    ) -> ResultT:
        df = pd.read_csv(file_path)
        return (did, generate_profile_report(df))

    return [
        process_input(did, file_path)
        for did, file_path in algorithm.job_details.inputs()
    ]


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    for did, report in result:
        report.to_file(base / f"{did}.html")
