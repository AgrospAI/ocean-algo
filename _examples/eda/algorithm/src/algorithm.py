from pathlib import Path
from typing import Callable, Sequence, Tuple

import pandas as pd
from ocean_runner import Algorithm, EmptyAlgorithm
from oceanprotocol_job_details.domain import DID
from ydata_profiling import ProfileReport  # type: ignore

type ResultT = Tuple[DID, ProfileReport]
type ResultsT = Sequence[ResultT]
algorithm: EmptyAlgorithm[ResultsT] = Algorithm[None, ResultsT].create(None)

TITLE = "Profiling Report"
SENSITIVE = False


@algorithm.run
def run(_) -> ResultsT:
    def generate_profile_report(df: pd.DataFrame) -> ProfileReport:
        return ProfileReport(df, title=TITLE, sensitive=SENSITIVE)

    def process_input(
        did: str,
        file_path: Path,
        generate_report: Callable[[pd.DataFrame], ProfileReport],
    ) -> ResultT:
        df = pd.read_csv(file_path)
        return (did, generate_report(df))

    return [
        process_input(did, file_path, generate_profile_report)
        for did, file_path in algorithm.job_details.inputs()
    ]


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    for did, report in result:
        report.to_file(base / f"{did}.html")

