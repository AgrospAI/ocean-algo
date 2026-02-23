from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
from ocean_runner import Algorithm, EmptyAlgorithm
from oceanprotocol_job_details.domain import DID

type ResultT = Tuple[DID, pd.DataFrame]
type ResultsT = Sequence[ResultT]

algorithm: EmptyAlgorithm[ResultsT] = Algorithm[None, ResultsT].create(None)


@algorithm.run
def run(_) -> ResultsT:
    return [
        (did, pd.read_csv(file_path).describe(include="all"))
        for did, file_path in algorithm.job_details.inputs()
    ]


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    for did, analysis in result:
        algorithm.logger.info(f"Descriptive statistics {did}: {analysis}")
        analysis.to_csv(base / did)


