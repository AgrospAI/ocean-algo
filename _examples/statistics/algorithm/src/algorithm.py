from pathlib import Path

import pandas as pd
from ocean_runner import Algorithm, EmptyInputParameters

type ResultT = tuple[str, pd.DataFrame]
type ResultsT = list[ResultT]

algorithm = Algorithm[EmptyInputParameters, ResultsT].create(None)


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
