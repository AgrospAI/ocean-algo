from pathlib import Path
from typing import List, Tuple, TypeAlias

import httpx
import pandas as pd  # type: ignore
from ocean_runner import Algorithm, Config
from oceanprotocol_job_details.domain import DID
from returns.io import IOFailure, IOSuccess
from returns.result import Failure, Success

from .aggregate import Aggregate, get_aggregate
from .benchmarking import benchmark
from .data import InputParameters

ResultT: TypeAlias = List[Tuple[DID, Tuple[pd.DataFrame, str]]]
algorithm = Algorithm[InputParameters, ResultT](Config(custom_input=InputParameters))


@algorithm.validate
async def validate(algorithm: Algorithm[InputParameters, ResultT]) -> None:
    assert algorithm.job_details.metadata, "DDOs missing"
    assert algorithm.job_details.files, "Files missing"

    input_parameters = await algorithm.job_details.input_parameters()
    assert input_parameters is not None

    # Check API is alive and accessible
    async with httpx.AsyncClient() as client:
        endpoint = f"{input_parameters.aggregate_api.url}/api/health/"
        response = await client.get(endpoint)
        response.raise_for_status()


async def run_benchmarks(
    algorithm: Algorithm[InputParameters, ResultT],
    aggregate: Aggregate,
) -> ResultT:
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    inputs: List[Tuple[DID, pd.DataFrame]] = []
    for idx, path in algorithm.job_details.inputs():
        name = path.parent.name
        algorithm.logger.info(f"Loading data from {name}")

        data = pd.read_csv(path, sep=parameters.responses_separator)
        inputs.append((name, data))

    return [
        (did, benchmark(aggregate, str(parameters.aggregate_filter), content))
        for did, content in inputs
    ]


@algorithm.run
async def run(algorithm: Algorithm[InputParameters, ResultT]) -> ResultT:
    parameters = await algorithm.job_details.ainput_parameters()
    assert parameters is not None

    match await get_aggregate(parameters):
        case IOSuccess(Success(aggregate)):
            return await run_benchmarks(algorithm, aggregate)
        case IOFailure(Failure(error)):
            raise Algorithm.Error(error)


@algorithm.save_results
async def save(
    algorithm: Algorithm[InputParameters, ResultT],
    result: ResultT,
    base: Path,
) -> None:
    for did, (comparison, template) in result:
        with open(base / f"{did}.html", "w+") as f:
            f.write(template)


if __name__ == "__main__":
    import logging

    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
    algorithm.logger.setLevel("INFO")

    algorithm()
