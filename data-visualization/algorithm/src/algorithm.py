import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, TypeAlias

import aiofiles
import httpx
import jinja2
import pandas as pd  # type: ignore  # type: ignore
from ocean_runner import Algorithm, Config
from returns.io import IOFailure, IOSuccess
from returns.result import Failure, Success

from src.aggregate import Aggregate, get_aggregate
from src.benchmarking.config_schema import DIMENSION_LABELS
from src.benchmarking.preprocessing import (
    calculate_maturity_kpis,
    compare_company_to_aggregate,
    process_survey,
)
from src.benchmarking.requests import ObjectType, get_object, make_request
from src.data import InputParameters

ResultT: TypeAlias = List[Tuple[str, str]]
algorithm = Algorithm[InputParameters, ResultT](Config(custom_input=InputParameters))


@algorithm.validate
async def validate(algorithm: Algorithm[InputParameters, ResultT]) -> None:
    assert algorithm.job_details.metadata, "DDOs missing"
    assert algorithm.job_details.files, "Files missing"

    input_parameters = await algorithm.job_details.input_parameters()
    assert input_parameters is not None

    request = httpx.Request("GET", f"{input_parameters.aggregate_api.url}/api/health/")
    match await make_request(request):
        case IOSuccess(Success(_)):
            algorithm.logger.info("API Healthcheck done")
        case IOFailure(Failure(error)):
            algorithm.logger.error("Checking API health")
            raise Algorithm.Error from error


async def benchmark(
    did: str,
    aggregate: Aggregate,
    survey: pd.DataFrame,
    url: str,
) -> Tuple[str, str]:
    algorithm.logger.info(f"Benchmarking {did}")

    survey = process_survey(survey)
    survey = calculate_maturity_kpis(survey)
    comparison = compare_company_to_aggregate(survey.iloc[0], aggregate)

    translations_response, template_response = await asyncio.gather(
        get_object(url, ObjectType.BENCHMARKING_TRANSLATIONS),
        get_object(url, ObjectType.BENCHMARKING_TEMPLATE),
    )

    match translations_response:
        case IOFailure(Failure(error)):
            raise Algorithm.Error from error
        case IOSuccess(Success(response)):
            translations = response.json()

    match template_response:
        case IOFailure(Failure(error)):
            raise Algorithm.Error from error
        case IOSuccess(Success(response)):
            template = jinja2.Template(response.text)

    return (
        did,
        template.render(
            **comparison,
            translations=translations,
            dimension_labels=DIMENSION_LABELS,
        ),
    )


async def run_benchmarks(
    algorithm: Algorithm[InputParameters, ResultT],
    aggregate: Aggregate,
) -> ResultT:
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    inputs = [
        (path.parent.name, pd.read_csv(path, sep=parameters.responses_separator))
        for _, path in algorithm.job_details.inputs()
    ]

    algorithm.logger.info(f"Loaded {len(inputs)} file(s)")

    return await asyncio.gather(
        *(
            benchmark(
                did,
                aggregate[str(parameters.aggregate_filter)],
                content,
                parameters.aggregate_api.url,
            )
            for did, content in inputs
        )
    )


@algorithm.run
async def run(algorithm: Algorithm[InputParameters, ResultT]) -> ResultT:  # type: ignore[return]
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    match await get_aggregate(parameters):
        case IOSuccess(Success(aggregate)):
            algorithm.logger.info("Running benchmarks")
            return await run_benchmarks(algorithm, aggregate)
        case IOFailure(Failure(error)):
            algorithm.logger.error("ERROR getting aggregate")
            raise Algorithm.Error from error
        case _:
            algorithm.logger.error("??")


@algorithm.save_results
async def save(
    _: Algorithm[InputParameters, ResultT],
    result: ResultT,
    base: Path,
) -> None:
    async def save_render(render: str, path: Path) -> None:
        async with aiofiles.open(path, "w+") as f:
            await f.write(render)

    def path(did: str) -> Path:
        return base / f"{did}.html"

    assert result is not None
    await asyncio.gather(*(save_render(render, path(did)) for did, render in result))


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
    algorithm.logger.setLevel("INFO")

    algorithm()
