import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, TypeAlias

import aiofiles
import httpx
import jinja2
from ocean_runner import Algorithm, Config
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success

from src.aggregation.benchmark_generation import generate_benchmark_reference
from src.aggregation.preprocessing import (
    calculate_maturity_kpis,
    process_surveys,
)
from src.aggregation.report_rendering import (
    generate_interactive_report,
)
from src.aggregation.requests import ObjectType, get_object, make_request, post_object
from src.data import InputParameters

ResultT: TypeAlias = IOResult[Tuple[str, str], Algorithm.Error]
algorithm = Algorithm[InputParameters, ResultT](Config(custom_input=InputParameters))
Aggregate: TypeAlias = Dict[str, Dict[str, Any]]


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
            algorithm.logger.error(error)
            raise Algorithm.Error("API is not healthy") from error


async def aggregation(
    algorithm: Algorithm[InputParameters, ResultT],
    url: str,
) -> IOResult[Tuple[str, str], Algorithm.Error]:
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    inputs = [(path.parent.name, path) for _, path in algorithm.job_details.inputs()]

    surveys_df = process_surveys(inputs, csv_separator=parameters.csv_separator)
    if surveys_df.empty:
        algorithm.logger.warning("No survey data found")
        return IOFailure(Algorithm.Error("No survey data found"))

    algorithm.logger.info(f"Processed {len(surveys_df)} survey entries")

    surveys_df = calculate_maturity_kpis(surveys_df)
    benchmark_json = generate_benchmark_reference(surveys_df)

    template_response = await asyncio.create_task(
        post_object(url, ObjectType.AGGREGATION_TEMPLATE, benchmark_json)
    )

    match template_response:
        case IOSuccess(Success(response)):
            algorithm.logger.info("Benchmark reference posted successfully")
        case IOFailure(Failure(error)):
            algorithm.logger.error(error)
            return IOFailure(
                Algorithm.Error(f"Failed to post benchmark reference: {error}")
            )
        case _:
            return IOFailure(
                Algorithm.Error(
                    "Unknown error occurred while posting benchmark reference"
                )
            )

    template_response = await asyncio.create_task(
        get_object(url, ObjectType.AGGREGATION_TEMPLATE)
    )

    match template_response:
        case IOSuccess(Success(response)):
            template = jinja2.Template(response.text)
            rendered = template.render(generate_interactive_report(surveys_df))
            return IOSuccess((rendered))
        case IOFailure(Failure(error)):
            return IOFailure(Algorithm.Error(f"Failed to fetch template: {error}"))

        case _:
            return IOFailure(
                Algorithm.Error("Unknown error occurred while fetching template")
            )


@algorithm.run
async def run(algorithm: Algorithm[InputParameters, ResultT]) -> ResultT:
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    return await aggregation(algorithm, parameters.aggregate_api.url)


@algorithm.save_results
async def save(
    _: Algorithm[InputParameters, ResultT],
    result: ResultT,
    base: Path,
) -> None:
    async def save_render_batch(
        result: IOResult[Tuple[str, str], Algorithm.Error],
    ) -> None:
        match result:
            case IOSuccess(Success((did, render))):
                _path = path(did)

                await save_render(render, _path)

            case IOFailure(Failure(error)):
                algorithm.logger.error(error)

    async def save_render(render: str, path: Path) -> None:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(render)

    def path(did: str) -> Path:
        return base / f"{did}.html"

    assert result is not None

    await asyncio.gather(*(save_render_batch(res) for res in result))  # type: ignore


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
    algorithm.logger.setLevel("INFO")

    algorithm()
