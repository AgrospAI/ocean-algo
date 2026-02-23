import logging
from pathlib import Path
from typing import Any, Dict, TypeAlias

import aiofiles
import httpx
import jinja2
import orjson
from ocean_runner import Algorithm, Config
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success

from src.aggregation.benchmark_generation import BenchmarkReference
from src.aggregation.preprocessing import Preprocessing
from src.aggregation.report_rendering import (
    generate_interactive_report,
)
from src.aggregation.requests import ObjectType, get_object, make_request, post_object
from src.data import InputParameters

ResultT: TypeAlias = IOResult[str, Algorithm.Error]
algorithm = Algorithm[InputParameters, ResultT](Config(custom_input=InputParameters))
Aggregate: TypeAlias = Dict[str, Dict[str, Any]]


@algorithm.validate
async def validate(algorithm: Algorithm[InputParameters, ResultT]) -> None:
    assert algorithm.job_details.metadata, "DDOs missing"
    assert algorithm.job_details.files, "Files missing"

    input_parameters = await algorithm.job_details.input_parameters()
    assert input_parameters is not None

    request = httpx.Request("GET", f"{input_parameters.aggregate_api}/api/health/")
    match await make_request(request):
        case IOSuccess(Success(_)):
            algorithm.logger.info("API Healthcheck done")
        case IOFailure(Failure(error)):
            algorithm.logger.error(error)
            raise Algorithm.Error("API is not healthy") from error


async def aggregation(
    algorithm: Algorithm[InputParameters, ResultT],
    url: str,
) -> IOResult[str, Algorithm.Error]:
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    inputs = [(path.parent.name, path) for _, path in algorithm.job_details.inputs()]

    with open("/algorithm/src/config_schema.json", "r") as file:
        config_file = orjson.loads(file.read())

    preprocessor = Preprocessing(config_file)
    surveys_df = preprocessor.process_surveys(
        inputs, csv_separator=parameters.csv_separator
    )
    if surveys_df.empty:
        algorithm.logger.warning("No survey data found")
        return IOFailure(Algorithm.Error("No survey data found"))

    algorithm.logger.info(f"Processed {len(surveys_df)} survey entries")

    surveys_df = preprocessor.calculate_maturity_kpis(surveys_df)

    benchmark_ref = BenchmarkReference(config_file)
    benchmark_json = benchmark_ref.generate_benchmark_reference(surveys_df)

    template_response = await post_object(url, ObjectType.AGGREGATE, benchmark_json)

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

    template_response = await get_object(url, ObjectType.AGGREGATE_TEMPLATE)

    match template_response:
        case IOSuccess(Success(response)):
            template = jinja2.Template(response.text)
            rendered = template.render(
                generate_interactive_report(surveys_df, config_file)
            )
            return IOSuccess(rendered)
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

    return await aggregation(algorithm, parameters.aggregate_api)


@algorithm.save_results
async def save(
    _: Algorithm[InputParameters, ResultT],
    result: ResultT,
    base: Path,
) -> None:
    async def save_render_batch(
        result: IOResult[str, Algorithm.Error],
    ) -> None:
        match result:
            case IOSuccess(Success(render)):
                await save_render(render, base / "report.html")

            case IOFailure(Failure(error)):
                algorithm.logger.error(error)

    async def save_render(render: str, path: Path) -> None:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(render)

    assert result is not None

    await save_render_batch(result)


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
    logging.getLogger("pandas").setLevel("ERROR")
    logging.getLogger("numpy").setLevel("ERROR")
    logging.getLogger("plotly").setLevel("ERROR")
    algorithm.logger.setLevel("INFO")

    algorithm()
