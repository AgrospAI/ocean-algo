import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, TypeAlias

import aiofiles
import jinja2
from dotenv import get_key
from ocean_runner import Algorithm, Config, ParametrizedAlgorithm
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success

from src.aggregation.benchmark_generation import BenchmarkReference
from src.aggregation.preprocessing import Preprocessing
from src.aggregation.report_rendering import (
    generate_interactive_report,
)
from src.aggregation.requests import ObjectType, get_object, post_object
from src.data import InputParameters

ResultT: TypeAlias = IOResult[str, Algorithm.Error]
algorithm = ParametrizedAlgorithm[InputParameters, ResultT].create(
    Config(custom_input=InputParameters)
)
Aggregate: TypeAlias = Dict[str, Dict[str, Any]]

logging.getLogger("asyncio").setLevel("WARNING")
logging.getLogger("httpx").setLevel("WARNING")
logging.getLogger("httpcore").setLevel("WARNING")
logging.getLogger("jinja2").setLevel("WARNING")
logging.getLogger("pandas").setLevel("WARNING")
algorithm.logger.setLevel("INFO")


async def validate(_) -> None:
    assert algorithm.job_details.metadata, "DDOs missing"
    assert algorithm.job_details.files, "Files missing"

    url = algorithm.job_details.input_parameters.url
    (healthcheck_response) = await asyncio.gather(
        *(
            get_object(url, ObjectType.HEALTHCHECK),
            get_object(url, ObjectType.CONFIG_SCHEMA),
        )
    )

    match healthcheck_response:
        case IOSuccess(Success(_)):
            algorithm.logger.info("API Healthcheck: OK")
        case IOFailure(Failure(error)):
            algorithm.logger.error(error)
            raise Algorithm.Error("API is not healthy") from error


async def aggregation(
    algorithm: ParametrizedAlgorithm[InputParameters, ResultT],
    url: str | None,
) -> IOResult[str, Algorithm.Error]:
    inputs = [(path.parent.name, path) for _, path in algorithm.job_details.inputs()]

    config_schema_response = await get_object(url, ObjectType.CONFIG_SCHEMA)

    match config_schema_response:
        case IOSuccess(Success(response)):
            config_file = response.json()
        case IOFailure(Failure(error)):
            return IOFailure(Algorithm.Error(f"Failed to fetch config schema: {error}"))
        case _:
            return IOFailure(
                Algorithm.Error("Unknown error occurred while fetching template")
            )

    preprocessor = Preprocessing(config_file)
    surveys_df = preprocessor.process_surveys(inputs)

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
async def run(algorithm: ParametrizedAlgorithm[InputParameters, ResultT]) -> ResultT:
    return await aggregation(algorithm, get_url())


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


def get_url() -> str | None:
    return get_key("/data/transformations/algorithm", "S3_WRAPPER_URL")
