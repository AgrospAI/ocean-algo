import asyncio
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeAlias

import aiofiles
import httpx
import jinja2
import pandas as pd  # type: ignore
from jsonschema import Draft202012Validator, ValidationError  # type: ignore
from ocean_runner import Algorithm, Config
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success

from src.benchmarking.config_schema import DIMENSION_LABELS
from src.benchmarking.preprocessing import (
    calculate_maturity_kpis,
    compare_company_to_aggregate,
    get_overall_kpis,
    process_survey,
)
from src.benchmarking.requests import ObjectType, get_object, make_request
from src.data import InputParameters

ResultT: TypeAlias = List[IOResult[Tuple[str, str], Algorithm.Error]]
algorithm = Algorithm[InputParameters, ResultT](Config(custom_input=InputParameters))
Aggregate: TypeAlias = Dict[str, Dict[str, Any]]


@algorithm.validate
async def validate(algorithm: Algorithm[InputParameters, ResultT]) -> None:
    assert algorithm.job_details.metadata, "DDOs missing"
    assert algorithm.job_details.files, "Files missing"

    input_parameters = await algorithm.job_details.input_parameters()
    assert input_parameters is not None

    request = httpx.Request("GET", f"{input_parameters.url}/api/health/")
    match await make_request(request):
        case IOSuccess(Success(_)):
            algorithm.logger.info("API Healthcheck done")
        case IOFailure(Failure(error)):
            algorithm.logger.error(error)
            raise Algorithm.Error("API is not healthy") from error


async def benchmark(
    did: str,
    aggregate: Aggregate,
    survey: pd.DataFrame,
    overall_kpis: dict,
    url: str,
) -> IOResult[Tuple[str, str], Algorithm.Error]:
    algorithm.logger.info(f"Benchmarking {did}")

    survey = process_survey(survey)
    survey = calculate_maturity_kpis(survey)
    comparison = compare_company_to_aggregate(survey.iloc[0], aggregate)

    for kpi, values in comparison["kpis"].items():
        comparison["kpis"][kpi]["aggregate_median"] = overall_kpis[kpi]["median"]

    translations_response, template_response = await asyncio.gather(
        get_object(url, ObjectType.BENCHMARKING_TRANSLATIONS),
        get_object(url, ObjectType.BENCHMARKING_TEMPLATE),
    )

    match translations_response:
        case IOSuccess(Success(response)):
            translations = response.json()
        case IOFailure(Failure(error)):
            return IOFailure(Algorithm.Error(error))

    match template_response:
        case IOSuccess(Success(response)):
            template = jinja2.Template(response.text)
        case IOFailure(Failure(error)):
            return IOFailure(Algorithm.Error(error))

    return IOSuccess(
        (
            did,
            template.render(
                **comparison,
                translations=translations,
                dimension_labels=DIMENSION_LABELS,
                date=datetime.date.today().strftime("%d %b %Y"),
            ),
        )
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

    overall_kpis = get_overall_kpis(aggregate)
    return await asyncio.gather(
        *(
            benchmark(
                did,
                aggregate[str(parameters.filter_key)],
                content,
                overall_kpis,
                parameters.url,
            )
            for did, content in inputs
        )
    )


class AggregateError(Exception): ...


async def get_aggregate(url: str) -> IOResult[Aggregate, AggregateError]:  # type: ignore[return]
    match await get_object(url, ObjectType.AGGREGATE_SCHEMA):
        case IOFailure(Failure(error)):
            return IOFailure(AggregateError(error))

        case IOSuccess(Success(response)):
            validator = Draft202012Validator(response.json())

            match await get_object(url, ObjectType.AGGREGATE):
                case IOSuccess(Success(aggregate)):
                    try:
                        data = aggregate.json()
                        validator.validate(data)
                        return IOSuccess(data)
                    except ValidationError as e:
                        return IOFailure(AggregateError(e))
                case IOFailure(Failure(error)):
                    return IOFailure(AggregateError(error))


@algorithm.run
async def run(algorithm: Algorithm[InputParameters, ResultT]) -> ResultT:  # type: ignore[return]
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    match await get_aggregate(parameters.url):
        case IOSuccess(Success(aggregate)):
            algorithm.logger.info("Running benchmarks")
            return await run_benchmarks(algorithm, aggregate)
        case IOFailure(Failure(error)):
            algorithm.logger.error(error)


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
        async with aiofiles.open(path, "w+") as f:
            await f.write(render)

    def path(did: str) -> Path:
        return base / f"{did}.html"

    assert result is not None

    await asyncio.gather(*(save_render_batch(res) for res in result))


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
    algorithm.logger.setLevel("INFO")

    algorithm()
