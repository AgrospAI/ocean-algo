import asyncio
import datetime
import logging
from os import environ
from pathlib import Path
from typing import Any

import aiofiles
import jinja2
import pandas as pd
from attr import dataclass
from dotenv import get_key
from jsonschema import Draft202012Validator, ValidationError
from ocean_runner import Algorithm, Config, ParametrizedAlgorithm
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Result, Success

from src.benchmarking.preprocessing import (
    calculate_maturity_kpis,
    compare_company_to_aggregate,
    configuration,
    get_overall_kpis,
    process_survey,
    set_configuration,
)
from src.benchmarking.requests import ObjectType, get_object
from src.data import InputParameters


@dataclass(frozen=True)
class NotEnoughDataError:
    message: str
    data_filter: str

    def __str__(self) -> str:
        return f"{self.message}: {self.data_filter}"


type ResultT = Result[
    list[IOResult[tuple[str, str], Algorithm.Error]],
    NotEnoughDataError,
]
type Aggregate = dict[str, dict[str, Any]]


algorithm = Algorithm[InputParameters, ResultT].create(
    Config(custom_input=InputParameters)
)


logging.getLogger("asyncio").setLevel("WARNING")
logging.getLogger("httpx").setLevel("WARNING")
logging.getLogger("httpcore").setLevel("WARNING")
algorithm.logger.setLevel("INFO")


def get_url() -> str:
    if algorithm.job_details.paths.algorithm.exists():
        url = get_key(str(algorithm.job_details.paths.algorithm), "S3_WRAPPER_URL")
    else:
        url = environ.get("S3_WRAPPER_URL")

    assert url is not None, (
        f"URL not populated from environment file {algorithm.job_details.paths.algorithm} nor environment"
    )

    return url


@algorithm.validate
async def validate(_) -> None:
    assert algorithm.job_details.metadata, "DDOs missing"
    assert algorithm.job_details.files, "Files missing"

    url = get_url()

    (
        healthcheck_response,
        config_schema_response,
    ) = await asyncio.gather(
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

    match config_schema_response:
        case IOSuccess(Success(response)):
            set_configuration(response.json())
        case IOFailure(Failure(error)):
            raise Algorithm.Error("Error loading config shcema") from error


async def benchmark(
    did: str,
    aggregate: Aggregate,
    survey: pd.DataFrame,
    overall_kpis: dict,
    url: str,
) -> IOResult[tuple[str, str], Algorithm.Error]:
    algorithm.logger.info("Benchmarking %s", did)

    survey = process_survey(survey)
    survey = calculate_maturity_kpis(survey)
    comparison = compare_company_to_aggregate(survey.iloc[0], aggregate)

    for kpi, _ in comparison["kpis"].items():
        comparison["kpis"][kpi]["aggregate_median"] = overall_kpis[kpi]["median"]

    (translations_response, template_response) = await asyncio.gather(
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
                dimension_labels=configuration["dimension_labels"],
                date=datetime.date.today().strftime("%d %b %Y"),
            ),
        )
    )


async def run_benchmarks(
    algorithm: ParametrizedAlgorithm[InputParameters, ResultT],
    aggregate: Aggregate,
    url: str,
) -> ResultT:
    parameters = algorithm.job_details.input_parameters

    inputs = [
        (path.parent.name, pd.read_csv(path, sep=parameters.responses_separator))
        for _, path in algorithm.job_details.inputs()
    ]

    algorithm.logger.info("Loaded {%s} file(s)", len(inputs))

    filtered_aggregate: dict[str, Any] | None = aggregate.get(
        str(parameters.filter_key), None
    )

    if filtered_aggregate is None:
        algorithm.logger.error(
            "There was an error fetching the aggregate with the given filter, check output."
        )
        return Failure(
            NotEnoughDataError(
                message="Los filtros aplicados no contienen suficientes datos o no están bien formados",
                data_filter=str(parameters.filter_key),
            )
        )

    overall_kpis = get_overall_kpis(aggregate)
    return Success(
        await asyncio.gather(
            *(
                benchmark(
                    did,
                    filtered_aggregate,
                    content,
                    overall_kpis,
                    url,
                )
                for did, content in inputs
            )
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
async def run(_) -> ResultT:  # type: ignore[return]
    url = get_url()

    match await get_aggregate(url):
        case IOSuccess(Success(aggregate)):
            return await run_benchmarks(algorithm, aggregate, url)
        case IOFailure(Failure(error)):
            algorithm.logger.error(error)


@algorithm.save_results
async def save(_, result: ResultT, base: Path) -> None:
    assert result is not None
    assert base is not None

    async def save_render_batch(
        result: IOResult[tuple[str, str], Algorithm.Error],
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

    async def save_error(error: NotEnoughDataError, path: Path) -> None:
        async with aiofiles.open(path / "error.txt", "w+") as f:
            await f.write(str(error))

    def path(did: str) -> Path:
        return base / f"{did}.html"

    match result:
        case Success(batches):
            await asyncio.gather(*(save_render_batch(batch) for batch in batches))
        case Failure(error):
            await save_error(error, base)
