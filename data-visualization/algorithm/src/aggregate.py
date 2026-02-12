from typing import Any, Dict, TypeAlias

import httpx
import orjson
from jsonschema import Draft202012Validator  # type: ignore
from jsonschema.exceptions import ValidationError  # type: ignore
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success

from src.benchmarking.requests import ObjectType, get_object, make_request
from src.data import InputParameters

Aggregate: TypeAlias = Dict[str, Dict[str, Any]]


class AggregateError(Exception): ...


async def get_aggregate(  # type: ignore[return]
    parameters: InputParameters,
) -> IOResult[Aggregate, AggregateError]:
    match await make_request(
        httpx.Request(
            "GET",
            f"{parameters.aggregate_api.url}/api/schemas/aggregate/",
        )
    ):
        case IOFailure(Failure(error)):
            return IOFailure(AggregateError(error))

        case IOSuccess(Success(response)):
            validator = Draft202012Validator(response.json())

            match await get_object(parameters.aggregate_api.url, ObjectType.AGGREGATE):
                case IOSuccess(Success(aggregate)):
                    try:
                        data = aggregate.json()
                        validator.validate(data)
                        return IOSuccess(data)
                    except ValidationError as e:
                        return IOFailure(AggregateError(e))
                case IOFailure(Failure(error)):
                    print("Error getting aggregate schema")
                    return IOFailure(AggregateError(error))
