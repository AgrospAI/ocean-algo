from typing import Any, Dict, TypeAlias

import httpx
import orjson
from jsonschema import Draft202012Validator  # type: ignore
from jsonschema.exceptions import ValidationError  # type: ignore
from returns.io import IO, IOFailure, IOResult, IOSuccess

from .data import InputParameters

Aggregate: TypeAlias = Dict[str, Dict[str, Any]]


class AggregateError(Exception): ...


async def get_aggregate_schema(
    url: str,
) -> IOResult[Draft202012Validator, AggregateError]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/api/schemas/aggregate/")
            response.raise_for_status()

            schema = response.json()
            validator = Draft202012Validator(schema)

            return IOSuccess(validator)
    except Exception as e:
        return IOFailure(AggregateError(e))


async def get_aggregate(
    parameters: InputParameters,
) -> IOResult[Aggregate, AggregateError]:  # type: ignore[return]
    aggregate_schema_result = await get_aggregate_schema(parameters.aggregate_api.url)

    if isinstance(aggregate_schema_result, IOFailure):
        return aggregate_schema_result

    validator: IO[Draft202012Validator] = aggregate_schema_result.unwrap()

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            endpoint = f"{parameters.aggregate_api.url}/api/aggregate/"

            response = await client.get(endpoint)
            response.raise_for_status()

            payload = orjson.loads(response.content)
            try:
                validator.bind(lambda validator: validator.validate(payload))
            except ValidationError as e:
                return IOFailure(AggregateError(e))

            return IOSuccess(payload)
    except httpx.HTTPStatusError as e:
        return IOFailure(AggregateError(e))
