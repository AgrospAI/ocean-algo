from enum import StrEnum, auto

import httpx
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success


class ObjectType(StrEnum):
    AGGREGATE = auto()
    AGGREGATE_SCHEMA = auto()
    BENCHMARKING_TEMPLATE = auto()
    BENCHMARKING_TRANSLATIONS = auto()


type EXCEPTION = httpx.TransportError | httpx.HTTPStatusError | httpx.InvalidURL
EXCEPTIONS = (httpx.TransportError, httpx.HTTPStatusError, httpx.InvalidURL)


async def make_request(request: httpx.Request) -> IOResult[httpx.Response, EXCEPTION]:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.send(request)
            response.raise_for_status()
            return IOSuccess(response)
    except EXCEPTIONS as e:
        return IOFailure(e)


async def get_object(  # type: ignore[return]
    endpoint: str,
    object_type: ObjectType,
) -> IOResult[httpx.Response, EXCEPTION]:
    match object_type:
        case (
            ObjectType.AGGREGATE
            | ObjectType.BENCHMARKING_TEMPLATE
            | ObjectType.BENCHMARKING_TRANSLATIONS
        ):
            endpoint = f"{endpoint}/api/aggregate/"
        case ObjectType.AGGREGATE_SCHEMA:
            endpoint = f"{endpoint}/api/schemas/aggregate/"

    request = httpx.Request(
        method="GET",
        url=endpoint,
        params={"object_type": str(object_type)},
    )

    match await make_request(request):
        case IOSuccess(Success(response)):
            return IOSuccess(response)
        case IOFailure(Failure(error)):
            return IOFailure(error)
