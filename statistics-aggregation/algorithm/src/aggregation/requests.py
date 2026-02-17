from enum import StrEnum, auto

import httpx
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success


class ObjectType(StrEnum):
    AGGREGATE = auto()
    AGGREGATION_TEMPLATE = auto()
    # BENCHMARKING_TRANSLATIONS = auto()


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


async def get_object(
    endpoint: str,
    object_type: ObjectType,
) -> IOResult[httpx.Response, EXCEPTION]:
    match object_type:
        case ObjectType.AGGREGATION_TEMPLATE:
            endpoint = f"{endpoint}/api/aggregate-template/"

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


async def post_object(
    endpoint: str,
    object_type: ObjectType,
    data: dict,
) -> IOResult[httpx.Response, EXCEPTION]:
    request_kwargs = {
        "method": "POST",
        "params": {"object_type": str(object_type)},
    }

    match object_type:
        case ObjectType.AGGREGATION_TEMPLATE:
            request_kwargs["url"] = f"{endpoint}/api/aggregate/"
            request_kwargs["json"] = data

        case ObjectType.AGGREGATE:
            request_kwargs["url"] = f"{endpoint}/api/aggregate-template/"
            request_kwargs["headers"] = {"Content-Type": "text/html"}

    match await make_request(httpx.Request(**request_kwargs)):
        case IOSuccess(Success(response)):
            return IOSuccess(response)
        case IOFailure(Failure(error)):
            return IOFailure(error)
