from enum import StrEnum, auto

import httpx
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success


class ObjectType(StrEnum):
    AGGREGATE = auto()
    AGGREGATE_TEMPLATE = auto()
    CONFIG_SCHEMA = auto()
    HEALTHCHECK = auto()


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
    """Get one of the defined objects from the S3 wrapper.

    Args:
        object_type (ObjectType): Type of object to get

    Returns:
        IOResult[httpx.Response, EXCEPTION]: Response with the object or one of the defined exceptions.
    """

    match object_type:
        case ObjectType.HEALTHCHECK:
            endpoint = f"{endpoint}/api/health/"
        case _:
            endpoint = f"{endpoint}/api/aggregate/"

    match await make_request(
        httpx.Request(
            method="GET",
            url=endpoint,
            params={"object_type": object_type.value},
        )
    ):
        case IOSuccess(Success(response)):
            return IOSuccess(response)
        case IOFailure(Failure(error)):
            return IOFailure(error)


async def post_object(
    endpoint: str | None,
    object_type: ObjectType,
    data: dict | str,
) -> IOResult[httpx.Response, EXCEPTION]:
    request_kwargs = {
        "method": "POST",
        "params": {"object_type": str(object_type)},
        "url": f"{endpoint}/api/aggregate/",
    }

    match object_type:
        case ObjectType.AGGREGATE:
            assert isinstance(data, dict)
            request_kwargs["json"] = data

        case ObjectType.AGGREGATE_TEMPLATE:
            assert isinstance(data, str)
            request_kwargs["headers"] = {"Content-Type": "text/html"}
            request_kwargs["content"] = data.encode("utf-8")

    match await make_request(httpx.Request(**request_kwargs)):
        case IOSuccess(Success(response)):
            return IOSuccess(response)
        case IOFailure(Failure(error)):
            return IOFailure(error)
        case _:
            return IOFailure(Exception("Unknown error occurred while posting object"))
