from enum import StrEnum, auto

import httpx
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Success


class ObjectType(StrEnum):
    """Different registered object types"""

    AGGREGATE = auto()
    AGGREGATE_SCHEMA = auto()
    BENCHMARKING_TEMPLATE = auto()
    BENCHMARKING_TRANSLATIONS = auto()
    CONFIG_SCHEMA = auto()
    HEALTHCHECK = auto()


type EXCEPTION = httpx.TransportError | httpx.HTTPStatusError | httpx.InvalidURL
EXCEPTIONS = (httpx.TransportError, httpx.HTTPStatusError, httpx.InvalidURL)


async def make_request(request: httpx.Request) -> IOResult[httpx.Response, EXCEPTION]:
    """Run an httpx request

    Args:
        request (httpx.Request): Request to be done

    Returns:
        IOResult[httpx.Response, EXCEPTION]: Result with a response or one of the defined exceptions
    """

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
        case ObjectType.AGGREGATE_SCHEMA:
            endpoint = f"{endpoint}/api/schemas/aggregate/"
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
