from logging import getLogger
from typing import Mapping, Optional, Tuple, TypeVar

T = TypeVar("T")
logger = getLogger(__name__)


def get(
    f: Mapping[str, T],
    key: str,
    default: Optional[T] = None,
) -> Tuple[T, bool]:

    if key in f.keys():
        return (f.get(key), True)

    if default is None:
        raise KeyError(f"Key {key} not found")

    logger.info(f"Key {key} not found, returning default value {default}")
    return (default, False)
