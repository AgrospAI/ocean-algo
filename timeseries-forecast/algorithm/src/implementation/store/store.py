from abc import ABCMeta, abstractmethod
from io import BufferedWriter
from pathlib import Path
from typing import Callable, Self


class Store(metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "store") and callable(subclass.store) or NotImplemented

    @abstractmethod
    def store(
        self, path: Path, store_function: Callable[[BufferedWriter], None]
    ) -> Self:
        """Open a file in the given path and call the `store_function`

        Args:
            path (Path): Path to store the data.
            store_function (Callable): Function that stores the data.
        """

        raise NotImplementedError
