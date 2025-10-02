from dataclasses import dataclass, field
from io import BufferedWriter
from logging import Logger, getLogger
from pathlib import Path
from typing import Callable, Self

from implementation.store import store


@dataclass(frozen=True)
class FileSystemStore(store.Store):

    logger: Logger = field(repr=False, default_factory=lambda: getLogger(__name__))

    def store(
        self, path: Path, store_function: Callable[[BufferedWriter], None]
    ) -> Self:
        """Open a file in the given path and call the `store_function`

        Args:
            path (Path): Path to store the data.
            store_function (Callable): Function that stores the data.
        """

        with open(path, "wb") as f:
            try:
                store_function(f)
            except Exception as e:
                self.logger(f"Error in the saving process: {e}")

        return self
