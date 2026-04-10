import csv
from logging import Logger
from typing import IO, Callable

import pandas as pd


def load_as_csv(content: IO[bytes]) -> pd.DataFrame | None:
    try:
        sample = content.read(2048)
        content.seek(0)

        # Detect delimiter (comma, tab, semicolon)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample.decode())

        return pd.read_csv(content, sep=dialect.delimiter)
    except Exception:
        return None


def load_as_excel(content: IO[bytes]) -> pd.DataFrame | None:
    try:
        header = content.read(2)

        # xlsx files start with 'PK' (hex: 50 4B)
        if header == b"PK":
            return pd.read_excel(content, engine="openpyxl")
    except Exception:
        return None


def load_as_json(content: IO[bytes]) -> pd.DataFrame | None:
    try:
        prefix = content.read(50).strip()
        content.seek(0)

        if prefix.startswith((b"{", b"[")):
            return pd.read_json(content)
    except Exception:
        return None


def try_read_df(logger: Logger, content: IO[bytes]) -> pd.DataFrame | None:
    assert content.readable()

    logger.info("Identifying file format...")

    loaders: list[tuple[Callable[[IO[bytes]], pd.DataFrame | None], str]] = [
        (load_as_csv, "csv"),
        (load_as_json, "json"),
        (load_as_excel, "excel"),
    ]

    for loader, name in loaders:
        initial_position = content.tell()
        df = loader(content)

        if df is not None:
            logger.info("Detected format %s", name)
            return df
        content.seek(initial_position)

    logger.warning(
        "Did not detect any of the following formats %s",
        ", ".join([name for _, name in loaders]),
    )

    return None
