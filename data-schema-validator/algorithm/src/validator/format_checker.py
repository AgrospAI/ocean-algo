import csv
import logging
import xml.etree.ElementTree as ET

import filetype
import ijson
from rdflib import Graph

logger = logging.getLogger(__name__)


def detect_json_type(file_path: str) -> str | None:
    try:
        has_context = False
        with open(file_path, "rb") as file:
            parser = ijson.parse(file)
            for prefix, event, value in parser:
                if (prefix == "" and event == "map_key" and value == "@context") or (
                    prefix == "item" and event == "map_key" and value == "@context"
                ):
                    has_context = True

        return "application/ld+json" if has_context else "application/json"

    except (
        ValueError,
        ijson.common.IncompleteJSONError,
        ijson.common.JSONError,
        IOError,
    ) as e:
        logger.debug(f"File is not valid JSON: {e}")
        return None


def is_valid_rdf_xml(file_path: str) -> bool:
    try:
        g = Graph()
        g.parse(file_path, format="xml")
        return len(g) > 0
    except Exception as e:
        logger.debug(f"File is not valid RDF+XML: {e}")
        return False


def is_valid_xml(file_path: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            ET.fromstring(file.read(4096))
        return True
    except (ET.ParseError, IOError, UnicodeDecodeError) as e:
        logger.debug(f"File is not valid XML: {e}")
        return False


def is_valid_turtle(file_path: str) -> bool:
    try:
        g = Graph()
        g.parse(file_path, format="turtle")
        return True
    except Exception as e:
        logger.debug(f"File is not valid Turtle: {e}")
        return False


def is_valid_csv(file_path: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            dialect = csv.Sniffer().sniff(file.read(4096))
            if dialect.delimiter not in [",", ";", "\t", "|"]:
                return False
            return True
    except (csv.Error, UnicodeDecodeError, IOError) as e:
        logger.debug(f"File is not valid CSV: {e}")
        return False


def check_format(file_path: str) -> str:
    try:
        kind = filetype.guess(file_path)
        if kind is not None:
            return kind.mime
    except Exception as e:
        logger.error(f"Error reading magic bytes: {e}")

    json_type = detect_json_type(file_path)
    if json_type:
        return json_type

    if is_valid_rdf_xml(file_path):
        return "application/rdf+xml"
    if is_valid_xml(file_path):
        return "application/xml"

    if is_valid_turtle(file_path):
        return "text/turtle"

    if is_valid_csv(file_path):
        return "text/csv"

    return "application/octet-stream"
