import json
import logging

import requests
import validators
import xmlschema
from csvw.metadata import Table
from jsonschema import ValidationError, validate
from pyshacl import validate as shacl_validate

logger = logging.getLogger(__name__)


def validate_json_schema(file_path: str, schema_url: str) -> dict:
    try:
        if validators.url(schema_url):
            schema_response = requests.get(schema_url)
            schema_response.raise_for_status()
            schema_rules = schema_response.json()

        else:
            with open(schema_url, "r", encoding="utf-8") as file:
                schema_rules = json.load(file)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        validate(instance=data, schema=schema_rules)
        return {
            "conforms": True,
            "details": "JSON dataset matches the schema.",
        }

    except ValidationError as e:
        return {
            "conforms": False,
            "details": f"Validation failed at path {e.json_path}: {e.message}",
        }
    except Exception as e:
        return {
            "conforms": False,
            "details": f"System error during JSON validation: {str(e)}",
        }


def validate_xml_xsd(file_path: str, schema_url: str) -> dict:
    try:
        schema = xmlschema.XMLSchema(schema_url)
        schema.validate(file_path)
        return {"conforms": True, "details": "XML dataset matches the XSD schema."}

    except xmlschema.XMLSchemaValidationError as e:
        return {"conforms": False, "details": f"XML Validation error: {e.reason}"}
    except Exception as e:
        return {
            "conforms": False,
            "details": f"System error during XML validation: {str(e)}",
        }


def validate_rdf_shacl(
    file_path: str,
    schema_url: str,
    rdf_format: str = "turtle",
    shacl_format: str = "turtle",
) -> dict:
    try:
        conforms, results_graph, results_text = shacl_validate(
            data_graph=file_path,
            data_graph_format=rdf_format,
            shacl_graph=schema_url,
            shacl_graph_format=shacl_format,
            inference="rdfs",
        )

        if conforms:
            return {"conforms": True, "details": "RDF graph passes all SHACL shapes."}
        else:
            return {
                "conforms": False,
                "details": "Not fit with SHACL shapes.",
                "shacl_report": results_text,
            }

    except Exception as e:
        return {
            "conforms": False,
            "details": f"System error during SHACL validation: {str(e)}",
        }


def validate_csvw(file_path: str, schema_url: str) -> dict:
    try:
        if validators.url(schema_url):
            schema_response = requests.get(schema_url)
            schema_response.raise_for_status()
            schema_rules = schema_response.json()
        else:
            with open(schema_url, "r", encoding="utf-8") as file:
                schema_rules = json.load(file)

        schema_rules["url"] = file_path
        table = Table.fromvalue(schema_rules)
        for row in table:
            pass

        return {"conforms": True, "details": "CSV dataset matches the CSVW schema."}

    except ValueError as e:
        return {"conforms": False, "details": f"CSVW datatype Error: {str(e)}"}

    except Exception as e:
        return {"conforms": False, "details": f"CSVW validation error: {str(e)}"}


def schema_validator(file_path: str, mime_type: str, conforms_to_url: str) -> dict:
    if not conforms_to_url:
        return {
            "conforms": None,
            "details": "Validation skipped: No schema URL provided.",
        }

    logger.info(f"Validating {mime_type} against {conforms_to_url}")

    if mime_type == "application/json":
        return validate_json_schema(file_path, conforms_to_url)

    elif mime_type == "application/xml":
        return validate_xml_xsd(file_path, conforms_to_url)

    elif mime_type in ["text/turtle", "application/ld+json", "application/rdf+xml"]:
        rdf_fmt = (
            "turtle"
            if mime_type == "text/turtle"
            else ("json-ld" if "json" in mime_type else "xml")
        )
        shacl_fmt = "turtle"
        if conforms_to_url.endswith(".rdf") or conforms_to_url.endswith(".xml"):
            shacl_fmt = "xml"
        elif conforms_to_url.endswith(".jsonld") or conforms_to_url.endswith(".json"):
            shacl_fmt = "json-ld"

        return validate_rdf_shacl(
            file_path, conforms_to_url, rdf_format=rdf_fmt, shacl_format=shacl_fmt
        )

    elif mime_type == "text/csv":
        return validate_csvw(file_path, conforms_to_url)
    else:
        return {
            "conforms": False,
            "details": f"No schema validation engine available for format: {mime_type}",
        }
