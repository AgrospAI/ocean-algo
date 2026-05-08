import json
import logging
from pathlib import Path

import aiofiles
from attr import dataclass
from ocean_runner import Algorithm, Config
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Result, Success

from src.data import Format, InputParameters
from src.validator.format_checker import check_format
from src.validator.validator import schema_validator


@dataclass(frozen=True)
class DataError:
    message: str
    details: str = ""

    def __str__(self) -> str:
        return f"{self.message}: {self.details}"


type ResultT = Result[
    list[IOResult[tuple[str, dict], Algorithm.Error]],
    DataError,
]

algorithm = Algorithm[InputParameters, ResultT].create(
    Config(custom_input=InputParameters)
)

logging.getLogger("asyncio").setLevel("WARNING")
algorithm.logger.setLevel("INFO")


@algorithm.validate
async def validate(_) -> None:
    assert algorithm.job_details.metadata, "DDOs missing"
    assert algorithm.job_details.files, "Files missing"
    algorithm.logger.info("Validation passed: Inputs and Metadata are present.")


def extract_metadata_from_ddo(did: str) -> tuple[str | None, str | None]:
    try:
        ddo_path = Path(f"/data/ddos/{did}")
        raw_json_string = ddo_path.read_text(encoding="utf-8")
        raw_ddo_dict = json.loads(raw_json_string)
        algorithm.logger.info(f"Extracted DDO for {did}")
        algorithm.logger.info(
            f"DDO input parameters: {algorithm.job_details.input_parameters}"
        )

        dct_format = (
            raw_ddo_dict.get("metadata", {})
            .get("additionalInformation", {})
            .get("dct:format")
        )
        conforms_to = (
            raw_ddo_dict.get("metadata", {})
            .get("additionalInformation", {})
            .get("dct:conformsTo")
        )

        return dct_format, conforms_to
    except Exception as e:
        algorithm.logger.warning(f"Failed to extract DDO metadata for {did}: {e}")
        return None, None


@algorithm.run
async def run(_) -> ResultT:
    parameters = algorithm.job_details.input_parameters
    algorithm.logger.info(f"Input parameters: {parameters}")
    results_list: list[IOResult[tuple[str, dict], Algorithm.Error]] = []

    for did, path in algorithm.job_details.inputs():
        file_path_str = str(path)
        algorithm.logger.info(f"Processing dataset: {did}")

        ddo_format, ddo_schema = extract_metadata_from_ddo(did)

        expected_format = (
            parameters.format if parameters.format != Format.NONE else ddo_format
        )

        expected_schema = (
            str(parameters.conformsTo) if parameters.conformsTo else ddo_schema
        )

        if not expected_schema:
            algorithm.logger.error(f"Missing schema URL for {did}")
            error_report = {
                "dataset": did,
                "status": "FAILED",
                "phase": "Schema Resolution",
                "error": "Missing Validation Schema",
                "details": "The 'conformsTo' URL was not found in the dataset metadata (DDO) and was not provided in the input parameters. Validation cannot proceed.",
            }
            results_list.append(IOSuccess((did, error_report)))
            continue

        detected_format = check_format(file_path_str)
        algorithm.logger.info(f"Detected format for {did}: {detected_format}")

        if expected_format and detected_format != str(expected_format):
            report = {
                "dataset": did,
                "status": "FAILED",
                "phase": "Format Validation",
                "details": f"Format mismatch: Metadata claims {expected_format}, but detected {detected_format}.",
            }
            results_list.append(IOSuccess((did, report)))
            continue

        validation_results = schema_validator(
            file_path_str, detected_format, expected_schema
        )

        report = {
            "dataset": did,
            "status": "COMPLETED",
            "detected_format": detected_format,
            "schema_url": expected_schema,
            "validation_results": validation_results,
        }
        results_list.append(IOSuccess((did, report)))

    return Success(results_list)


@algorithm.save_results
async def save(_, result: ResultT, base: Path) -> None:
    assert result is not None
    assert base is not None

    async def save_report(report_data: dict, file_path: Path) -> None:
        async with aiofiles.open(file_path, "w+", encoding="utf-8") as f:
            await f.write(json.dumps(report_data, indent=4))

    async def save_error(error: str, file_path: Path) -> None:
        async with aiofiles.open(file_path / "error.txt", "w+", encoding="utf-8") as f:
            await f.write(str(error))

    match result:
        case Success(batches):
            for batch in batches:
                match batch:
                    case IOSuccess(Success((did, report_dict))):
                        await save_report(report_dict, base / f"{did}_report.json")
                    case IOFailure(Failure(error)):
                        algorithm.logger.error(
                            f"Validation pipeline failed for a dataset: {error}"
                        )

        case Failure(error):
            await save_error(str(error), base)
