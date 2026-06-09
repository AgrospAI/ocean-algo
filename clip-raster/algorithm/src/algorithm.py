import json
import requests
import logging
from typing import Any
from pathlib import Path
from .data import InputParameters
from ocean_runner import Algorithm, Config

logger = logging.getLogger(__name__)

type ResultsT = list[dict[str, Any]]

algorithm = Algorithm[InputParameters, ResultsT].create(
    Config(custom_input=InputParameters)
)

SIGPAC_API_URL = "https://sigpac-hubcloud.es/ogcapi/collections/recintos/items"

@algorithm.run
def run(_) -> ResultsT:
    parameters = algorithm.job_details.input_parameters

    return list(dict())

@algorithm.save_results
def save(_,result: ResultsT, base: Path):
    output_path = base / 'YOUR_FILE'

