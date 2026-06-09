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

    try:
        response = requests.get(SIGPAC_API_URL, params=parameters.to_sigpac_params(), timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info(f'SIGPAC data fetched successfully: {data}')
        return [data]

    except requests.RequestException as e:
        logger.error(f'Error fetching SIGPAC data: {e}')
        raise Algorithm.Error(f'Error fetching SIGPAC data: {e}')


@algorithm.save_results
def save(_,result: ResultsT, base: Path):
    output_path = base / 'sigpac_data.json'

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result[0], file, indent=2, ensure_ascii=False)
        logger.info(f'SIGPAC data saved to {output_path}')