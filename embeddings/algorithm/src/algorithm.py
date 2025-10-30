import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from ocean_runner import Algorithm, Config

from .data import InputParameters, ModelSpecification, Result

ResultsT = list[Result]
algorithm = Algorithm(config=Config(custom_input=InputParameters))


def headers(parameters: InputParameters) -> dict[str, str]:
    if not parameters.token:
        raise ValueError("API token is required for authentication.")
    return {"Authorization": f"Bearer {parameters.token}"}


def get_model_specifications() -> ModelSpecification:
    parameters: InputParameters = algorithm.job_details.input_parameters

    def infer_url(embedding_url: str) -> str:
        parsed = urlparse(embedding_url)
        return f"{parsed.scheme}://{parsed.netloc}/api/models"

    models_url = parameters.models_url or infer_url(parameters.embedding_url)

    request = requests.get(models_url, headers=headers(parameters))

    if request.status_code != 200:
        raise RuntimeError(
            f"Models endpoint returned status code {request.status_code}: {request.text}"
        )

    response = request.json()

    # Model not specified -> log available models and select the first that contains word "embed"
    # Model specified but not found -> log warning, and choose first that contains word "embed"
    # Model specified and found -> use it

    def list_models(models_response: dict) -> list[str]:
        return [model["name"] for model in models_response["data"]]

    def filter_models(models: list[str], included_word: str = "embed") -> list[str]:
        return [m for m in models if included_word in m]

    available_models = list_models(response)

    algorithm.logger.info(f"Available models: {available_models}")
    if not parameters.model or parameters.model not in available_models:
        if parameters.model:
            algorithm.logger.warning(
                f"Model {parameters.model} not found in available models. Selecting the first found embedding model."
            )
        else:
            algorithm.logger.info(
                "No model specified, selecting the first found embedding model."
            )

        embedding_models = filter_models(available_models, included_word="embed")
        if not embedding_models:
            raise RuntimeError("No embedding models found in available models.")

        embedding_model = embedding_models[0]
        algorithm.logger.info(f"Selected model: {embedding_model}")
        return ModelSpecification(
            name=embedding_model,
            url=parameters.embedding_url,
        )
    else:
        algorithm.logger.info(f"Proceeding using specified model: {parameters.model}")
        return ModelSpecification(
            name=parameters.model,
            url=parameters.embedding_url,
        )


def embed_content(model_specification: ModelSpecification, content: str) -> Result:
    request = requests.post(
        model_specification.url,
        json={"model": model_specification.name, "input": [content]},
        headers=headers(algorithm.job_details.input_parameters),
    )

    try:
        result = request.json()
        return result["embeddings"]
    except Exception as e:
        algorithm.logger.error(f"Error obtaining embedding: {e}")
        return []


@algorithm.run
def run() -> ResultsT:
    results: ResultsT = []

    model = get_model_specifications()

    files = algorithm.job_details.files
    for file in files:
        for path in file.input_files:
            algorithm.logger.info(f"Adding file [{path.name}] for {file.did}")

            with open(path, "r") as f:
                embeddings = embed_content(model, f.read())

                results.append(
                    Result(
                        embeddings=embeddings,
                        metadata={
                            "filepath": str(path),
                            "did": file.did,
                            "idx": path.name,
                            # Maybe add more metadata, like file type,
                        },
                    )
                )

    return results


@algorithm.save_results
def save(
    results: ResultsT,
    base_path: Path,
) -> None:
    for result in results:
        filename = "_".join(result.metadata["filepath"].split("/")[-2:])
        output = base_path / f"{filename}_embedding.txt"
        with open(output, "w") as f:
            json.dump(result.asdict(), f)


if __name__ == "__main__":
    algorithm()
