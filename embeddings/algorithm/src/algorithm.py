import logging
import zipfile
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Tuple
from urllib.parse import urlparse

import aiofiles
import httpx
import orjson
from ocean_runner import Algorithm, Config
from pydantic import BaseModel

from .data import InputParameters, Metadata, ModelSpecification, Result

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

ResultsT = list[Result]
algorithm = Algorithm[InputParameters, ResultsT](Config(custom_input=InputParameters))


def headers(parameters: InputParameters) -> Dict[str, str]:
    assert parameters.token, "API token is required for authentication"
    return {"Authorization": f"Bearer {parameters.token}"}


async def get_model_specifications(parameters: InputParameters) -> ModelSpecification:
    class ResponseData(BaseModel):
        name: str

    class EmbeddingModelResponse(BaseModel):
        data: List[ResponseData]

    def infer_url(embedding_url: str) -> str:
        parsed = urlparse(embedding_url)
        return f"{parsed.scheme}://{parsed.netloc}/api/models"

    def list_models(response: EmbeddingModelResponse) -> list[str]:
        return [model.name for model in response.data]

    def filter_models(models: list[str], included_word: str = "embed") -> list[str]:
        return [m for m in models if included_word in m]

    async with httpx.AsyncClient(
        headers=headers(parameters),
        timeout=parameters.timeout,
    ) as client:
        models_url = parameters.models_url or infer_url(parameters.embedding_url)
        response = await client.get(models_url, headers=headers(parameters))

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Models endpoint returned status code {e.response.status_code}: {e.response.text}"
            ) from e

        embedding_response = EmbeddingModelResponse.model_validate(response.json())

        # Model not specified -> log available models and select the first that contains word "embed"
        # Model specified but not found -> log warning, and choose first that contains word "embed"
        # Model specified and found -> use it

        available_models = list_models(embedding_response)

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
            algorithm.logger.info(
                f"Proceeding using specified model: {parameters.model}"
            )
            return ModelSpecification(
                name=parameters.model,
                url=parameters.embedding_url,
            )


async def embed_content(
    parameters: InputParameters,
    model_specification: ModelSpecification,
    content: str,
) -> List[List[float]]:
    try:
        async with httpx.AsyncClient(
            headers=headers(parameters),
            timeout=parameters.timeout,
        ) as client:
            response = await client.post(
                model_specification.url,
                json={"model": model_specification.name, "input": [content]},
            )
            response.raise_for_status()

            return response.json()["embeddings"]
    except httpx.HTTPStatusError as e:
        algorithm.logger.error(f"Error obtaining embedding: {e}")
        return []


async def contents(algorithm: Algorithm) -> AsyncGenerator[Tuple[str, Metadata], None]:
    for _, path in algorithm.job_details.inputs():
        algorithm.logger.info(f"Adding file [{path.name}]")

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zip_ref:
                for member in (m for m in zip_ref.namelist() if not m.endswith("/")):
                    with zip_ref.open(member) as extracted_file:
                        try:
                            yield (
                                extracted_file.read().decode("utf-8"),
                                Metadata(
                                    filepath=f"{path.name}:{member}",
                                    did=path.parent.name,
                                    idx=path.name,
                                ),
                            )
                        except UnicodeDecodeError:
                            algorithm.logger.warning(
                                f"Skipping non-text file [{member}] inside zip"
                            )
        else:
            async with aiofiles.open(path, "r") as f:
                yield (
                    await f.read(),
                    Metadata(
                        filepath=path.name,
                        did=path.parent.name,
                        idx=path.name,
                    ),
                )


@algorithm.run
async def run(algorithm: Algorithm) -> ResultsT:
    parameters = await algorithm.job_details.input_parameters()
    assert parameters is not None

    model = await get_model_specifications(parameters)
    return [
        Result(
            embeddings=await embed_content(parameters, model, content),
            metadata=metadata,
        )
        async for content, metadata in contents(algorithm)
    ]


@algorithm.save_results
async def save(
    algorithm: Algorithm,
    result: ResultsT,
    base: Path,
) -> None:
    for res in result:
        fname = "_".join(res.metadata.filepath.split("/")[-2:])
        async with aiofiles.open(base / f"{fname}_embedding.txt", "wb") as f:
            await f.write(orjson.dumps(res.model_dump()))


if __name__ == "__main__":
    algorithm()
