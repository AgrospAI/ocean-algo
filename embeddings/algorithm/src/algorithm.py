import json
from pathlib import Path

import requests
from ocean_runner import Algorithm, Config

from .data import InputParameters, Result

ResultsT = list[Result]

algorithm = Algorithm(config=Config(custom_input=InputParameters))


def embed_content(content: str) -> Result:
    parameters: InputParameters = algorithm.job_details.input_parameters

    request = requests.post(
        parameters.model_url,
        json={
            "model": parameters.model,
            "input": [content],
        },
        headers={"Authorization": f"Bearer {parameters.token}"},
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

    files = algorithm.job_details.files
    for file in files:
        for path in file.input_files:
            algorithm.logger.info(f"Adding file [{path.name}] for {file.did}")

            with open(path, "r") as f:
                embeddings = embed_content(f.read())

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
