from pathlib import Path
import json
import time
import os
from ocean_runner import Algorithm
from dotenv import load_dotenv

load_dotenv("/data/transformations/algorithm")

algorithm = Algorithm()

Results = dict[str, list[str]]


@algorithm.run
def run(algorithm: Algorithm) -> Results:
    files = algorithm.job_details.files

    results: Results = {}

    print("Environment Variables:")
    for key, value in os.environ.items():
        print(f"{key}={value}")

    for file in files:
        path_names = []
        print(f"Processing file with DID: {file.did}")
        for path in file.input_files:
            path_names.append(path.name)
            print(f"  Input file path: {path}")

        results[file.did] = path_names

    print("Waiting for 5 minutes before saving results...")
    time.sleep(300)

    return results


@algorithm.save_results
def save(
    algorithm: Algorithm,
    result: Results,
    base: Path,
) -> None:
    output_file = base / "results.json"
    with open(output_file, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    algorithm()
