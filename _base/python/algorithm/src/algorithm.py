from pathlib import Path
from typing import TypeVar

from ocean_runner import Algorithm, Config

from .data import InputParameters

# Change to real result type
ResultsT = TypeVar("ResultsT")
algorithm = Algorithm(config=Config(custom_input=InputParameters))


@algorithm.validate
def validate(algorithm: Algorithm) -> None:
    # Can remove this function to use the default behaviour. DEFAULT: Check DDOs and input files
    raise NotImplementedError()


@algorithm.run
def run(algorithm: Algorithm, **kwargs) -> ResultsT:
    raise NotImplementedError()


@algorithm.save_results
def save(
    results: ResultsT,
    algorithm: Algorithm,
    base_path: Path,
) -> None:
    # Can remove this function to use the default behaviour. DEFAULT: Save to results.txt
    raise NotImplementedError()


if __name__ == "__main__":
    algorithm()
