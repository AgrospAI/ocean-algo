from pathlib import Path

from ocean_runner import Algorithm, Config, ParametrizedAlgorithm

from .data import InputParameters

# TODO: Change to real result type
type ResultsT = None
algorithm: ParametrizedAlgorithm[InputParameters, ResultsT] = Algorithm.create(
    Config(custom_input=InputParameters)
)


@algorithm.validate
async def validate(_) -> None:
    # Can remove this function to use the default behaviour. DEFAULT: Check DDOs and input files
    raise NotImplementedError()


@algorithm.run
async def run(_) -> ResultsT:
    # TODO: Implement algorithm run function
    raise NotImplementedError()


@algorithm.save_results
async def save(_, results: ResultsT, base_path: Path) -> None:
    # Remove this function to use the default behaviour. DEFAULT: Save to results.txt
    raise NotImplementedError()
