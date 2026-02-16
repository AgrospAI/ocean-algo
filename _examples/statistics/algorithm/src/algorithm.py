from pathlib import Path

import pandas as pd
from ocean_runner import Algorithm

algorithm = Algorithm()


@algorithm.on_error
def error_callback(algorithm: Algorithm, ex: Exception):
    algorithm.logger.exception(ex)
    raise algorithm.Error() from ex


@algorithm.validate
def val(algorithm: Algorithm):
    assert algorithm.job_details.files, "Empty input dir"


@algorithm.run
def run(algorithm: Algorithm) -> pd.DataFrame:
    _, filename = next(algorithm.job_details.inputs())
    return pd.read_csv(filename).describe(include="all")


@algorithm.save_results
def save(algorithm: Algorithm, result: pd.DataFrame, base: Path):
    algorithm.logger.info(f"Descriptive statistics: {result}")
    result.to_csv(base / "result.csv")


if __name__ == "__main__":
    algorithm()
