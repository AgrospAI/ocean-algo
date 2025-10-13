from pathlib import Path

import pandas as pd
from ocean_runner import Algorithm

algorithm = Algorithm()


@algorithm.on_error
def error_callback(ex: Exception):
    algorithm.logger.exception(ex)
    raise algorithm.Error() from ex


@algorithm.validate
def val():
    assert algorithm.job_details.files, "Empty input dir"


@algorithm.run
def run() -> pd.DataFrame:
    _, filename = next(algorithm.job_details.next_path())
    return pd.read_csv(filename).describe(include="all")


@algorithm.save_results
def save(results: pd.DataFrame, path: Path):
    algorithm.logger.info(f"Descriptive statistics: {results}")
    results.to_csv(path / "results.csv")


if __name__ == "__main__":
    algorithm()
