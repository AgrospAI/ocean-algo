import pandas as pd
from ocean_runner import Algorithm

algorithm = Algorithm()


@algorithm.run
def run(*args) -> pd.DataFrame:
    _, filename = next(algorithm.job_details.next_path())
    return pd.read_csv(filename).describe(include="all")


@algorithm.save_results
def save_result(results: pd.DataFrame, path, *args) -> None:
    algorithm.logger.info(f"Descriptive statistics: {results}")
    results.to_csv(path / "results.csv")


if __name__ == "__main__":
    algorithm()
