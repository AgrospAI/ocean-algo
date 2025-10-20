from pathlib import Path
import pandas as pd
from ocean_runner import Algorithm
from ydata_profiling import ProfileReport

algorithm = Algorithm()


@algorithm.run
def run():
    _, filename = next(algorithm.job_details.next_path())
    df = pd.read_csv(filename)

    return ProfileReport(df, title="Profiling Report", sensitive=False)


@algorithm.save_results
def save(result: ProfileReport, path: Path):
    result.to_file(path / "profiling_report.html")


if __name__ == "__main__":
    algorithm()
