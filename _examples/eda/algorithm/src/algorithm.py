import pandas as pd
from ocean_runner import Algorithm
from ydata_profiling import ProfileReport

algorithm = Algorithm()


@algorithm.run
def run(*args):
    """The algorithm entrypoint. This method does the following:

    1. Loads the input data from the first given file.
    1. Generate the summary report.

    """

    # Get input filepath
    _, filename = next(algorithm.job_details.next_path())
    df = pd.read_csv(filename)

    return ProfileReport(df, title="Profiling Report", sensitive=False)

@algorithm.save_results
def save(result, path, *args):
    result.to_file(path / "profiling_report.html")
