import subprocess

from ocean_runner import Algorithm

algorithm = Algorithm()


@algorithm.run
def run() -> int:
    _, filename = next(algorithm.job_details.next_path())
    return int(subprocess.check_output(["wc", "-l", filename]).split()[0])


if __name__ == "__main__":
    algorithm()