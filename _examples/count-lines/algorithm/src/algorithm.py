import subprocess

from ocean_runner import Algorithm, EmptyAlgorithm

type ResultT = int
algorithm: EmptyAlgorithm[ResultT] = Algorithm[None, ResultT].create(None)
# Since we do not have custom input parameters, "Algorithm" will be of type "EmptyAlgorithm"


@algorithm.run
def run(algorithm: Algorithm) -> int:
    _, filename = next(algorithm.job_details.inputs())
    return int(subprocess.check_output(["wc", "-l", filename]).split()[0])


# We will not define a "@algorithm.save_results" since the default implementation
# works just fine, saving a results.txt.