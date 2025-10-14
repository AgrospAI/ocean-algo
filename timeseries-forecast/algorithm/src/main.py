from ocean_runner import Algorithm, Config
from implementation.data import InputParameters

from implementation.algorithm import run, save_results

Algorithm(
    Config(custom_input=InputParameters),
).validate().run(
    run
).save_results(save_results)
