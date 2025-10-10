import pandas as pd
from ocean_runner import Algorithm, Config

from implementation import InputParameters, run, save_data, validate

Algorithm[InputParameters, pd.DataFrame](
    Config(custom_input=InputParameters),
).validate(validate).run(run).save_results(save_data)
