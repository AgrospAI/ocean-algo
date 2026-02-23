import os
import cv2
from data import InputParameters
from ocean_runner import Algorithm, Config

algorithm = Algorithm(config=Config(custom_input=InputParameters))

input_parameters: InputParameters = algorithm.job_details.input_parameters

image_digest = input_parameters.model_image_digest_reference

os.execv("/bin/bash", ["bash", "/algorithm/src/run.sh", image_digest])