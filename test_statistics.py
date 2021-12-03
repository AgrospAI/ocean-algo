import os
from unittest import TestCase, mock
from statistics import descriptive_statistics, get_job_details


class Test(TestCase):

    @mock.patch.dict(os.environ, {"DIDS": " [ \"8f67E08be5dD941a701c2491E814535522c33bC2\" ]"})
    @mock.patch.dict(os.environ, {"TRANSFORMATION_DID": "6EDaE15f7314dC306BB6C382517D374356E6B9De"})
    @mock.patch.dict(os.environ, {"secret": "MOCK-SECRET"})
    @mock.patch.dict(os.environ, {"ROOT_FOLDER": os.path.dirname(os.path.realpath(__file__))})
    def test_line_counter(self):
        expected = """,Apple_ID,Diameter [mm],Apple centre X [m],Apple centre Y [m],Apple centre Z [m]
count,10,10.0,10.0,10.0,10.0
unique,10,,,,
top,2018_01_001,,,,
freq,1,,,,
mean,,61.529999999999994,-0.29760604944076113,-0.0829260696913199,0.5602932110560095
std,,15.862047646995503,0.10960086709139778,0.16880661539065187,0.21081778596122974
min,,46.3,-0.443178688514008,-0.357916632301215,0.170980372740605
25%,,51.0,-0.38570813186701625,-0.13468551035547527,0.4208598281823085
50%,,52.599999999999994,-0.319747782744784,-0.06282366167890635,0.549630641079167
75%,,76.325,-0.23563831715640926,0.039403688207511355,0.7370597213725636
max,,86.0,-0.124847314680155,0.128873791673685,0.853300025891764
"""
        descriptive_statistics(get_job_details())
        root = os.getenv('ROOT_FOLDER', '')
        with open(root + '/data/outputs/result') as f:
            actual = f.read()
            self.assertEqual(actual, expected)
