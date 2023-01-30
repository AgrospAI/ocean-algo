import os
from unittest import TestCase, mock
from eda import exploratory_data_analysis, get_job_details


class Test(TestCase):

    @mock.patch.dict(os.environ, {"DIDS": " [ \"8f67E08be5dD941a701c2491E814535522c33bC2\" ]"})
    @mock.patch.dict(os.environ, {"TRANSFORMATION_DID": "6EDaE15f7314dC306BB6C382517D374356E6B9De"})
    @mock.patch.dict(os.environ, {"secret": "MOCK-SECRET"})
    @mock.patch.dict(os.environ, {"ROOT_FOLDER": os.path.dirname(os.path.realpath(__file__))})
    def test_eda(self):

        exploratory_data_analysis(get_job_details())
        root = os.getenv('ROOT_FOLDER', '')
        with open(root + '/data/outputs/Profiling_Report.html') as f:
            self.assertIsNotNone(f)
