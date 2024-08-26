import os
import sys
import json
import pandas as pd
from ydata_profiling import ProfileReport


def get_job_details():
    root = os.getenv('ROOT_FOLDER', '')
    """Reads in metadata information about assets used by the algo"""
    job = dict()
    job['dids'] = json.loads(os.getenv('DIDS', None))
    job['metadata'] = dict()
    job['files'] = dict()
    job['algo'] = dict()
    job['secret'] = os.getenv('secret', None)
    algo_did = os.getenv('TRANSFORMATION_DID', None)
    if job['dids'] is not None:
        for did in job['dids']:
            job['files'][did] = list()
            # Just one file for DID with name "0"
            job['files'][did].append(root + '/data/inputs/' + did + '/0')
    if algo_did is not None:
        job['algo']['did'] = algo_did
        job['algo']['ddo_path'] = root + '/data/ddos/' + algo_did
    return job


def exploratory_data_analysis(job_details, sensitive=True):
    root = os.getenv('ROOT_FOLDER', '')
    print('Starting compute job with the following input information:')
    print(json.dumps(job_details, sort_keys=True, indent=4))
    """ Preparing exploratory data analysis report for the first file in first did """
    first_did = job_details['dids'][0]
    filename = job_details['files'][first_did][0]
    df = pd.read_csv(filename, engine='python', sep=None)
    profile = ProfileReport(df, title="Profiling Report", sensitive=sensitive)  # , tsmode=True)
    print('Generated profiling report for %s' % filename)
    """ Write profiling report to output """
    profile.to_file(root + '/data/outputs/Profiling_Report.html')


if __name__ == '__main__':
    isDataSensitive = True
    if len(sys.argv) > 1 and (sys.argv[1] == 'False' or sys.argv[1] == 'false'):
        isDataSensitive = False
    exploratory_data_analysis(get_job_details(), isDataSensitive)
