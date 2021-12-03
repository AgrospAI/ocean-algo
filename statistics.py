import os
import json
import pandas as pd


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
            # get the ddo from disk
            filename = root + '/data/ddos/' + did
            print(f'Reading json from {filename}')
            with open(filename) as json_file:
                ddo = json.load(json_file)
                # search for metadata service
                for service in ddo['service']:
                    if service['type'] == 'metadata':
                        job['files'][did] = list()
                        index = 0
                        for file in service['attributes']['main']['files']:
                            job['files'][did].append(
                                root + '/data/inputs/' + did + '/' + str(index))
                            index = index + 1
    if algo_did is not None:
        job['algo']['did'] = algo_did
        job['algo']['ddo_path'] = root + '/data/ddos/' + algo_did
    return job


def descriptive_statistics(job_details):
    root = os.getenv('ROOT_FOLDER', '')
    print('Starting compute job with the following input information:')
    print(json.dumps(job_details, sort_keys=True, indent=4))
    """ Computes descriptive statistics for the first file in first did """
    first_did = job_details['dids'][0]
    filename = job_details['files'][first_did][0]
    df = pd.read_csv(filename)
    stats = df.describe(include='all')
    print('Descriptive statistics for %s:\n%s' % (filename, stats))
    """ Write statistics to generate algo output """
    stats.to_csv(root + '/data/outputs/result')


if __name__ == '__main__':
    descriptive_statistics(get_job_details())
