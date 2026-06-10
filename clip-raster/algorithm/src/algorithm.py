import os
import boto3
import tqdm
import logging
import requests
import geopandas as gpd
from time import sleep
from typing import Any
from pathlib import Path
from .viz import draw_raster
from numpy.ma import MaskedArray
from shapely.geometry import box
from .data import InputParameters
from .utils import format_filename
from .indices import compute_indices
from ocean_runner import Algorithm, Config
from .auth import get_copernicus_token, request_temp_s3_creds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATASTRO_URL_PREFIX = 'http://ovc.catastro.meh.es/INSPIRE/wfsCP.aspx?service=WFS&version=2.0.0&request=GetFeature&STOREDQUERY_ID=GetParcel&refcat=' 
ODATA_BASE_URL      = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
S3_ENDPOINT_URL     = "https://eodata.dataspace.copernicus.eu",


def download_file_s3(s3, bucket_name, s3_key, local_path, failed_downloads):
    try:
        file_size = s3.head_object(Bucket=bucket_name, Key=s3_key)['ContentLength']
        formatted_filename = format_filename(os.path.basename(local_path))
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=formatted_filename, ncols=80, bar_format='{desc:.40}|{bar:20}| {percentage:3.0f}% {n_fmt}/{total_fmt}B') as pbar:
            def progress_callback(bytes_transferred):
                pbar.update(bytes_transferred)

            s3.download_file(bucket_name, s3_key, local_path, Callback=progress_callback)
    except Exception as e:
        logger.error(f'Failed to download {s3_key}. Error: {e}')
        failed_downloads.append(s3_key)


def traverse_and_download_s3(s3_resource, bucket_name, base_s3_path, local_path, failed_downloads):
    bucket = s3_resource.Bucket(bucket_name)
    files = bucket.objects.filter(Prefix=base_s3_path)

    for obj in files:
        s3_key = obj.key
        relative_path = os.path.relpath(s3_key, base_s3_path)
        local_path_file = os.path.join(local_path, relative_path)
        local_dir = os.path.dirname(local_path_file)
        os.makedirs(local_dir, exist_ok=True)
        download_file_s3(s3_resource.meta.client, bucket_name, s3_key, local_path_file, failed_downloads)


def download_product(geometry) -> None:
    bounds = geometry.bounds
    box_ = box(*bounds)
    wkt_string = box_.wkt

    filter = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_string}') and "
        f"ContentDate/Start gt 2026-05-01T00:00:00.000Z"
    )

    params = {
        '$filter': filter,
        '$top': 1,
        '$orderby': 'ContentDate/Start desc'
    }

    try:
        response = requests.get(ODATA_BASE_URL, params=params, timeout=15)

        if response.status_code != 200:
            logger.error(f'Error calling Copernicus API. Status code = {response.status_code}')
            raise Algorithm.Error(f'Error calling Copernicus API. Status code = {response.status_code}')
        
        else:
            data = response.json()

            s3_path = data['value'][0]['S3Path']

            token = get_copernicus_token()

            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json'
            }

            s3_credentials = request_temp_s3_creds(headers)

            sleep(5)

            s3_resource = boto3.resource('s3',
                                         endpoint_url=S3_ENDPOINT_URL,
                                         aws_access_key_id=s3_credentials['access_id'],
                                         aws_secret_access_key=s3_credentials['secret']
                                         )
            
            out_dir = Path('/tmp/downloads/eodata/Sentinel-2/')
            out_dir.mkdir(parents=True, exist_ok=True)

            bucket_name, base_s3_path = s3_path.lstrip('/').split('/', 1)
            
            failed_downloads = []
            traverse_and_download_s3(s3_resource, bucket_name, base_s3_path, out_dir, failed_downloads)
            
            logger.info(f'Failed downloads: {failed_downloads}')

    except Exception as e:
        logger.error(f'Error downloading product: {e}')
        raise Algorithm.Error(f'Error downloading product: {e}')


type ResultsT = list[MaskedArray]

algorithm = Algorithm[InputParameters, ResultsT].create(
    Config(custom_input=InputParameters)
)

@algorithm.run
def run(_) -> ResultsT:
    parameters = algorithm.job_details.input_parameters

    refcat = parameters.refcat

    catastro_url = f'{CATASTRO_URL_PREFIX}{refcat}'
    gdf = gpd.read_file(catastro_url)

    geometry = gdf.geometry.item()

    download_product(geometry)
    ndvi, gndvi, ndwi, ndmi = compute_indices(geometry)

    draw_raster(ndvi, 'ndvi', 'RdYlGn')
    draw_raster(gndvi, 'gndvi', 'RdYlGn')
    draw_raster(ndwi, 'ndwi', 'Blues')
    draw_raster(ndmi, 'ndmi', 'BrBG')


@algorithm.save_results
def save(_,result: ResultsT, base: Path):
    output_path = base / 'YOUR_FILE'
