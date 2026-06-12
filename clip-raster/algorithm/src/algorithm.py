import numpy as np
import logging
import geopandas as gpd
from pathlib import Path
from .raster import clip
from .viz import draw_raster
from .s3 import download_product
from numpy.ma import MaskedArray
from .data import InputParameters
from .indices import compute_indices
from .utils import require, get_band_path
from ocean_runner import Algorithm, Config
from .indices import ndvi, gndvi, ndwi, ndmi
from shapely.geometry.base import BaseMultipartGeometry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATASTRO_URL_PREFIX = 'http://ovc.catastro.meh.es/INSPIRE/wfsCP.aspx?service=WFS&version=2.0.0&request=GetFeature&STOREDQUERY_ID=GetParcel&refcat=' 


def geometry_to_clip(geometry: BaseMultipartGeometry, *args) -> np.ndarray[np.float32]:
    value = require(get_band_path(*args))
    return clip(geometry, value)


def compute_indices(geometry):
    green_band = require(get_band_path('B03'))
    green_band_20m = require(get_band_path('B03', '20m'))
    red_band = require(get_band_path('B04'))
    infrared_band = require(get_band_path('B08'))
    narrow_infrared_band_20m = require(get_band_path('B8A', '20m'))
    swir_1_band_20m = require(get_band_path('B11', '20m'))

    green = clip(geometry, green_band)
    green_20m = clip(geometry, green_band_20m)
    red = clip(geometry, red_band)
    infrared = clip(geometry, infrared_band)
    narrow_infrared = clip(geometry, narrow_infrared_band_20m)
    swir_1 = clip(geometry, swir_1_band_20m)

    ndvi_ = ndvi(red, infrared)
    gndvi_ = gndvi(green, infrared)
    ndwi_ = ndwi(green_20m, swir_1)
    ndmi_ = ndmi(narrow_infrared, swir_1)

    return ndvi_, gndvi_, ndwi_, ndmi_


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
