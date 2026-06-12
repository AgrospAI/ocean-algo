import logging
import geopandas as gpd
from pathlib import Path
from numpy.ma import MaskedArray
from .data import InputParameters
from .raster import clip, save_as_img
from .download import download_product
from ocean_runner import Algorithm, Config
from .raster import require, get_band_path
from .indices import INDEXES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATASTRO_URL_PREFIX = 'http://ovc.catastro.meh.es/INSPIRE/wfsCP.aspx?service=WFS&version=2.0.0&request=GetFeature&STOREDQUERY_ID=GetParcel&refcat=' 


def compute_indices(geometry):
    return {
        index: fn(clip(geometry, require(get_band_path(a))), 
                  clip(geometry, require(get_band_path(b))))
        for index, (fn, a, b) in INDEXES.items()
    }


type ResultsT = dict[str, MaskedArray]

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
    
    return compute_indices(geometry)


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    out_path = base / 'indices.png'

    save_as_img(result, out_path)