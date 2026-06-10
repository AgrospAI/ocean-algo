from click import Path
from numpy.ma import MaskedArray
from shapely.ops import transform
from .utils import require, get_band_path
import pyproj
import rasterio
import rasterio.mask

def ndvi(red: MaskedArray, infrared: MaskedArray) -> MaskedArray:
    out = (infrared - red) / (red + infrared)
    return out


def gndvi(green: MaskedArray, infrared: MaskedArray) -> MaskedArray:
    out = (infrared - green) / (green + infrared)
    return out


def ndwi(green: MaskedArray, swir_1: MaskedArray) -> MaskedArray:
    out = (green - swir_1) / (green + swir_1)
    return out


def ndmi(narrow_infrared: MaskedArray, swir_1: MaskedArray) -> MaskedArray:
    out = (narrow_infrared - swir_1) / (narrow_infrared + swir_1)
    return out


def clip(geometry, band_path: Path):

    project = pyproj.Transformer.from_crs(
        'EPSG:4326',
        'EPSG:32631',
        always_xy=True,
    ).transform

    geom_utm = transform(project, geometry)
    
    with rasterio.open(band_path) as band_image:
        band_out, _ = rasterio.mask.mask(
            band_image,
            [geom_utm],
            crop=True,
            filled=False,
        )

    return band_out.squeeze().astype('float32')


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