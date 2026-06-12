import numpy as np

import pyproj
import logging
import rasterio
import rasterio.mask
from pathlib import Path
from shapely.ops import transform
from ocean_runner import Algorithm

logger = logging.getLogger(__name__)

def require(band: Path | None) -> Path:
    if not band:
        logger.error('Missing band path')
        raise Algorithm.Error(f'Missing band path')
    return band


def get_band_path(band: str, res: str = '10m') -> Path | None:
    prefix = Path('/tmp/downloads/eodata/Sentinel-2/GRANULE')

    band_path = next(prefix.rglob(f'*{band}_{res}.jp2'), None)

    return band_path


def clip(geometry, band_path: Path) -> np.ndarray[np.float32]:

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
