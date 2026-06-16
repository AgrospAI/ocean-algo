import pyproj
import logging
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.ma import MaskedArray
from shapely.ops import transform
from ocean_runner import Algorithm

logger = logging.getLogger(__name__)

BANDS = {
    'blue': ('B02', '10m'),
    'green': ('B03', '10m'),
    'green_20m': ('B03', '20m'),
    'red': ('B04', '10m'),
    'red_edge_1': ('B05', '20m'),
    'red_edge_2': ('B06', '20m'),
    'red_edge_3': ('B07', '20m'),
    'infrared': ('B08', '10m'),
    'narrow_infrared': ('B8A', '20m'),
    'swir_1': ('B11', '20m'),
    'swir_2': ('B12', '20m')
}


def require(band: Path | None) -> Path:
    if not band:
        logger.error('Missing band path')
        raise Algorithm.Error('Missing band path')
    return band


def get_band_path(band: str) -> Path | None:
    prefix = Path('/tmp/downloads/eodata/Sentinel-2/GRANULE')

    code, resolution = BANDS[band]
    band_path = next(prefix.rglob(f'*{code}_{resolution}.jp2'), None)

    return band_path


def clip(geometry, band_path: Path):
    with rasterio.open(band_path) as band_image:
        project = pyproj.Transformer.from_crs(
            'EPSG:4326',
            band_image.crs,
            always_xy=True,
        ).transform

        geom_utm = transform(project, geometry)
        
        band_out, _ = rasterio.mask.mask(
            band_image,
            [geom_utm],
            crop=True,
            filled=False,
        )

    return band_out.squeeze().astype('float32')


def save_as_img(data: dict[str, MaskedArray], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    color_mappings = {
        'ndvi': 'RdYlGn',
        'gndvi': 'RdYlGn',
        'ndwi': 'Blues',
        'ndmi': 'BrBG'
    }

    vmin_mappings = {
        'ndwi': -1,
        'ndmi': -1
    }

    for ax, name, values in zip(axes.flat, data.keys(), data.values()):
        vmin = vmin_mappings.get(name, 0)
        im = ax.imshow(values, cmap=color_mappings.get(name), vmin=vmin, vmax=1)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')