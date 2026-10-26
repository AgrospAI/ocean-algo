import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyproj
import rasterio
from affine import Affine
from numpy.ma import MaskedArray
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

logger = logging.getLogger(__name__)

# Earth Search asset keys of the bands the indices need.
BANDS = {
    "green": "green",  # B03, 10 m
    "red": "red",  # B04, 10 m
    "nir": "nir",  # B08, 10 m
    "nir08": "nir08",  # B8A, 20 m
    "swir16": "swir16",  # B11, 20 m
}
SCL = "scl"  # Scene Classification Layer, 20 m
TRUE_COLOR = "visual"  # TCI, 10 m RGB

# SCL classes that make a pixel unusable: no data, saturated or defective, cloud shadow,
# cloud (medium and high probability), thin cirrus and snow or ice.
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)

# GDAL settings for windowed reads of remote COGs.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
}

RESOLUTION = 10  # metres


@dataclass(frozen=True)
class Grid:
    """10 m pixel grid of a scene's tile, cropped to the parcel."""

    crs: str
    transform: Affine
    width: int
    height: int
    inside: np.ndarray
    """True for pixels whose centre lies inside the parcel."""


def project(geometry: BaseGeometry, src: str, dst: str) -> BaseGeometry:
    to_dst = pyproj.Transformer.from_crs(src, dst, always_xy=True).transform
    return transform(to_dst, geometry)


def scene_grid(item: dict[str, Any], geometry: BaseGeometry) -> Grid:
    """The tile's 10 m grid around the parcel, built from the item's projection metadata."""
    crs = f"EPSG:{item['properties']['proj:epsg']}"
    tile_transform = item["assets"]["red"]["proj:transform"]
    x0, y0 = tile_transform[2], tile_transform[5]
    local = project(geometry, "EPSG:4326", crs)
    minx, miny, maxx, maxy = local.bounds

    # Snap outwards to the tile's pixel edges, with one pixel of margin.
    col0 = math.floor((minx - x0) / RESOLUTION) - 1
    col1 = math.ceil((maxx - x0) / RESOLUTION) + 1
    row0 = math.floor((y0 - maxy) / RESOLUTION) - 1
    row1 = math.ceil((y0 - miny) / RESOLUTION) + 1
    grid_transform = Affine(
        RESOLUTION, 0, x0 + col0 * RESOLUTION, 0, -RESOLUTION, y0 - row0 * RESOLUTION
    )
    width, height = col1 - col0, row1 - row0

    def mask(all_touched: bool) -> np.ndarray:
        return geometry_mask(
            [local],
            out_shape=(height, width),
            transform=grid_transform,
            invert=True,
            all_touched=all_touched,
        )

    # Parcels thinner than a pixel have no pixel centre inside: keep the pixels they touch.
    inside = mask(False)
    if not inside.any():
        inside = mask(True)
    return Grid(crs, grid_transform, width, height, inside)


def read_on_grid(
    href: str, grid: Grid, resampling: Resampling, bands: int | list[int] = 1
) -> np.ndarray:
    """Read a remote raster resampled onto `grid`. Only the overlapping blocks are fetched."""
    with rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
        ) as vrt:
            return vrt.read(bands)


def read_scl(item: dict[str, Any], grid: Grid) -> np.ndarray:
    return read_on_grid(item["assets"][SCL]["href"], grid, Resampling.nearest)


def usable(scl: np.ndarray, grid: Grid) -> np.ndarray:
    """Pixels inside the parcel that the SCL does not flag as unusable."""
    return grid.inside & ~np.isin(scl, INVALID_SCL)


def unusable_share(scl: np.ndarray, grid: Grid) -> float:
    """Share of the parcel's pixels that the SCL flags as unusable."""
    return float(np.isin(scl[grid.inside], INVALID_SCL).mean())


def read_reflectance(
    item: dict[str, Any], grid: Grid, valid: np.ndarray
) -> dict[str, MaskedArray]:
    """Surface reflectance of each band on the grid, masked outside `valid`."""
    out = {}
    for name, key in BANDS.items():
        asset = item["assets"][key]
        meta = (asset.get("raster:bands") or [{}])[0]
        nodata = meta.get("nodata", 0)
        scale, offset = meta.get("scale", 1.0), meta.get("offset", 0.0)
        native = meta.get("spatial_resolution", RESOLUTION)
        resampling = Resampling.nearest if native == RESOLUTION else Resampling.bilinear

        dn = read_on_grid(asset["href"], grid, resampling).astype("float32")
        out[name] = np.ma.masked_array(dn * scale + offset, mask=~valid | (dn == nodata))
        logger.info(f"Read {name} ({key}) on a {grid.width}x{grid.height} grid")
    return out


def read_true_color(item: dict[str, Any], grid: Grid) -> np.ndarray | None:
    """8-bit RGB true colour image on the grid (bands first), or None if unavailable."""
    asset = item["assets"].get(TRUE_COLOR)
    if not asset:
        return None
    try:
        return read_on_grid(asset["href"], grid, Resampling.nearest, [1, 2, 3])
    except RasterioError as e:
        logger.warning(f"Could not read the true colour image: {e}")
        return None
