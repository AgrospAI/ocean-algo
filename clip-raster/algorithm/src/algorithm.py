import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from numpy.ma import MaskedArray
from ocean_runner import Algorithm, Config

from .data import InputParameters
from .geometry import Parcel, load_parcel
from .indices import compute_indices
from .raster import (
    GDAL_ENV,
    Grid,
    read_reflectance,
    read_scl,
    read_true_color,
    scene_grid,
    unusable_share,
    usable,
)
from .report import render_html, summary
from .scenes import select_scene, search_items

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Result:
    parcel: Parcel
    grid: Grid
    indices: dict[str, MaskedArray]
    true_color: np.ndarray | None
    summary: dict[str, Any]


algorithm = Algorithm[InputParameters, Result].create(Config(custom_input=InputParameters))


@algorithm.run
def run(_) -> Result:
    parameters = algorithm.job_details.input_parameters
    parcel = load_parcel([path for _, path in algorithm.job_details.inputs()])
    logger.info(f"Parcel: {len(parcel.properties)} feature(s), bounds {parcel.geometry.bounds}")

    with rasterio.Env(**GDAL_ENV):
        items = search_items(parcel.geometry, parameters.lookback_days)

        # The SCL read to rank a scene is reused to mask the chosen one.
        scl_cache: dict[str, tuple[Grid, np.ndarray]] = {}

        def cloud_fraction(item: dict[str, Any]) -> float:
            grid = scene_grid(item, parcel.geometry)
            scl_cache[item["id"]] = (grid, read_scl(item, grid))
            return unusable_share(scl_cache[item["id"]][1], grid)

        scene = select_scene(items, parcel.geometry, parameters.max_cloud_cover, cloud_fraction)
        grid, scl = scl_cache[scene.id]
        logger.info(f"Using scene {scene.id} ({scene.datetime})")

        valid = usable(scl, grid)
        if not valid.any():
            raise Algorithm.Error(f"Scene {scene.id} has no usable pixels over the parcel")
        bands = read_reflectance(scene.item, grid, valid)
        true_color = read_true_color(scene.item, grid)

    indices = compute_indices(bands)
    result_summary = summary(
        parcel, grid, scene, indices, parameters.lookback_days, parameters.max_cloud_cover
    )
    for name, stats in result_summary["indices"].items():
        logger.info(f"{name}: mean {stats.get('mean')} over {stats['validPixels']} pixels")
    return Result(parcel, grid, indices, true_color, result_summary)


@algorithm.save_results
def save(_, result: Result, base: Path):
    summary_path = base / "indices.json"
    summary_path.write_text(json.dumps(result.summary, indent=2, ensure_ascii=False))
    logger.info(f"Summary saved to {summary_path}")

    report_path = base / "report.html"
    report_path.write_text(
        render_html(result.summary, result.parcel, result.grid, result.indices, result.true_color),
        encoding="utf-8",
    )
    logger.info(f"Report saved to {report_path}")
