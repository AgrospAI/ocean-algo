import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
from ocean_runner import Algorithm
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# Element84 Earth Search: public STAC catalogue of the Sentinel-2 L2A archive on AWS
# Open Data, served as Cloud-Optimized GeoTIFFs (no credentials needed).
STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-c1-l2a"

# Pages of the newest scenes are enough: each candidate costs one small SCL read.
SEARCH_LIMIT = 50
MAX_CANDIDATES = 12

type Item = dict[str, Any]


@dataclass
class Scene:
    item: Item
    coverage: float
    """Share of the parcel inside the scene footprint (0-1)."""
    cloud_fraction: float
    """Share of the parcel's pixels that are cloudy, shadowed or missing (0-1)."""
    meets_threshold: bool

    @property
    def id(self) -> str:
        return self.item["id"]

    @property
    def datetime(self) -> str:
        return self.item["properties"]["datetime"]


def search_items(
    geometry: BaseGeometry,
    lookback_days: int,
    now: datetime | None = None,
    post: Callable[..., requests.Response] = requests.post,
) -> list[Item]:
    """Sentinel-2 L2A items intersecting the parcel, newest first."""
    end = now or datetime.now(UTC)
    start = end - timedelta(days=lookback_days)
    body = {
        "collections": [COLLECTION],
        "intersects": mapping(geometry.convex_hull),
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": SEARCH_LIMIT,
    }
    try:
        response = post(STAC_SEARCH_URL, json=body, timeout=30)
        response.raise_for_status()
        items = response.json().get("features", [])
    except (requests.RequestException, ValueError) as e:
        raise Algorithm.Error(f"Sentinel-2 catalogue search failed: {e}") from e

    logger.info(f"Found {len(items)} Sentinel-2 scenes in the last {lookback_days} days")
    if not items:
        raise Algorithm.Error(
            f"No Sentinel-2 scene covers the parcel in the last {lookback_days} days"
        )
    return items


def candidates(items: list[Item], geometry: BaseGeometry) -> list[tuple[Item, float]]:
    """Items to try, newest first, with the share of the parcel they cover.

    Neighbouring MGRS tiles overlap, so the same acquisition often appears twice: only the
    tile covering most of the parcel is kept. Scenes that cover the whole parcel are
    preferred; the others are only used when none does.
    """
    by_acquisition: dict[tuple[str, str], tuple[Item, float]] = {}
    for item in items:
        props = item.get("properties", {})
        coverage = shape(item["geometry"]).intersection(geometry).area / geometry.area
        key = (props.get("platform", ""), props.get("datetime", "")[:10])
        if key not in by_acquisition or coverage > by_acquisition[key][1]:
            by_acquisition[key] = (item, coverage)

    ranked = sorted(
        by_acquisition.values(), key=lambda pair: pair[0]["properties"]["datetime"], reverse=True
    )
    full = [pair for pair in ranked if pair[1] >= 0.999]
    return (full or ranked)[:MAX_CANDIDATES]


def select_scene(
    items: list[Item],
    geometry: BaseGeometry,
    max_cloud_cover: float,
    cloud_fraction: Callable[[Item], float],
) -> Scene:
    """The newest scene whose cloudy share inside the parcel is at most `max_cloud_cover` %.

    When none qualifies, the clearest candidate is returned with `meets_threshold=False`.
    """
    best: Scene | None = None
    for item, coverage in candidates(items, geometry):
        try:
            fraction = cloud_fraction(item)
        except Exception as e:  # a broken asset must not stop the search
            logger.warning(f"Skipping scene {item.get('id')}: {e}")
            continue
        logger.info(f"Scene {item['id']}: {fraction:.0%} of the parcel is cloudy or missing")
        scene = Scene(item, coverage, fraction, fraction * 100 <= max_cloud_cover)
        if scene.meets_threshold:
            return scene
        if best is None or fraction < best.cloud_fraction:
            best = scene

    if best is None:
        raise Algorithm.Error("None of the Sentinel-2 scenes found could be read")
    logger.warning(
        f"No scene has at most {max_cloud_cover}% clouds over the parcel; "
        f"using the clearest one ({best.id}, {best.cloud_fraction:.0%})"
    )
    return best
