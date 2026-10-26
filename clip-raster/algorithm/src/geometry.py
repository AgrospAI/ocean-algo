import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocean_runner import Algorithm
from shapely import wkt
from shapely.errors import ShapelyError
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

GEOMETRY_TYPES = {"Polygon", "MultiPolygon"}


@dataclass
class Parcel:
    """Parcel geometry (EPSG:4326) and the attributes of the features that form it."""

    geometry: BaseGeometry
    properties: list[dict[str, Any]] = field(default_factory=list)


def wkt_features(records: list[Any]) -> list[dict[str, Any]]:
    """Turn SIGPAC `format=json` records (attributes plus a `wkt` geometry) into features.

    Their `srid` is 4258 (ETRS89), which matches WGS84 to well under a Sentinel-2 pixel.
    """
    features = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("wkt"), str):
            continue
        try:
            geometry = mapping(wkt.loads(record["wkt"]))
        except ShapelyError as e:
            raise Algorithm.Error(f"Invalid WKT geometry in SIGPAC record: {e}") from e
        properties = {k: v for k, v in record.items() if k not in ("wkt", "srid")}
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    return features


def features_of(data: Any) -> list[dict[str, Any]]:
    """Return the features of a GeoJSON FeatureCollection, Feature or bare geometry,
    or of a list of SIGPAC `format=json` records."""
    if isinstance(data, list):
        return wkt_features(data)
    if not isinstance(data, dict):
        return []
    match data.get("type"):
        case "FeatureCollection":
            return [f for f in data.get("features") or [] if isinstance(f, dict)]
        case "Feature":
            return [data]
        case kind if kind in GEOMETRY_TYPES:
            return [{"type": "Feature", "geometry": data, "properties": {}}]
    return []


def load_parcel(files: list[Path]) -> Parcel:
    """Merge the polygons of every GeoJSON input into one parcel."""
    if not files:
        raise Algorithm.Error("No input files: link a SIGPAC dataset to this job")

    geometries: list[BaseGeometry] = []
    properties: list[dict[str, Any]] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise Algorithm.Error(f"Input {path.name} is not JSON: {e}") from e

        recognised = isinstance(data, list) or (
            isinstance(data, dict)
            and data.get("type") in ({"FeatureCollection", "Feature"} | GEOMETRY_TYPES)
        )
        if not recognised:
            raise Algorithm.Error(
                f"Input {path.name} is neither GeoJSON nor SIGPAC records with a WKT geometry"
            )

        features = features_of(data)
        for feature in features:
            geometry = feature.get("geometry")
            if not geometry or geometry.get("type") not in GEOMETRY_TYPES:
                continue
            geometries.append(shape(geometry))
            properties.append(feature.get("properties") or {})
        logger.info(f"Read {len(features)} feature(s) from {path.name}")

    if not geometries:
        raise Algorithm.Error(
            "No SIGPAC parcel matches the given codes (the input has no polygons)"
        )

    geometry = unary_union(geometries)
    if geometry.is_empty or not geometry.is_valid:
        geometry = geometry.buffer(0)
    return Parcel(geometry=geometry, properties=properties)
