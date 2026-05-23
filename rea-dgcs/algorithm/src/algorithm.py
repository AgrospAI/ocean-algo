from pathlib import Path
from typing import Any

import folium
import json
import logging
import requests

from ocean_runner import Algorithm
from oceanprotocol_job_details import EmptyInputParameters

SIGPAC_API_URL = "https://sigpac-hubcloud.es/ogcapi/collections/recintos/items"

logger = logging.getLogger(__name__)

type GeoJSONFeature = dict[str, Any]
type ParcelLogEntry = dict[str, Any]

class ProcessingResult:
    """Structured result containing GeoJSON features and parcel processing log."""
    def __init__(
        self,
        features: list[GeoJSONFeature],
        parcel_log: list[ParcelLogEntry],
    ):
        self.features = features
        self.parcel_log = parcel_log

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "parcel_log": self.parcel_log,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingResult":
        return cls(
            features=data.get("features", []),
            parcel_log=data.get("parcel_log", []),
        )

# The algorithm framework serializes results, so we pass a dict representation
type ResultsT = list[dict[str, Any]]

algorithm = Algorithm[EmptyInputParameters, dict[str, Any]].create(None)


def extract_parcels(dgc_data: dict) -> list[dict[str, str]]:
    """Extract parcel details from the DGC input JSON structure.

    Navigates: resultado -> explotacionREA -> EXPLOTACION -> DGC
    and returns provincia, municipio, agregado, zona, poligono, parcela, recinto for each DGC item.
    """
    parcels = []
    explotaciones = (
        dgc_data
        .get("resultado", {})
        .get("explotacionREA", {})
        .get("EXPLOTACION", [])
    )
    for explotacion in explotaciones:
        for dgc in explotacion.get("DGC", []):
            parcel = {
                "provincia": dgc.get("provincia", ""),
                "municipio": dgc.get("municipio", ""),
                "agregado": dgc.get("agregado", ""),
                "zona": dgc.get("zona", ""),
                "poligono": dgc.get("poligono", ""),
                "parcela": dgc.get("parcela", ""),
                "recinto": dgc.get("recinto", ""),
            }
            parcels.append(parcel)
    return parcels


def fetch_parcel_geojson(parcel: dict[str, str]) -> tuple[GeoJSONFeature | None, ParcelLogEntry]:
    """Query the SIGPAC API for a single parcel and return the GeoJSON feature and a log entry."""
    filter_expr = (
        f"provincia={parcel['provincia']} AND "
        f"municipio={parcel['municipio']} AND "
        f"agregado={parcel['agregado']} AND "
        f"zona={parcel['zona']} AND "
        f"poligono={parcel['poligono']} AND "
        f"parcela={parcel['parcela']} AND "
        f"recinto={parcel['recinto']}"
    )

    params = {
        "f": "json",
        "limit": 1,
        "filter": filter_expr,
    }

    log_entry: ParcelLogEntry = {
        "provincia": parcel["provincia"],
        "municipio": parcel["municipio"],
        "agregado": parcel["agregado"],
        "zona": parcel["zona"],
        "poligono": parcel["poligono"],
        "parcela": parcel["parcela"],
        "recinto": parcel["recinto"],
        "success": False,
        "feature_id": None,
        "error": None,
    }

    try:
        response = requests.get(SIGPAC_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        features = data.get("features", [])
        if features:
            feature_id = features[0].get("id")
            log_entry["success"] = True
            log_entry["feature_id"] = feature_id
            logger.info(
                f"Parcel {parcel['provincia']}/{parcel['municipio']}/{parcel['agregado']}/"
                f"{parcel['zona']}/{parcel['poligono']}/{parcel['parcela']}/{parcel['recinto']}"
                f" -> feature_id={feature_id}"
            )
            # Attach DGC metadata to the feature properties for map tooltips
            feature = features[0]
            if "properties" not in feature:
                feature["properties"] = {}
            feature["properties"]["id"] = feature_id
            feature["properties"]["provincia"] = parcel["provincia"]
            feature["properties"]["municipio"] = parcel["municipio"]
            feature["properties"]["agregado"] = parcel["agregado"]
            feature["properties"]["zona"] = parcel["zona"]
            feature["properties"]["poligono"] = parcel["poligono"]
            feature["properties"]["parcela"] = parcel["parcela"]
            feature["properties"]["recinto"] = parcel["recinto"]
            return feature, log_entry
        else:
            log_entry["error"] = "No features returned by API"
            logger.warning(
                f"Parcel {parcel['provincia']}/{parcel['municipio']}/{parcel['agregado']}/"
                f"{parcel['zona']}/{parcel['poligono']}/{parcel['parcela']}/{parcel['recinto']}"
                f" -> No features returned"
            )
    except requests.RequestException as e:
        log_entry["error"] = str(e)
        logger.error(f"Error fetching parcel {parcel}: {e}")

    return None, log_entry


@algorithm.run
def run(_) -> ResultsT:
    """Load DGC inputs, extract parcels, query SIGPAC API, return structured result."""

    def process_input(file_path: Path) -> tuple[list[GeoJSONFeature], list[ParcelLogEntry]]:
        with open(file_path, "r", encoding="utf-8") as file:
            dgc_data = json.load(file)

        parcels = extract_parcels(dgc_data)
        features: list[GeoJSONFeature] = []
        log_entries: list[ParcelLogEntry] = []
        for parcel in parcels:
            feature, log_entry = fetch_parcel_geojson(parcel)
            log_entries.append(log_entry)
            if feature is not None:
                features.append(feature)
        return features, log_entries

    all_features: list[GeoJSONFeature] = []
    all_log_entries: list[ParcelLogEntry] = []
    for did, file_path in algorithm.job_details.inputs():
        features, log_entries = process_input(file_path)
        all_features.extend(features)
        all_log_entries.extend(log_entries)

    # Return structured result as a single dict in a list
    result = ProcessingResult(
        features=all_features,
        parcel_log=all_log_entries,
    )
    return [result.to_dict()]


def generate_map(features: list[GeoJSONFeature]) -> folium.Map:
    """Generate a Folium map displaying all GeoJSON features, auto-zoomed."""
    # Build a FeatureCollection from individual features
    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Create map centered on Spain (default fallback)
    m = folium.Map(location=[40.0, -4.0], zoom_start=6, tiles="CartoDB positron")

    # Add GeoJSON layer with styling and tooltips
    style_function = lambda feature: {
        "fillColor": "#ff7800",
        "color": "#000000",
        "weight": 2,
        "fillOpacity": 0.4,
    }

    highlight_function = lambda feature: {
        "fillColor": "#0000ff",
        "fillOpacity": 0.6,
    }

    # Use GeoJsonTooltip which reads field values from feature properties
    tooltip = folium.GeoJsonTooltip(
        fields=[
            "id",
            "provincia",
            "municipio",
            "agregado",
            "zona",
            "poligono",
            "parcela",
            "recinto",
        ],
        aliases=[
            "Identificador:",
            "Provincia:",
            "Municipio:",
            "Agregado:",
            "Zona:",
            "Polígono:",
            "Parcela:",
            "Recinto:",
        ],
        labels=True,
        sticky=True,
        max_width=300,
    )

    geojson_layer = folium.GeoJson(
        feature_collection,
        name="Parcels",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
    )
    geojson_layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Auto-zoom to fit all features
    if features:
        bounds = geojson_layer.get_bounds()
        m.fit_bounds(bounds)

    return m


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    """Save each GeoJSON feature, parcel log, and an interactive HTML map."""
    # Reconstruct the structured result from the serialized dict
    processing_result = ProcessingResult.from_dict(result[0])

    # Save individual GeoJSON feature files
    for feature in processing_result.features:
        feature_id = feature.get("id", "unknown")
        output_path = base / f"{feature_id}.geojson"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(feature, file, indent=2, ensure_ascii=False)

    # Save parcel processing log
    if processing_result.parcel_log:
        log_path = base / "parcel_log.json"
        with open(log_path, "w", encoding="utf-8") as file:
            json.dump(processing_result.parcel_log, file, indent=2, ensure_ascii=False)
        logger.info(
            f"Parcel log saved with {len(processing_result.parcel_log)} entries to {log_path}"
        )

    # Generate and save interactive HTML map
    if processing_result.features:
        m = generate_map(processing_result.features)
        map_path = base / "parcel_map.html"
        m.save(str(map_path))
        logger.info(f"Interactive map saved with {len(processing_result.features)} features to {map_path}")
