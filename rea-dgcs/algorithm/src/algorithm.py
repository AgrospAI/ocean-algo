from pathlib import Path
from typing import Any, Sequence

import json
import logging
import requests

from ocean_runner import Algorithm
from oceanprotocol_job_details import EmptyInputParameters

SIGPAC_API_URL = "https://sigpac-hubcloud.es/ogcapi/collections/recintos/items"

logger = logging.getLogger(__name__)

type GeoJSONFeature = dict[str, Any]
type ParcelLogEntry = dict[str, Any]
# Results can be either GeoJSON features or a special parcel log entry
type ResultItem = dict[str, Any]
type ResultsT = Sequence[ResultItem]

algorithm = Algorithm[EmptyInputParameters, ResultItem].create(None)


def extract_parcels(dgc_data: dict) -> list[dict[str, str]]:
    """Extract parcel details from the DGC input JSON structure.

    Navigates: resultado -> explotacionREA -> EXPLOTACION -> DGC
    and returns provincia, municipio, poligono, parcela, recinto for each DGC item.
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
                f"Parcel {parcel['provincia']}/{parcel['municipio']}/{parcel['poligono']}/"
                f"{parcel['parcela']}/{parcel['recinto']} -> feature_id={feature_id}"
            )
            return features[0], log_entry
        else:
            log_entry["error"] = "No features returned by API"
            logger.warning(
                f"Parcel {parcel['provincia']}/{parcel['municipio']}/{parcel['poligono']}/"
                f"{parcel['parcela']}/{parcel['recinto']} -> No features returned"
            )
    except requests.RequestException as e:
        log_entry["error"] = str(e)
        logger.error(f"Error fetching parcel {parcel}: {e}")

    return None, log_entry


@algorithm.run
def run(_) -> ResultsT:
    """Load DGC inputs, extract parcels, query SIGPAC API, return GeoJSON features and log."""

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

    # Build results: all geojson features + a single parcel log entry at the end
    results: list[ResultItem] = list(all_features)
    if all_log_entries:
        results.append({
            "_parcel_log": True,
            "entries": all_log_entries,
        })

    return results


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    """Save each GeoJSON feature as a separate file named by its 'id' field,
    and write the parcel processing log."""
    for item in result:
        # Skip the parcel log marker entry; handle it separately
        if item.get("_parcel_log"):
            continue

        feature_id = item.get("id", "unknown")
        output_path = base / f"{feature_id}.geojson"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(item, file, indent=2, ensure_ascii=False)

    # Extract and save parcel processing log
    for item in result:
        if item.get("_parcel_log"):
            log_path = base / "parcel_log.json"
            with open(log_path, "w", encoding="utf-8") as file:
                json.dump(item["entries"], file, indent=2, ensure_ascii=False)
            logger.info(f"Parcel log saved with {len(item['entries'])} entries to {log_path}")
