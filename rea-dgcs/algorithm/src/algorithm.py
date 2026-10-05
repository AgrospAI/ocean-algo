from pathlib import Path
from typing import Any
from collections import defaultdict

import csv
import io
import json
import logging
import requests

from ocean_runner import Algorithm
from oceanprotocol_job_details import EmptyInputParameters

SIGPAC_API_URL = "https://sigpac-hubcloud.es/ogcapi/collections/recintos/items"

logger = logging.getLogger(__name__)

type GeoJSONFeature = dict[str, Any]
type CropRecord = dict[str, Any]
type ParcelLogEntry = dict[str, Any]

# Palette used to colour polygons by crop code (stable assignment per code).
CROP_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
    "#808080",
]

# Path of the bundled SIEX product (crop) catalogue. To update just replace this
# file (same name/columns) with the latest version from:
#   https://www3.sede.fega.gob.es/bdcsixpor/descarga-csv/PRODUCTOS/csv
CROP_CSV_PATH = Path(__file__).resolve().parent / "cultivos.csv"


def load_crop_names(csv_path: Path | None = None) -> dict[int, str]:
    """Load the crop-code -> name mapping from the bundled SIEX catalogue CSV."""
    path = csv_path or CROP_CSV_PATH
    if not path.exists():
        logger.warning("Crop catalogue CSV not found at %s; using codes only.", path)
        return {}
    raw = path.read_bytes()
    decoded = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        logger.warning("Could not decode crop catalogue CSV at %s.", path)
        return {}
    mapping: dict[int, str] = {}
    for row in csv.DictReader(io.StringIO(decoded), delimiter=";"):
        code = (row.get("Código") or "").strip()
        name = (row.get("Cultivo") or "").strip()
        if code.isdigit() and name:
            mapping[int(code)] = name
    return mapping


# Mapping of crop codes to human-readable names (loaded from the bundled CSV).
CROP_NAMES: dict[int, str] = load_crop_names()


class ProcessingResult:
    """Structured result containing GeoJSON features, a parcel log and holding info."""

    def __init__(
        self,
        features: list[GeoJSONFeature],
        parcel_log: list[ParcelLogEntry],
        holding: dict[str, Any] | None = None,
    ):
        self.features = features
        self.parcel_log = parcel_log
        self.holding = holding or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "parcel_log": self.parcel_log,
            "holding": self.holding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingResult":
        return cls(
            features=data.get("features", []),
            parcel_log=data.get("parcel_log", []),
            holding=data.get("holding"),
        )


# The algorithm framework serializes results, so we pass a dict representation
type ResultsT = list[dict[str, Any]]

algorithm = Algorithm[EmptyInputParameters, dict[str, Any]].create(None)


def _parse_year(date_str: str | None) -> int | None:
    """Parse a 'dd/mm/YYYY' date and return the year (int) or None."""
    if not date_str:
        return None
    parts = str(date_str).split("/")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _date_key(date_str: str | None) -> tuple[int, int, int]:
    """Sortable key for a 'dd/mm/YYYY' date (year, month, day); None sorts last."""
    if not date_str:
        return (9999, 0, 0)
    parts = str(date_str).split("/")
    if len(parts) != 3:
        return (9999, 0, 0)
    try:
        return (int(parts[2]), int(parts[1]), int(parts[0]))
    except ValueError:
        return (9999, 0, 0)


def extract_crops(dgc: dict) -> list[CropRecord]:
    """Normalise the crop records of a single DGC entry."""
    crops: list[CropRecord] = []
    for c in dgc.get("cultivos", []):
        detalles = []
        for det in c.get("detalleHorticolasViveros", []) or []:
            detalles.append(
                {
                    "cultivo": det.get("cultivo"),
                    "superficie": det.get("superficie"),
                    "fechaInicio": det.get("fechaInicio"),
                    "fechaFin": det.get("fechaFin"),
                }
            )
        for det in c.get("detalleVariedadesFrutales", []) or []:
            detalles.append({"variedadFrutal": det.get("variedadFrutal")})
        crops.append(
            {
                "cultivo": c.get("cultivo"),
                "variedad": c.get("variedad"),
                "fechaInicio": c.get("fechaInicio"),
                "fechaFin": c.get("fechaFin"),
                "sistema": c.get("sistemaExplotacion"),
                "detalles": detalles,
            }
        )
    return crops


def _active_years(crop: CropRecord, max_year: int) -> list[int]:
    """Years a crop covers, up to the dataset max year (for open-ended crops)."""
    start = _parse_year(crop.get("fechaInicio"))
    if start is None:
        return []
    end = _parse_year(crop.get("fechaFin")) or max_year
    return list(range(start, max(end, start) + 1))


def _global_years(fields_crops: dict[tuple[str, ...], list[CropRecord]]) -> list[int]:
    """Sorted distinct years referenced by any crop date across the dataset."""
    years: set[int] = set()
    for crops in fields_crops.values():
        for crop in crops:
            for d in (crop.get("fechaInicio"), crop.get("fechaFin")):
                y = _parse_year(d)
                if y is not None:
                    years.add(y)
    return sorted(years)


def crops_by_year(
    fields_crops: dict[tuple[str, ...], list[CropRecord]], years: list[int]
) -> dict[tuple[str, ...], dict[int, list[CropRecord]]]:
    """Map field identifier -> {year: [active crop records]} for the given years."""
    if not years:
        return {}
    max_year = max(years)
    result: dict[tuple[str, ...], dict[int, list[CropRecord]]] = {}
    for ident, crops in fields_crops.items():
        by_year: dict[int, list[CropRecord]] = {}
        for year in years:
            active = [c for c in crops if year in _active_years(c, max_year)]
            active.sort(key=lambda c: _date_key(c.get("fechaInicio")))
            by_year[year] = active
        result[ident] = by_year
    return result


def fetch_parcel_geojson(parcel: dict[str, str]) -> tuple[GeoJSONFeature | None, ParcelLogEntry]:
    """Query the SIGPAC API for a single parcel's geometry and return a log entry."""
    filter_expr = (
        f"provincia={parcel['provincia']} AND "
        f"municipio={parcel['municipio']} AND "
        f"agregado={parcel['agregado']} AND "
        f"zona={parcel['zona']} AND "
        f"poligono={parcel['poligono']} AND "
        f"parcela={parcel['parcela']} AND "
        f"recinto={parcel['recinto']}"
    )
    params = {"f": "json", "limit": 1, "filter": filter_expr}

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
            feature = features[0]
            feature_id = feature.get("id")
            log_entry["success"] = True
            log_entry["feature_id"] = feature_id
            if "properties" not in feature:
                feature["properties"] = {}
            feature["properties"]["id"] = feature_id
            return feature, log_entry
        log_entry["error"] = "No features returned by API"
        logger.warning(f"Parcel {parcel} -> No features returned")
    except requests.RequestException as e:
        log_entry["error"] = str(e)
        logger.error(f"Error fetching parcel {parcel}: {e}")

    return None, log_entry


# SIGPAC property keys that are just the parcel identifiers (shown separately).
_SIGPAC_IDENT_KEYS = {
    "provincia", "municipio", "agregado", "zona", "poligono", "parcela", "recinto", "id",
}


@algorithm.run
def run(_) -> ResultsT:
    """Load DGC inputs, extract parcels, crop history and general info, query SIGPAC."""
    fields_crops: dict[tuple[str, ...], list[CropRecord]] = defaultdict(list)
    rea_general: dict[tuple[str, ...], dict[str, Any]] = {}
    parcels: dict[tuple[str, ...], dict[str, str]] = {}
    holding: dict[str, Any] = {}

    for did, file_path in algorithm.job_details.inputs():
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                dgc_data = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Could not read input file {file_path}: {e}")
            continue

        explotaciones = (
            dgc_data.get("resultado", {}).get("explotacionREA", {}).get("EXPLOTACION", [])
        )
        if not explotaciones:
            logger.warning(
                f"Input {file_path} does not contain DGCs "
                f"(codigoResultado={dgc_data.get('resultado', {}).get('codigoResultado')})."
            )
        for explotacion in explotaciones:
            if not holding:
                # Privacy: keep only what is needed to label the holdings map.
                gen_exp = dict(
                    explotacion.get("datosExplotacion", {}).get("datosGeneralesExplotacion") or {}
                )
                # Cross-link to another farm registry (identifies the holding).
                gen_exp.pop("codigoREGEPA", None)
                holding = {
                    "codigoREA": explotacion.get("codigoREA"),
                    "coauExplotacion": explotacion.get("coauExplotacion"),
                    "cacheSIGPAC": explotacion.get("cacheSIGPAC"),
                    "datosGeneralesExplotacion": gen_exp,
                }
            for dgc in explotacion.get("DGC", []):
                ident = (
                    dgc.get("provincia", ""),
                    dgc.get("municipio", ""),
                    dgc.get("agregado", ""),
                    dgc.get("zona", ""),
                    dgc.get("poligono", ""),
                    dgc.get("parcela", ""),
                    dgc.get("recinto", ""),
                )
                fields_crops[ident].extend(extract_crops(dgc))
                # General REA data for the parcel (beyond the crops).
                general = dict(dgc.get("datosGeneralesDGC", {}))
                # The geometry is already available via the SIGPAC feature; the
                # WKT text is huge and not shown in the tables/map, so drop it.
                general.pop("WKT", None)
                # The cadastral reference links the parcel to public ownership
                # records; the parcel is already identified by its SIGPAC codes
                # and geometry, so drop it for privacy.
                general.pop("referenciaCatastral", None)
                general["codigoDGC"] = dgc.get("codigoDGC")
                rea_general[ident] = general
                parcels[ident] = {
                    "provincia": ident[0],
                    "municipio": ident[1],
                    "agregado": ident[2],
                    "zona": ident[3],
                    "poligono": ident[4],
                    "parcela": ident[5],
                    "recinto": ident[6],
                }

    years = _global_years(fields_crops)
    by_year = crops_by_year(fields_crops, years)

    features: list[GeoJSONFeature] = []
    parcel_log: list[ParcelLogEntry] = []
    for ident, parcel in parcels.items():
        feature, log_entry = fetch_parcel_geojson(parcel)
        parcel_log.append(log_entry)
        if feature is not None:
            props = feature["properties"]
            # Preserve all SIGPAC attributes beyond the parcel identifiers.
            props["sigpac"] = {
                k: v
                for k, v in props.items()
                if k not in _SIGPAC_IDENT_KEYS and k != "sigpac" and k != "dn_pk"
            }
            for key, value in zip(
                ("provincia", "municipio", "agregado", "zona", "poligono", "parcela", "recinto"),
                ident,
            ):
                props[key] = value
            props["superficie"] = rea_general[ident].get("superficieDGC")
            props["rea"] = rea_general[ident]
            props["holding"] = holding
            # crop history: {year: [crop records]}
            props["crops_by_year"] = {str(y): by_year[ident][y] for y in years}
            features.append(feature)

    return [ProcessingResult(features=features, parcel_log=parcel_log, holding=holding).to_dict()]


def crop_label(code: Any) -> str:
    """Human label for a crop code, preferring names when available."""
    if code is None:
        return "N/D"
    try:
        c = int(code)
    except (TypeError, ValueError):
        return str(code)
    return CROP_NAMES.get(c, f"Cultivo {c}")


def generate_leaflet_map(
    features: list[GeoJSONFeature],
    years: list[int],
    crop_codes: list[int],
    holding: dict[str, Any] | None = None,
) -> str:
    """Generate an interactive Leaflet HTML map browsable by year and crop."""
    color_by_code: dict[int, str] = {}
    for i, code in enumerate(sorted(set(crop_codes))):
        color_by_code[code] = CROP_PALETTE[i % len(CROP_PALETTE)]

    payload = {
        "features": features,
        "years": [str(y) for y in years],
        "colors": {str(k): v for k, v in color_by_code.items()},
        "legend": {str(c): crop_label(c) for c in sorted(set(crop_codes))},
        "holding": holding or {},
    }

    return _LEAFLET_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))


@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    """Save GeoJSON features, a parcel log and an interactive Leaflet HTML map + tables."""
    processing_result = ProcessingResult.from_dict(result[0])

    crop_codes: set[int] = set()
    years: set[int] = set()
    for feature in processing_result.features:
        props = feature.get("properties", {})
        for y, crops in (props.get("crops_by_year") or {}).items():
            try:
                years.add(int(y))
            except (TypeError, ValueError):
                pass
            for c in crops:
                if c.get("cultivo") is not None:
                    try:
                        crop_codes.add(int(c["cultivo"]))
                    except (TypeError, ValueError):
                        pass

    # Save individual GeoJSON feature files.
    for feature in processing_result.features:
        feature_id = feature.get("id", "unknown")
        output_path = base / f"{feature_id}.geojson"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(feature, file, indent=2, ensure_ascii=False)

    # Save parcel processing log.
    if processing_result.parcel_log:
        log_path = base / "parcel_log.json"
        with open(log_path, "w", encoding="utf-8") as file:
            json.dump(processing_result.parcel_log, file, indent=2, ensure_ascii=False)
        logger.info(
            f"Parcel log saved with {len(processing_result.parcel_log)} entries to {log_path}"
        )

    # Generate and save interactive Leaflet HTML map with parcel tables.
    if processing_result.features:
        m = generate_leaflet_map(
            processing_result.features,
            sorted(years),
            sorted(crop_codes),
            processing_result.holding,
        )
        map_path = base / "parcel_map.html"
        with open(map_path, "w", encoding="utf-8") as file:
            file.write(m)
        logger.info(
            f"Interactive Leaflet map saved with {len(processing_result.features)} "
            f"features to {map_path}"
        )


_LEAFLET_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REA — Mapa y detalle de parcelas</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;font-family:system-ui,Arial,sans-serif}
  .bar{display:flex;align-items:center;gap:16px;padding:10px 16px;background:#f4f4f4;border-bottom:1px solid #ccc;flex-wrap:wrap}
  .bar select{font-size:15px;padding:4px 8px}
  .bar .title{font-weight:600}
  .bar button{font-size:14px;padding:6px 12px;cursor:pointer}
  a.yearlink{color:#1565c0;cursor:pointer;text-decoration:underline}
  .filters{display:flex;align-items:center;gap:14px;padding:8px 16px;background:#eef1f4;border-bottom:1px solid #ccc;flex-wrap:wrap;font-size:14px}
  .filters label{font-weight:600}
  .filters input,.filters select{padding:4px 6px;font-size:14px}
  .filters input[type=number]{width:80px}
  .filters .hint{color:#666;font-style:italic;font-weight:400}
  #count{font-weight:600}
  #mapwrap{position:relative;height:520px;border-bottom:1px solid #ccc}
  #map{height:100%;width:100%}
  #legend{position:absolute;right:12px;bottom:12px;z-index:1000;background:#fff;border:1px solid #ccc;border-radius:6px;padding:8px 12px;max-height:60%;overflow:auto;font-size:13px;box-shadow:0 1px 4px rgba(0,0,0,.2)}
  #legend h4{margin:0 0 6px}
  #legend .item{display:flex;align-items:center;gap:6px;margin-bottom:2px}
  #legend .sw{width:14px;height:14px;border:1px solid #333;display:inline-block;flex:0 0 auto}
  #tables{padding:20px 24px}
  .parcel{margin-bottom:32px}
  .parcel h3{margin:0 0 8px;border-bottom:2px solid #d0d0d0;padding-bottom:6px}
  table{border-collapse:collapse;width:100%;margin-bottom:12px;font-size:14px}
  caption{text-align:left;font-weight:600;padding:6px 0;color:#333}
  th,td{border:1px solid #d6d6d6;padding:6px 10px;text-align:left;vertical-align:top}
  th{background:#f4f4f4;font-weight:600}
  .info th{width:260px}
  #holding{background:#f9f9f9;border:1px solid #ddd;border-radius:6px;padding:12px 16px;margin-bottom:24px}
  #holding h3{margin:0 0 6px}
</style>
</head>
<body>
<div class="bar">
  <span class="title">Parcelas REA por campaña</span>
  <label>Año: <select id="yearSel"></select></label>
  <button id="resetBtn" type="button">Reset zoom</button>
</div>
<div id="mapwrap">
  <div id="map"></div>
  <div id="legend"><h4>Leyenda de cultivos</h4><div id="legendItems"></div></div>
</div>
<div class="filters">
  <span class="hint">Filtros:</span>
  <label>Provincia <select id="f-provincia"></select></label>
  <label>Municipio <select id="f-municipio"></select></label>
  <label>Polígono <select id="f-poligono"></select></label>
  <label>Parcela <select id="f-parcela"></select></label>
  <label>Cultivo <select id="cropFilter"><option value="">Todos</option></select></label>
  <label>Superficie (ha):
    <input id="minArea" type="number" min="0" step="0.01" placeholder="mín">
    –
    <input id="maxArea" type="number" min="0" step="0.01" placeholder="máx">
  </label>
  <button id="clearFilters" type="button">Limpiar filtros</button>
  <span id="count"></span>
</div>
<div id="holding"></div>
<div id="tables"></div>

<script>
const DATA = __DATA__;
const paletteYears = DATA.years;
const colors = DATA.colors;
const legend = DATA.legend;

function anchorIdFromProps(p){
  return [p.provincia, p.municipio, p.agregado, p.zona,
          p.poligono, p.parcela, p.recinto].join('-');
}

// ---- Map ----
const map = L.map('map', {scrollWheelZoom: true});

const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// Free satellite imagery (Esri World Imagery).
const sat = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
  }
);

L.control.layers({'Mapa': osm, 'Satélite': sat}, null, {position: 'topleft'}).addTo(map);

const featureCollection = {type:'FeatureCollection', features: DATA.features};
const geojson = L.geoJSON(featureCollection);
geojson.addTo(map);

const layers = [];
const layerByAnchor = {};
geojson.eachLayer(l => {
  layers.push(l);
  layerByAnchor[anchorIdFromProps(l.feature.properties)] = l;
});

function styleFor(feature, year){
  const props = feature.properties || {};
  const crops = (props.crops_by_year || {})[year] || [];
  if(crops.length === 0){
    return {color:'#888', weight:1, fillColor:'#cccccc', fillOpacity:0.4};
  }
  const code = Number(crops[0].cultivo);
  const col = colors[String(code)] || '#888';
  return {color:'#000', weight:1.5, fillColor:col, fillOpacity:0.6};
}

function cropName(code){
  return legend[String(code)] || ('Cultivo ' + code);
}

function popupFor(feature, year){
  const props = feature.properties || {};
  const id = parcelId(props);
  let html = '<div><b>Parcela:</b> <a href="#parcel-' + anchorIdFromProps(props) + '">' + id + '</a></div>';
  if(props.superficie != null) html += '<div><b>Superficie:</b> ' + props.superficie + ' ha</div>';
  const crops = (props.crops_by_year || {})[year] || [];
  if(crops.length === 0){
    html += '<div><i>Sin cultivo registrado este año (' + year + ')</i></div>';
  } else {
    html += '<div><b>Cultivos ' + year + ':</b></div>';
    crops.forEach(function(c){
      html += '<div>&bull; ' + cropName(c.cultivo) + ' (cód. ' + c.cultivo + ')' +
              (c.variedad ? ' · variedad ' + c.variedad : '') +
              (c.fechaInicio ? ' · ' + c.fechaInicio : '') +
              (c.fechaFin ? ' \u2192 ' + c.fechaFin : '') +
              (c.sistema ? ' · ' + c.sistema : '') + '</div>';
    });
  }
  return html;
}

function applyYear(year){
  layers.forEach(function(l){
    l.setStyle(styleFor(l.feature, year));
    l.bindPopup(popupFor(l.feature, year));
  });
}

const yearSel = document.getElementById('yearSel');
paletteYears.forEach(function(y){
  const o = document.createElement('option');
  o.value = y; o.textContent = y;
  yearSel.appendChild(o);
});
if(paletteYears.length){
  yearSel.value = paletteYears[0];
  applyYear(paletteYears[0]);
}
yearSel.addEventListener('change', function(){ applyYear(yearSel.value); });

// Populate the crop facet from the crops present (legend).
const cropFilter = document.getElementById('cropFilter');
Object.keys(legend).sort(function(a,b){ return (a-b); }).forEach(function(code){
  const o = document.createElement('option');
  o.value = code;
  o.textContent = legend[code] + ' (cód. ' + code + ')';
  cropFilter.appendChild(o);
});

const legendItems = document.getElementById('legendItems');
const codes = Object.keys(legend);
if(codes.length){
  codes.forEach(function(code){
    const div = document.createElement('div');
    div.className = 'item';
    div.innerHTML = '<span class="sw" style="background:' + (colors[code]||'#888') + '"></span> ' +
                    legend[code] + ' <small>(cód. ' + code + ')</small>';
    legendItems.appendChild(div);
  });
} else {
  legendItems.innerHTML = '<i>Sin datos de cultivo</i>';
}

const bounds = geojson.getBounds();
if(bounds.isValid()){ map.fitBounds(bounds); }

// ---- Helpers for tables ----
function parcelId(p){
  return [p.provincia, p.municipio, p.agregado, p.zona,
          p.poligono, p.parcela, p.recinto].join('/');
}

function labelFor(key){
  const map = {
    codigoREA:'Código REA', coauExplotacion:'Coau. explotación', cacheSIGPAC:'Caché SIGPAC',
    codigoDGC:'Código DGC', superficieDGC:'Superficie (ha)', fechaAlta:'Fecha de alta',
    referenciaCatastral:'Referencia catastral', origenCUE:'Origen CUE', indParcelaSIGPAG:'Ind. parcela SIGPAC',
    indicadorConcentracionParcelaria:'Concentración parcelaria', indicadorPastoComun:'Pasto común',
    indicadorDehesaComun:'Dehesa común', regimenTenencia:'Régimen tenencia',
    identificadorExtranjero:'Ident. extranjero', pequeniaSup:'Pequeña superficie',
    altitud:'Altitud', pendiente_media:'Pendiente media', dn_pk:'ID interno SIGPAC',
    punctosAgua:'Puntos de agua', SRID:'SRID'
  };
  return map[key] || key.replace(/_/g,' ').replace(/\b\w/g, function(m){ return m.toUpperCase(); });
}

function fmt(v){
  if(v === null || v === undefined || v === '') return '\u2014';
  if(typeof v === 'boolean') return v ? 'Sí' : 'No';
  if(typeof v === 'object') return JSON.stringify(v);
  return v;
}

function detailStr(detalles){
  if(!detalles || !detalles.length) return '\u2014';
  return detalles.map(function(d){
    if(d.cultivo != null){
      let s = cropName(d.cultivo) + ' (cód. ' + d.cultivo + ')';
      if(d.superficie !== undefined && d.superficie !== null) s += ' · ' + d.superficie + ' ha';
      if(d.fechaInicio) s += ' · ' + d.fechaInicio + (d.fechaFin ? ' \u2192 ' + d.fechaFin : '');
      return s;
    }
    if(d.variedadFrutal != null) return 'variedad frutal ' + d.variedadFrutal;
    return fmt(d);
  }).join('<br>');
}

// ---- Holding info ----
function renderHolding(){
  const h = DATA.holding || {};
  const el = document.getElementById('holding');
  if(!h.codigoREA){ el.style.display='none'; return; }
  let html = '<h3>Explotación ' + (h.codigoREA || '') + '</h3><div>';
  if(h.datosGeneralesExplotacion){
    html += '<div><b>Fecha registro:</b> ' + fmt(h.datosGeneralesExplotacion.fechaRegistro) + '</div>';
    html += '<div><b>SRID:</b> ' + fmt(h.datosGeneralesExplotacion.SRID) + '</div>';
  }
  html += '<div><b>Caché SIGPAC:</b> ' + fmt(h.cacheSIGPAC) + '</div>';
  html += '</div>';
  el.innerHTML = html;
}

// ---- Per-parcel tables ----
function renderTables(matched){
  const wrap = document.getElementById('tables');
  let out = '';
  const feats = matched
    ? DATA.features.filter(function(f){ return matched.has(anchorIdFromProps(f.properties)); })
    : DATA.features;
  feats.forEach(function(f){
    const p = f.properties || {};
    const id = parcelId(p);
    const anchor = anchorIdFromProps(p);
    out += '<div class="parcel" id="parcel-' + anchor + '"><h3>Parcela ' + id + '</h3>';

    // General info table (REA + SIGPAC beyond geometry).
    out += '<table class="info"><caption>Información general de la parcela</caption><tbody>';
    out += '<tr><th>Superficie (REA)</th><td>' + fmt(p.superficie) + ' ha</td></tr>';
    const rea = p.rea || {};
    Object.keys(rea).forEach(function(k){
      if(k === 'puntosAgua') return; // handled below if needed
      out += '<tr><th>' + labelFor(k) + '</th><td>' + fmt(rea[k]) + '</td></tr>';
    });
    const si = p.sigpac || {};
    Object.keys(si).forEach(function(k){
      out += '<tr><th>SIGPAC · ' + labelFor(k) + '</th><td>' + fmt(si[k]) + '</td></tr>';
    });
    out += '</tbody></table>';

    // Crops per year table.
    out += '<table class="crops"><caption>Cultivos por año</caption>';
    out += '<thead><tr><th>Año</th><th>Cultivo</th><th>Variedad</th><th>Inicio</th><th>Fin</th><th>Sistema</th><th>Detalles</th></tr></thead><tbody>';
    DATA.years.forEach(function(y){
      const crops = (p.crops_by_year || {})[y] || [];
      const ycell = '<td><a class="yearlink" data-anchor="' + anchor + '" data-year="' + y + '">' + y + '</a></td>';
      if(crops.length === 0){
        out += '<tr>' + ycell + '<td colspan="6"><i>Sin cultivo registrado este año</i></td></tr>';
      } else {
        crops.forEach(function(c){
          out += '<tr>' + ycell +
                 '<td>' + cropName(c.cultivo) + ' (cód. ' + c.cultivo + ')</td>' +
                 '<td>' + fmt(c.variedad) + '</td>' +
                 '<td>' + fmt(c.fechaInicio) + '</td>' +
                 '<td>' + fmt(c.fechaFin) + '</td>' +
                 '<td>' + fmt(c.sistema) + '</td>' +
                 '<td>' + detailStr(c.detalles) + '</td></tr>';
        });
      }
    });
    out += '</tbody></table></div>';
  });
  wrap.innerHTML = out;
}

renderHolding();
renderTables();

// Clicking a year in a parcel table jumps to the map, zooms to that parcel
// and switches the year selector to that year.
function gotoYear(anchor, year){
  if(yearSel){ yearSel.value = year; applyYear(year); }
  window.scrollTo({top:0, behavior:'smooth'}); // show year selector + reset zoom
  const layer = layerByAnchor[anchor];
  if(layer && layer.getBounds && layer.getBounds().isValid()){
    map.fitBounds(layer.getBounds(), {padding:[40,40], maxZoom:18});
    layer.openPopup(); // simulate a click so the parcel details are shown
  }
}
document.addEventListener('click', function(e){
  const el = e.target.closest ? e.target.closest('.yearlink') : null;
  if(el){ gotoYear(el.getAttribute('data-anchor'), el.getAttribute('data-year')); }
});

// Reset zoom to include the geometries of the parcels currently shown.
document.getElementById('resetBtn').addEventListener('click', function(){
  let b = null;
  layers.forEach(function(l){
    if(map.hasLayer(l) && l.getBounds && l.getBounds().isValid()){
      const lb = l.getBounds();
      b = b ? b.extend(lb) : lb;
    }
  });
  if(b){ map.fitBounds(b); }
});

// ---- Faceted filters (SIGPAC components, crop, surface range) ----
// The parcel set is decided only by these facets; the selected year only
// controls the crop colouring/popup and never hides or shows parcels.
const SIGPAC_FIELDS = ['provincia','municipio','poligono','parcela'];
const fieldSel = {};
SIGPAC_FIELDS.forEach(function(f){ fieldSel[f] = document.getElementById('f-' + f); });

function parentMatch(i){
  return function(f){
    for(let j=0;j<i;j++){
      const v = fieldSel[SIGPAC_FIELDS[j]].value;
      if(v && String(f.properties[SIGPAC_FIELDS[j]]) !== v) return false;
    }
    return true;
  };
}

function populateField(i){
  const field = SIGPAC_FIELDS[i];
  const sel = fieldSel[field];
  const pred = parentMatch(i);
  const values = new Set();
  DATA.features.forEach(function(f){ if(pred(f)) values.add(String(f.properties[field])); });
  const prev = sel.value;
  sel.innerHTML = '';
  const o0 = document.createElement('option'); o0.value=''; o0.textContent='(todos)'; sel.appendChild(o0);
  Array.from(values).sort(function(a,b){ return (parseInt(a,10)||0)-(parseInt(b,10)||0); }).forEach(function(v){
    const o=document.createElement('option'); o.value=v; o.textContent=v; sel.appendChild(o);
  });
  if(prev && values.has(prev)){ sel.value = prev; } else { sel.value=''; }
}

function fieldChanged(i){
  for(let j=i+1;j<SIGPAC_FIELDS.length;j++){ fieldSel[SIGPAC_FIELDS[j]].value=''; }
  for(let j=i+1;j<SIGPAC_FIELDS.length;j++){ populateField(j); }
  applyFilters();
}

function matchedAnchors(){
  const crop = document.getElementById('cropFilter').value;
  const smin = parseFloat(document.getElementById('minArea').value);
  const smax = parseFloat(document.getElementById('maxArea').value);
  const set = new Set();
  DATA.features.forEach(function(f){
    const p = f.properties || {};
    for(let i=0;i<SIGPAC_FIELDS.length;i++){
      const v = fieldSel[SIGPAC_FIELDS[i]].value;
      if(v && String(p[SIGPAC_FIELDS[i]]) !== v) return;
    }
    const s = p.superficie;
    if(s != null && s !== ''){
      if(!isNaN(smin) && s < smin) return;
      if(!isNaN(smax) && s > smax) return;
    }
    if(crop){
      const by = p.crops_by_year || {};
      const has = Object.keys(by).some(function(y){
        return (by[y] || []).some(function(c){ return String(c.cultivo) === crop; });
      });
      if(!has) return;
    }
    set.add(anchorIdFromProps(p));
  });
  return set;
}

function applyFilters(){
  const matched = matchedAnchors();
  layers.forEach(function(l){
    const a = anchorIdFromProps(l.feature.properties);
    if(matched.has(a)){
      if(!map.hasLayer(l)) map.addLayer(l);
    } else if(map.hasLayer(l)){
      map.removeLayer(l);
    }
  });
  renderTables(matched);
  document.getElementById('count').textContent =
    'Parcelas mostradas: ' + matched.size + ' / ' + DATA.features.length;
}

SIGPAC_FIELDS.forEach(function(f, i){
  fieldSel[f].addEventListener('change', function(){ fieldChanged(i); });
});
document.getElementById('cropFilter').addEventListener('change', applyFilters);
document.getElementById('minArea').addEventListener('input', applyFilters);
document.getElementById('maxArea').addEventListener('input', applyFilters);
document.getElementById('clearFilters').addEventListener('click', function(){
  SIGPAC_FIELDS.forEach(function(f){ fieldSel[f].value=''; });
  document.getElementById('cropFilter').value='';
  document.getElementById('minArea').value='';
  document.getElementById('maxArea').value='';
  for(let i=0;i<SIGPAC_FIELDS.length;i++){ populateField(i); }
  applyFilters();
});
for(let i=0;i<SIGPAC_FIELDS.length;i++){ populateField(i); }
applyFilters();
</script>
</body>
</html>
"""
