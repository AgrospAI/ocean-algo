#!/usr/bin/env python3

import base64
import json
import os
import re
import requests
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# --- Paths ------------------------------------------------------------------
INPUT_DIR = Path("/data/inputs")
OUTPUT_DIR = Path("/data/outputs")
RAW_DIR = Path("/tmp/soil-pdfs")

OUTPUT_MAP = OUTPUT_DIR / "soil-characteristics-map.html"

OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
OPENWEBUI_URL = "https://chat.agrospai.udl.cat"
CHAT_COMPLETIONS_URL = f"{OPENWEBUI_URL}/api/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL")

# In-memory cache: province_int → {NORMALIZED_NAME: mun_3digit}
_province_muni_cache: dict[int, dict[str, int]] = {}

# --- Colour scale for pH ----------------------------------------------------
def ph_color(ph: float | None) -> str:
    if ph is None:
        return "#6b7280"
    if ph < 5.5:
        return "#9b2226"
    if ph < 6.0:
        return "#ca6702"
    if ph <= 7.0:
        return "#2a9d8f"
    if ph <= 8.5:
        return "#457b9d"
    return "#6a040f"


def ph_label(ph: float | None) -> str:
    if ph is None:
        return "pH unavailable"
    if ph < 5.5:
        return f"pH {ph:.1f} – Strongly acidic"
    if ph < 6.0:
        return f"pH {ph:.1f} – Moderately acidic"
    if ph <= 7.0:
        return f"pH {ph:.1f} – Optimal/neutral"
    if ph <= 8.5:
        return f"pH {ph:.1f} – Slightly alkaline"
    return f"pH {ph:.1f} – Strongly alkaline"


# ============================================================================
# PHASE 1: PDF EXTRACTION VIA LLM
# ============================================================================

_EXTRACTION_PROMPT = """\
You are an expert at reading agricultural laboratory reports written in Catalan and Spanish.
Extract the following fields from the provided PDF page images and return a JSON object.

Fields:
- "document_type": string  (one of "soil", "water", "foliar", "other"). Classify by the report's
  content (header, section titles, sample description, units used) — NEVER by the file name.
  Use "soil" only for soil/land analyses ("Análisis de Suelo", "Anàlisi de Sòls", "Análisis de
  Tierras"). Use "water" for water analyses ("Análisis de Agua", units in mg/l). Use "foliar"
  for leaf/plant tissue analyses ("Análisis Foliar", "Foliars", values typically in ppm or %
  s.m.s. of plant matter). Use "other" if it is none of the above.
- "poligon": integer or null  (Polígon / Polígono / polygon number)
- "parcella_raw": string or null  (raw Parcella / Parcela text, e.g. "143-R:1")
- "parcela": integer or null  (numeric part of Parcella / Parcela)
- "recinto": integer  (Recinto / Recinte enclosure number; use 1 if not found)
- "muni_name": string  (Terme Municipal / Término Municipal / municipality name, uppercase)
- "raw_ine": integer or null  (5-digit INE municipality code if present, e.g. 25094)
- "cultiu": string  (Cultiu / Cultivo / crop type; empty string if absent)
- "pH": float or null
- "MO": float or null  (organic matter %; Matèria orgànica / Materia orgánica)
- "CE": float or null  (electrical conductivity dS/m; Conductivitat / Conductividad)
- "Caliza": float or null  (calcium carbonate %; Carbonat càlcic / Carbonato cálcico)
- "N_Nitrico": float or null  (nitric nitrogen mg/kg; Nitrogen nítric / Nitrógeno nítrico / N-NO₃)
- "Fosforo": float or null  (phosphorus mg/kg; Fòsfor / Fósforo)
- "Potasio": float or null  (potassium mg/kg; Potassi / Potasio)
- "Texture": string  (soil texture classification; empty string if absent)

Rules:
- Convert comma decimal separators to points for numeric values.
- Return null for any field not present in the document.
- Return ONLY a valid JSON object, no markdown fences, no extra explanation.

Strict unit/method rules for N / P / K (CRITICAL — wrong units cause downstream errors):
- N_Nitrico: ONLY return a value when the report explicitly labels it as nitric nitrogen
  ("Nitrogen nítric", "Nitrógeno nítrico", "N-NO₃") AND it is expressed in mg/kg (or a unit
  trivially convertible to mg/kg, e.g. mg/100g). Return null if the document only reports
  "Nitrógeno Total" / "N total" / "Nitrógeno Kjeldahl", or if the unit is ppm-of-total-N, %,
  or anything other than mg/kg of nitric nitrogen.
- Fosforo: ONLY return a value when extracted by the Olsen method OR expressed in mg/kg of
  soil. Return null if the extraction method is "extracto ácido" / "extract àcid" / Mehlich /
  Bray, or if the unit is %, % s.m.s., or ppm of plant matter (foliar).
- Potasio: ONLY return a value when expressed in mg/kg of soil (e.g. ammonium acetate
  extraction). Return null if the unit is "% s.m.s." / "%" or if it comes from a foliar
  "extracto ácido" extraction.
- When in doubt about the method or units for N / P / K, prefer returning null over guessing.

You may receive both rendered page images and a markdown transcription of the same PDF. Treat the images as ground truth; use the markdown only to resolve ambiguous digits.
"""


def pdf_to_images_base64(pdf_path: Path) -> list[str]:
    """Convert each PDF page to a base64-encoded JPEG data URL using pdftoppm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["pdftoppm", "-jpeg", "-r", "100", str(pdf_path),
             str(Path(tmpdir) / "page")],
            capture_output=True, timeout=60,
        )
        images = []
        for img_file in sorted(Path(tmpdir).glob("page-*.jpg")):
            with open(img_file, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            images.append(f"data:image/jpeg;base64,{data}")
    return images


_doc_converter = None


def _docling_converter():
    global _doc_converter
    if _doc_converter is None:
        from docling.document_converter import DocumentConverter
        _doc_converter = DocumentConverter()
    return _doc_converter


def pdf_to_markdown(pdf_path: Path) -> str | None:
    """Extract layout-aware markdown from a PDF via docling. None on failure."""
    try:
        result = _docling_converter().convert(str(pdf_path))
        md = result.document.export_to_markdown()
        return md.strip() or None
    except Exception as e:
        print(f"  [WARN] docling failed for {pdf_path.name}: {type(e).__name__}: {e}")
        return None


_warmed_up = False


def _warmup_llm() -> None:
    """Send a tiny ping so the first real PDF doesn't pay the full cold-start cost."""
    global _warmed_up
    if _warmed_up or not OPENWEBUI_API_KEY:
        return
    try:
        requests.post(
            CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "stream": False,
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.0,
                "chat_id": str(uuid.uuid4()),
                "id": str(uuid.uuid4()),
            },
            timeout=120,
        )
    except Exception as e:
        print(f"  [WARN] LLM warmup failed: {type(e).__name__}: {e}")
    finally:
        _warmed_up = True


def extract_with_llm(pdf_path: Path) -> dict | None:
    """Extract a soil analysis record from a PDF using the local LLM."""
    if not OPENWEBUI_API_KEY:
        print("  [ERROR] OPENWEBUI_API_KEY not set — cannot extract via LLM")
        return None

    _warmup_llm()

    images = pdf_to_images_base64(pdf_path)
    markdown = pdf_to_markdown(pdf_path)

    if not images or not markdown:
        print(f"  [SKIP] {pdf_path.name}: missing images or markdown for vision+markdown")
        return None

    headers = {
        "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
        "Content-Type": "application/json",
    }

    content: list[dict] = [{"type": "text", "text": _EXTRACTION_PROMPT}]
    content.append({
        "type": "text",
        "text": (
            "Below is the markdown extracted from the same PDF (may be incomplete "
            "or have OCR errors — use the page images as ground truth, but the "
            "markdown can disambiguate small numerics):\n\n" + markdown
        ),
    })
    for img_url in images:
        content.append({"type": "image_url", "image_url": {"url": img_url}})

    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        # OpenWebUI's /api/chat/completions requires chat_id/id since a
        # recent update; without them the router 400s with a cryptic
        # "'NoneType' object has no attribute 'startswith'".
        "chat_id": str(uuid.uuid4()),
        "id": str(uuid.uuid4()),
    }
    try:
        response = requests.post(
            CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=300
        )
        if not response.ok:
            body = response.text[:400] if response.text else "<empty>"
            print(f"  [WARN] vision+markdown failed for {pdf_path.name}: "
                  f"{response.status_code} {response.reason} — {body}")
            return None
        raw = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [WARN] vision+markdown failed for {pdf_path.name}: "
              f"{type(e).__name__}: {e}")
        return None

    # Parse JSON — handle optional markdown code fences
    extracted = None
    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1)
        else:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError:
            pass

    if extracted is None:
        print(f"  [WARN] Could not parse LLM JSON for {pdf_path.name}")
        return None

    doc_type = (extracted.get("document_type") or "").strip().lower()
    if doc_type != "soil":
        print(f"  [SKIP] {pdf_path.name}: document_type={doc_type or 'unknown'}")
        return None

    return {
        "pdf_name": pdf_path.name,
        "year": _infer_year(pdf_path),
        "poligon": extracted.get("poligon"),
        "parcella_raw": extracted.get("parcella_raw") or "",
        "parcela": extracted.get("parcela"),
        "recinto": extracted.get("recinto") or 1,
        "muni_name": (extracted.get("muni_name") or "UNKNOWN").upper(),
        "raw_ine": extracted.get("raw_ine"),
        "cultiu": extracted.get("cultiu") or "",
        "pH": extracted.get("pH"),
        "MO": extracted.get("MO"),
        "CE": extracted.get("CE"),
        "Caliza": extracted.get("Caliza"),
        "N_Nitrico": extracted.get("N_Nitrico"),
        "Fosforo": extracted.get("Fosforo"),
        "Potasio": extracted.get("Potasio"),
        "Texture": extracted.get("Texture") or "",
        "lat": None,
        "lng": None,
    }


def _infer_year(pdf_path: Path) -> int | None:
    """Try to infer the sample year from a year-shaped directory part or filename."""
    for part in pdf_path.parts:
        if re.match(r"^20\d{2}$", part):
            return int(part)
    m = re.search(r"20\d{2}", pdf_path.stem)
    return int(m.group(0)) if m else None


def _normalize_muni_name(name: str) -> str:
    """Uppercase and strip accents for fuzzy name matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name.upper())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _fetch_province_munis(prov: int) -> dict[str, int]:
    """
    Query Catastro ConsultaMunicipioCodigos for all municipalities in a province.
    Returns {NORMALIZED_NAME: mun_3digit}. Result is cached in _province_muni_cache.
    """
    if prov in _province_muni_cache:
        return _province_muni_cache[prov]

    url = (
        "https://ovc.catastro.meh.es/ovcservweb/OVCSWLocalizacionRC/"
        "OVCCallejeroCodigos.asmx/ConsultaMunicipioCodigos"
    )
    params = urllib.parse.urlencode({
        "CodigoProvincia": f"{prov:02d}",
        "CodigoMunicipio": "",
        "CodigoMunicipioIne": "",
    })
    req = urllib.request.Request(
        f"{url}?{params}",
        headers={"User-Agent": "SoilMappingResearch/1.0"},
    )
    mapping: dict[str, int] = {}
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        tree = ET.fromstring(raw)
        # Response: <consulta_municipio><municipiero><muni><nm>CORBINS</nm><locat><cmc>094</cmc>...
        ns = {"c": "http://www.catastro.meh.es/"}
        munis = list(tree.iter("muni")) or list(tree.iter("{http://www.catastro.meh.es/}muni"))
        for muni in munis:
            nm_el  = muni.find("nm")  or muni.find("{http://www.catastro.meh.es/}nm")
            cmc_el = muni.find(".//cmc") or muni.find(".//{http://www.catastro.meh.es/}cmc")
            if nm_el is not None and cmc_el is not None and nm_el.text and cmc_el.text:
                key = _normalize_muni_name(nm_el.text.strip())
                try:
                    mapping[key] = int(cmc_el.text.strip())
                except ValueError:
                    pass
        if mapping:
            print(f"    [CAT] Loaded {len(mapping)} municipalities for province {prov}")
            _province_muni_cache[prov] = mapping
        else:
            print(f"    [CAT] No municipalities parsed for province {prov} — raw response head: {raw[:200]}")
    except Exception as e:
        print(f"    [CAT] Could not fetch municipalities for province {prov}: {type(e).__name__}: {e}")
        # Do NOT cache on failure so the next call retries

    return mapping


def _lookup_muni_code_by_name(muni_name: str, prov: int = 25) -> tuple[int, int] | None:
    """
    Resolve a municipality name to (province, mun_3digit) via Catastro API.
    Tries exact normalized match, then partial match.
    """
    mapping = _fetch_province_munis(prov)
    if not mapping:
        return None

    normalized = _normalize_muni_name(muni_name)

    # Exact match
    if normalized in mapping:
        return prov, mapping[normalized]

    # Partial match: name contains key or key contains name
    for key, code in mapping.items():
        if normalized in key or key in normalized:
            return prov, code

    return None


def resolve_catastro_code(muni_name: str, raw_code: int | None) -> tuple[int, int] | None:
    """
    Return (province, municipality_3digit) for Catastro API lookup.
    Trusts the raw code extracted from the PDF (Catastro DGC code or SIGPAC-derived).
    Falls back to Catastro API name lookup for province 25 when code is absent.
    Returns None if the municipality cannot be resolved.
    """
    if raw_code is not None:
        prov = raw_code // 1000
        mun = raw_code % 1000
        if 1 <= prov <= 52:
            return prov, mun

    # Fallback: Catastro API name lookup (province 25 — Lleida region default)
    result = _lookup_muni_code_by_name(muni_name, prov=25)
    if result:
        return result

    print(f"  [WARN] Unknown municipality: '{muni_name}' (raw_code={raw_code})")
    return None



def extract_inputs() -> None:
    """Unzip every archive found under INPUT_DIR into RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Busca cualquier archivo en INPUT_DIR, sin depender de la extensión .zip
    for file_path in INPUT_DIR.rglob("*"):
        # Ignorar directorios
        if not file_path.is_file():
            continue
            
        # Comprobar si realmente es un archivo ZIP (lee la cabecera del archivo)
        if zipfile.is_zipfile(file_path):
            print(f"  [INFO] Extracting ZIP: {file_path.name}")
            with zipfile.ZipFile(file_path) as zf:
                zf.extractall(RAW_DIR)


def extract_all_pdfs() -> list[dict]:
    """Extract records from all PDFs found under RAW_DIR using the local LLM."""
    records = []
    pdfs = sorted(
        {p for p in (*RAW_DIR.rglob("*.pdf"), *RAW_DIR.rglob("*.PDF"))
         if not p.name.startswith("._")},
        key=lambda p: p.name,
    )

    print(f"\nProcessing {len(pdfs)} PDFs from {RAW_DIR}...")
    for pdf in pdfs:
        print(f"  {pdf.name}")
        rec = extract_with_llm(pdf)
        if rec:
            records.append(rec)
            if rec["poligon"] is not None:
                print(f"    → Polígon {rec['poligon']}, Parcella {rec['parcela']}, "
                      f"{rec['muni_name']}")

    geocodable = sum(1 for r in records if r["poligon"] is not None)
    print(f"\nExtracted {len(records)} total records "
          f"({geocodable} with Polígon/Parcella, "
          f"{len(records) - geocodable} without)")
    return records


# ============================================================================
# PHASE 2: GEOCODING VIA SPAIN CATASTRO API
# ============================================================================

# In-memory cache: (prov, mun, pol, par) → (lat, lng)
_coord_cache: dict = {}
# Municipality fallback cache: muni_name → (lat, lng)
_muni_cache: dict = {}


# Fruilar data region: Lleida lowlands (Segrià/Pla d'Urgell comarcas).
# Reject Catastro/Nominatim results outside this box to catch wrong municipality
# codes. Wide enough to cover Lleida, Huesca (lat 42.14), and Castellón (lat 39.97).
_LLEIDA_BBOX = (39.0, 43.5, -1.5, 1.5)   # (lat_min, lat_max, lng_min, lng_max)


def catastro_geocode(prov: int, mun: int, pol: int, par: int) -> tuple[float, float] | None:
    """
    Query the Spain Catastro (Land Registry) Consulta_CPMRC endpoint to get
    parcel centroid coordinates.

    RC (Referencia Catastral) format for rural/agricultural parcels:
      {prov2:02d}{mun3:03d}A{pol:03d}{par:05d}  (14 chars)
    Example: CORBINS (25094), Pol 6, Par 143 → "25094A006000143"

    Returns (lat, lng) or None if the parcel is not found or is outside the
    expected Lleida region (which would indicate a wrong municipality code).
    """
    rc = f"{prov:02d}{mun:03d}A{pol:03d}{par:05d}"
    url = (
        "https://ovc.catastro.meh.es/ovcservweb/ovcswlocalizacionrc/"
        "ovccoordenadas.asmx/Consulta_CPMRC"
    )
    params = urllib.parse.urlencode({
        "SRS": "EPSG:4326",
        "Provincia": "",
        "Municipio": "",
        "RC": rc,
    })
    req = urllib.request.Request(
        f"{url}?{params}",
        headers={"User-Agent": "SoilMappingResearch/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        tree = ET.fromstring(raw)
        ns = {"c": "http://www.catastro.meh.es/"}
        xcen = tree.find(".//c:xcen", ns)   # longitude
        ycen = tree.find(".//c:ycen", ns)   # latitude
        if xcen is not None and ycen is not None:
            lat = float(ycen.text)
            lng = float(xcen.text)
            lat_min, lat_max, lng_min, lng_max = _LLEIDA_BBOX
            if lat_min < lat < lat_max and lng_min < lng < lng_max:
                return lat, lng
            print(f"    [CAT] RC={rc} returned coords outside Lleida region "
                  f"(lat={lat:.4f}, lng={lng:.4f}) — likely wrong mun code")
    except Exception as e:
        print(f"    [CAT] Error querying RC={rc}: {type(e).__name__}: {e}")
    return None


def nominatim_geocode(muni_name: str) -> tuple[float, float] | None:
    """
    Fallback: geocode municipality name via Nominatim (OSM) using stdlib urllib.
    Returns (lat, lng) for the municipality centroid.
    """
    query = urllib.parse.urlencode({
        "q": f"{muni_name}, Lleida, Spain",
        "format": "json",
        "limit": "1",
    })
    url = f"https://nominatim.openstreetmap.org/search?{query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SoilMappingResearch/1.0 (educational)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
        if data:
            lat = float(data[0]["lat"])
            lng = float(data[0]["lon"])
            print(f"    [NOM] {muni_name} → lat={lat:.5f}, lng={lng:.5f}")
            return lat, lng
    except Exception as e:
        print(f"    [NOM] Error for '{muni_name}': {type(e).__name__}: {e}")
    return None


def geocode_record(rec: dict) -> tuple[float, float] | None:
    """
    Geocode a record using the Spain Catastro parcel API.
    Only exact parcel coordinates are accepted — no municipality centroid fallback.
    """
    if rec.get("poligon") is None or rec.get("parcela") is None:
        print(f"    [GEO] Skipping – no Polígon/Parcella for {rec['pdf_name']}")
        return None

    prov_mun = resolve_catastro_code(rec["muni_name"], rec["raw_ine"])
    if prov_mun is None:
        print(f"    [GEO] Cannot resolve municipality '{rec['muni_name']}'")
        return None

    prov, mun = prov_mun
    pol = rec["poligon"]
    par = rec["parcela"]

    cache_key = (prov, mun, pol, par)
    if cache_key in _coord_cache:
        return _coord_cache[cache_key]

    print(f"    [GEO] Catastro prov={prov} mun={mun:03d} pol={pol} par={par}")
    result = catastro_geocode(prov, mun, pol, par)
    time.sleep(0.5)

    if result:
        lat, lng = result
        lat_min, lat_max, lng_min, lng_max = _LLEIDA_BBOX
        if lat_min < lat < lat_max and lng_min < lng < lng_max:
            _coord_cache[cache_key] = result
            print(f"    [GEO] → lat={lat:.5f}, lng={lng:.5f}")
            return result
        print(f"    [GEO] Rejected out-of-region coords lat={lat:.4f}, lng={lng:.4f}")

    print(f"    [GEO] Could not geocode Polígon {pol} Parcella {par}")
    return None


def geocode_records(records: list[dict]) -> list[dict]:
    """Geocode all records via Catastro API."""
    for rec in records:
        coords = geocode_record(rec)
        if coords:
            rec["lat"], rec["lng"] = coords
    return records


# ============================================================================
# PHASE 3: MAP ROW CONVERSION
# ============================================================================

def records_to_map_rows(records: list[dict]) -> list[dict]:
    """Convert geocoded records directly to the row format expected by generate_map."""
    rows = []
    for i, rec in enumerate(records, 1):
        if rec["lat"] is None:
            continue
        rows.append({
            "id": f"Fruilar-{rec['year']}-{i:03d}",
            "year": rec["year"],
            "municipio": rec["muni_name"].title(),
            "cultivo": rec["cultiu"],
            "lat": rec["lat"],
            "lng": rec["lng"],
            "pH": rec["pH"],
            "MO": rec["MO"],
            "CE": rec["CE"],
            "Caliza": rec["Caliza"],
            "N_Nitrico": rec["N_Nitrico"],
            "Fosforo": rec["Fosforo"],
            "Potasio": rec["Potasio"],
            "source_file": rec["pdf_name"],
        })
    geocoded = len(rows)
    print(f"\n  {geocoded}/{len(records)} records geocoded successfully")
    return rows


def idw_grid(lats, lngs, values, grid_lat, grid_lng, power=2, min_dist=0.02):
    """IDW interpolation over a regular grid. min_dist (degrees) prevents singularities."""
    import numpy as np
    result = np.zeros(grid_lat.shape)
    weight_sum = np.zeros(grid_lat.shape)
    for lat, lng, val in zip(lats, lngs, values):
        d = np.sqrt((grid_lat - lat)**2 + (grid_lng - lng)**2 + min_dist**2)
        w = 1.0 / d**power
        result += w * val
        weight_sum += w
    return result / weight_sum


def raster_to_base64(grid_values, cmap, vmin, vmax, hull_mask=None):
    """Render interpolated grid as a transparent PNG with edge-fade alpha, return base64 data URI.

    hull_mask: optional boolean array (H, W); True = inside convex hull, False = outside.
    Pixels outside the hull are set fully transparent.
    """
    import numpy as np
    import io
    import base64
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter
    from PIL import Image

    smoothed = gaussian_filter(grid_values, sigma=3)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(smoothed)).copy()  # shape (H, W, 4), values in [0, 1]

    # Distance-based edge fade: each pixel fades to transparent over the outermost 15% of the grid
    H, W = smoothed.shape
    row_dist = np.minimum(np.arange(H), H - 1 - np.arange(H)).astype(np.float32)
    col_dist = np.minimum(np.arange(W), W - 1 - np.arange(W)).astype(np.float32)
    dist = np.minimum(row_dist[:, None], col_dist[None, :])
    fade_pixels = int(0.15 * min(H, W))
    alpha_fade = np.clip(dist / fade_pixels, 0, 1) if fade_pixels > 0 else np.ones((H, W), dtype=np.float32)

    # Apply global alpha (0.72) modulated by edge fade
    rgba[..., 3] = 0.72 * alpha_fade

    # Clip to convex hull with soft feathered edge (gaussian blur on the boolean mask)
    if hull_mask is not None:
        soft_hull = gaussian_filter(hull_mask.astype(np.float32), sigma=5)
        soft_hull = np.clip(soft_hull, 0, 1)
        rgba[..., 3] *= soft_hull

    # Encode to PNG using PIL for precise per-pixel alpha control
    img_array = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"


def _cluster_hull(lngs_seq, lats_seq, buffer: float = 0.05, eps: float = 0.5):
    """
    Return a shapely geometry that is the union of per-cluster buffered convex
    hulls.  Points within `eps` degrees of each other belong to the same
    cluster, so distant regions (e.g. Lleida vs Huesca vs Castellón) produce
    separate, non-connected hulls — preventing IDW from being shown across the
    empty space between them.
    """
    from scipy.spatial import KDTree
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    pts = list(zip(lngs_seq, lats_seq))
    if not pts:
        return None
    tree = KDTree(pts)
    visited = [False] * len(pts)
    hulls = []
    for i in range(len(pts)):
        if visited[i]:
            continue
        component: list[int] = []
        queue = [i]
        visited[i] = True
        while queue:
            j = queue.pop()
            component.append(j)
            for k in tree.query_ball_point(pts[j], eps):
                if not visited[k]:
                    visited[k] = True
                    queue.append(k)
        cluster_pts = [pts[idx] for idx in component]
        g = MultiPoint(cluster_pts)
        hulls.append(g.convex_hull.buffer(buffer) if len(cluster_pts) >= 3 else g.buffer(buffer))
    return unary_union(hulls)


def generate_map(all_rows: list[dict]) -> None:
    import numpy as np
    import matplotlib.colors as mcolors
    import folium
    import folium.raster_layers

    if not all_rows:
        print("[WARN] No data to plot")
        return

    lats = [r["lat"] for r in all_rows]
    lngs = [r["lng"] for r in all_rows]
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    # --- Per-parameter configuration ---
    PARAMS = {
        "ph":    ("pH",             lambda r: r.get("pH")),
        "mo":    ("OM — Organic Matter (%)",              lambda r: float(r["MO"])        if r.get("MO")        else None),
        "ce":    ("EC — Electrical Conductivity (dS/m)", lambda r: float(r["CE"])        if r.get("CE")        else None),
        "n_no3": ("N-NO₃ (mg/kg)", lambda r: float(r["N_Nitrico"]) if r.get("N_Nitrico") else None),
        "p":     ("P (mg/kg)",      lambda r: float(r["Fosforo"])   if r.get("Fosforo")   else None),
    }
    CMAPS = {
        "ph":    mcolors.LinearSegmentedColormap.from_list("ph",    ["#2166ac", "#74c476", "#238b45", "#fd8d3c", "#d73027"]),
        "mo":    mcolors.LinearSegmentedColormap.from_list("mo",    ["#d73027", "#fee08b", "#1a9850"]),
        "ce":    mcolors.LinearSegmentedColormap.from_list("ce",    ["#1a9850", "#fee08b", "#d73027"]),
        "n_no3": mcolors.LinearSegmentedColormap.from_list("n_no3", ["#fee08b", "#1a9850", "#d73027"]),
        "p":     mcolors.LinearSegmentedColormap.from_list("p",     ["#fee08b", "#1a9850", "#d73027"]),
    }
    # CSS gradient stops for JS legend rendering
    CMAP_STOPS = {
        "ph":    ["#2166ac", "#74c476", "#238b45", "#fd8d3c", "#d73027"],
        "mo":    ["#d73027", "#fee08b", "#1a9850"],
        "ce":    ["#1a9850", "#fee08b", "#d73027"],
        "n_no3": ["#fee08b", "#1a9850", "#d73027"],
        "p":     ["#fee08b", "#1a9850", "#d73027"],
    }

    # --- Build IDW rasters ---
    PAD = 0.05
    lat_min = min(lats) - PAD
    lat_max = max(lats) + PAD
    lng_min = min(lngs) - PAD
    lng_max = max(lngs) + PAD
    GRID_N = 120
    grid_lng_vals = np.linspace(lng_min, lng_max, GRID_N)
    grid_lat_vals = np.linspace(lat_max, lat_min, GRID_N)  # top→bottom for imshow
    grid_lng, grid_lat = np.meshgrid(grid_lng_vals, grid_lat_vals)

    # --- Per-subset (aggregate + per-year) hull masks ---
    # Each geographic cluster gets its own buffered convex hull so the IDW
    # raster is never shown in the empty space between distant regions.
    # For yearly subsets the hull is recomputed from just that year's points
    # so areas not sampled in a given year stay transparent (honest coverage).
    from shapely.geometry import Point
    from shapely.prepared import prep

    years_available = sorted({r["year"] for r in all_rows if r["year"] is not None})
    subsets: dict[str, list[dict]] = {"all": all_rows}
    for y in years_available:
        subsets[str(y)] = [r for r in all_rows if r["year"] == y]

    samples_per_subset = {k: len(rows) for k, rows in subsets.items()}

    def _build_hull_mask(rows_subset):
        if not rows_subset:
            return None
        sub_lngs = [r["lng"] for r in rows_subset]
        sub_lats = [r["lat"] for r in rows_subset]
        hull = _cluster_hull(sub_lngs, sub_lats, buffer=0.05, eps=0.5)
        if hull is None:
            return None
        ph = prep(hull)
        return np.array([
            ph.contains(Point(lon, lat))
            for lon, lat in zip(grid_lng.ravel(), grid_lat.ravel())
        ]).reshape(grid_lng.shape)

    subset_hull_masks = {k: _build_hull_mask(rows) for k, rows in subsets.items()}

    # raster_b64[param][subset_key] → data URI
    # raster_ranges[param] → (vmin, vmax)  — same scale across subsets so
    # cross-year comparison is visually honest.
    raster_b64: dict[str, dict[str, str]] = {}
    raster_ranges: dict[str, tuple[float, float]] = {}
    print("  Computing IDW rasters (aggregate + per-year)...")
    for key, (label, extractor) in PARAMS.items():
        all_pts = [(r["lat"], r["lng"], extractor(r))
                   for r in all_rows if extractor(r) is not None]
        if len(all_pts) < 3:
            print(f"    [SKIP] {label}: fewer than 3 valid samples in aggregate")
            continue
        _, _, all_vals = zip(*all_pts)
        all_vals_arr = np.array(all_vals, dtype=float)
        vmin = float(np.percentile(all_vals_arr, 2))
        vmax = float(np.percentile(all_vals_arr, 98))
        if vmin == vmax:
            vmax = vmin + 1  # prevent degenerate colormap
        raster_ranges[key] = (round(vmin, 2), round(vmax, 2))
        raster_b64[key] = {}

        for subset_key, rows_subset in subsets.items():
            pts = [(r["lat"], r["lng"], extractor(r))
                   for r in rows_subset if extractor(r) is not None]
            if len(pts) < 3:
                continue
            plats, plngs, pvals = zip(*pts)
            grid_vals = idw_grid(plats, plngs, pvals, grid_lat, grid_lng)
            raster_b64[key][subset_key] = raster_to_base64(
                grid_vals, CMAPS[key], vmin, vmax,
                hull_mask=subset_hull_masks.get(subset_key),
            )

        print(f"    {label}: [{vmin:.2f}, {vmax:.2f}]  "
              f"({len(raster_b64[key])}/{len(subsets)} subsets, "
              f"{len(all_pts)} samples aggregate)")

    # --- Aggregated statistics (computed in Python, injected as static values) ---
    def safe_float(val):
        try:
            return float(val) if val else None
        except (ValueError, TypeError):
            return None

    valid_ph = [r["pH"] for r in all_rows if r["pH"] is not None]
    total_n = len(all_rows)
    ph_n = len(valid_ph)

    def pct(count):
        return round(100 * count / ph_n, 1) if ph_n else 0

    ph_cats = {
        "Strongly acidic (pH &lt; 5.5)":         pct(sum(1 for p in valid_ph if p < 5.5)),
        "Moderately acidic (5.5–6.0)":             pct(sum(1 for p in valid_ph if 5.5 <= p < 6.0)),
        "Optimal / neutral (6.0–7.5)":             pct(sum(1 for p in valid_ph if 6.0 <= p <= 7.5)),
        "Slightly alkaline (7.5–8.5)":             pct(sum(1 for p in valid_ph if 7.5 < p <= 8.5)),
        "Strongly alkaline (pH &gt; 8.5)":         pct(sum(1 for p in valid_ph if p > 8.5)),
    }

    def param_stats_html(label, key, extractor):
        vals = [extractor(r) for r in all_rows if extractor(r) is not None]
        if not vals:
            return ""
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        return f'<tr><td>{label}</td><td style="text-align:right">{mean:.2f} &plusmn; {std:.2f}</td></tr>'

    stats_rows_html = "\n".join([
        param_stats_html("pH", "ph", lambda r: r.get("pH")),
        param_stats_html("OM — Organic Matter (%)", "mo", lambda r: safe_float(r.get("MO"))),
        param_stats_html("EC — Electrical Conductivity (dS/m)", "ce", lambda r: safe_float(r.get("CE"))),
        param_stats_html("N-NO₃ (mg/kg)", "n_no3", lambda r: safe_float(r.get("N_Nitrico"))),
        param_stats_html("P (mg/kg)", "p", lambda r: safe_float(r.get("Fosforo"))),
    ])

    # --- Build Folium map ---
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=10,
        tiles=None,
        zoom_control=False,
    )
    folium.TileLayer("CartoDB Positron", control=False).add_to(m)

    # WMS soil type layer (always visible, opacity controllable via slider)
    wms_layer = folium.WmsTileLayer(
        url="https://maps.isric.org/mapserv?map=/map/wrb.map",
        layers="MostProbable",
        name="Soil Types (WRB)",
        fmt="image/png",
        transparent=True,
        opacity=0.6,
        attr='<a href="https://www.isric.org/explore/soilgrids">ISRIC SoilGrids WRB</a>',
        show=True,
    )
    wms_layer.add_to(m)
    wms_var = wms_layer.get_name()

    # IDW raster overlays — one per (parameter, subset). JS toggles which one
    # is visible based on the parameter + year dropdowns. All are hidden at
    # build time; JS calls switchView("ph", "all") on init.
    overlay_vars: dict[str, dict[str, str]] = {}
    for key in raster_b64:
        overlay_vars[key] = {}
        for subset_key, img in raster_b64[key].items():
            overlay = folium.raster_layers.ImageOverlay(
                image=img,
                bounds=[[lat_min, lng_min], [lat_max, lng_max]],
                name=f"IDW: {PARAMS[key][0]} ({subset_key})",
                opacity=1.0,
                show=False,
                interactive=False,
                cross_origin=False,
            )
            overlay.add_to(m)
            overlay_vars[key][subset_key] = overlay.get_name()

    map_var = m.get_name()

    # --- Parameter + year selector + WMS slider panel (top-left) ---
    param_options_html = "\n".join(
        f'<option value="{k}">{PARAMS[k][0]}</option>'
        for k in raster_b64
    )
    year_options_html = '<option value="all" selected>All years</option>\n' + "\n".join(
        f'<option value="{y}">{y}</option>' for y in years_available
    )
    selector_html = f"""
    <div id="param-selector-panel" style="position:fixed; top:12px; left:12px; z-index:1000;
              background:rgba(255,255,255,0.94); border:1px solid #d6d3d1;
              padding:14px 16px; font-size:13px; line-height:1.55;
              border-radius:10px; box-shadow:0 12px 24px rgba(0,0,0,0.10);
              min-width:230px;">
      <div style="font-size:15px; font-weight:700; margin-bottom:10px;">Soil Characteristics</div>

      <label style="font-size:11px; color:#57534e; display:block; margin-bottom:4px;">Parameter</label>
      <select id="param-select" style="width:100%; padding:5px 8px; border:1px solid #d6d3d1;
              border-radius:6px; font-size:13px; background:#fff; cursor:pointer; margin-bottom:10px;">
        {param_options_html}
      </select>

      <label style="font-size:11px; color:#57534e; display:block; margin-bottom:4px;">Year</label>
      <select id="year-select" style="width:100%; padding:5px 8px; border:1px solid #d6d3d1;
              border-radius:6px; font-size:13px; background:#fff; cursor:pointer; margin-bottom:14px;">
        {year_options_html}
      </select>

      <label style="font-size:11px; color:#57534e; display:block; margin-bottom:4px;">
        WRB Soil Type opacity: <span id="wms-opacity-label">60%</span>
      </label>
      <input id="wms-opacity-slider" type="range" min="0" max="100" value="60"
             style="width:100%; accent-color:#2a9d8f; cursor:pointer; margin-bottom:14px;">

      <div id="legend-title" style="font-size:13px; font-weight:600; margin-bottom:6px;">pH</div>
      <div style="position:relative; margin-bottom:5px;">
        <div id="legend-gradient" style="width:100%; height:12px; border-radius:4px;
                border:1px solid #d6d3d1;"></div>
        <div id="legend-optimal-bar" style="position:absolute; top:0; height:12px;
                background:rgba(255,255,255,0.35); border:2px solid rgba(255,255,255,0.9);
                border-radius:3px; pointer-events:none; display:none;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:10px; color:#57534e; margin-bottom:6px;">
        <span id="legend-min"></span>
        <span id="legend-max"></span>
      </div>
      <div id="legend-optimal" style="font-size:10px; color:#166534; background:#dcfce7;
              border:1px solid #bbf7d0; border-radius:5px; padding:3px 7px; display:none;
              margin-bottom:5px;"></div>
      <div id="legend-warning" style="font-size:10px; color:#92400e; background:#fef3c7;
              border:1px solid #fde68a; border-radius:5px; padding:3px 7px; display:none;"></div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(selector_html))

    # --- WRB soil type legend (top-right, collapsed by default) ---
    wrb_legend_html = """
    <div id="wrb-legend" style="position:fixed; top:12px; right:12px; z-index:1000;
                background:rgba(255,255,255,0.94); border:1px solid #d6d3d1;
                border-radius:10px; box-shadow:0 12px 24px rgba(0,0,0,0.10);
                font-size:12px; overflow:hidden;">
      <div id="wrb-toggle" onclick="(function(){
              var c=document.getElementById('wrb-content');
              var t=document.getElementById('wrb-toggle');
              var open=c.style.display!=='none';
              c.style.display=open?'none':'block';
              t.querySelector('span.arrow').textContent=open?'▼':'▲';
            })()"
           style="padding:10px 14px; font-size:13px; font-weight:700; cursor:pointer;
                  display:flex; justify-content:space-between; align-items:center;
                  user-select:none;">
        <span>Soil Types (WRB)</span>
        <span class="arrow" style="font-size:10px; color:#78716c; margin-left:8px;">&#9660;</span>
      </div>
      <div id="wrb-content" style="display:none; padding:0 14px 12px 14px; max-width:230px;">
        <div style="color:#57534e; margin-bottom:9px; font-size:11px;">
          ISRIC SoilGrids &mdash; most probable class
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#FFFACD;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Calcisols</b> &mdash; calcareous, alkaline</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#708090;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Vertisols</b> &mdash; clayey, shrink-swell</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#DEB887;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Cambisols</b> &mdash; moderately developed</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#ADFF2F;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Fluvisols</b> &mdash; alluvial, river valleys</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#A0522D;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Leptosols</b> &mdash; shallow, stony</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#F4A460;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Regosols</b> &mdash; weakly developed</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#D2691E;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Kastanozems</b> &mdash; semi-arid grassland</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#FA8072;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Luvisols</b> &mdash; clay-enriched subsoil</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#FFB6C1;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Gypsisols</b> &mdash; gypsum-rich, arid areas</span>
        </div>
        <div style="display:flex;align-items:center;">
          <span style="display:inline-block;width:13px;height:13px;border-radius:2px;background:#FFDEAD;border:1px solid #bbb;margin-right:7px;flex-shrink:0;"></span>
          <span><b>Arenosols</b> &mdash; sandy soils</span>
        </div>
        <div style="margin-top:8px;font-size:10px;color:#78716c;">
          Colour swatches are approximate &mdash; see map tiles for exact classification
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(wrb_legend_html))

    # --- Aggregated stats panel (bottom-right, collapsible) ---
    ph_cats_rows = "\n".join(
        f'<tr><td style="padding:2px 6px 2px 0">{label}</td>'
        f'<td style="padding:2px 0; text-align:right">{pct_val}%</td></tr>'
        for label, pct_val in ph_cats.items()
    )
    samples_year_rows = "\n".join(
        f'<tr><td style="padding:2px 6px 2px 0">{y}</td>'
        f'<td style="padding:2px 0; text-align:right">{samples_per_subset.get(str(y), 0)}</td></tr>'
        for y in years_available
    )
    samples_year_section = f"""
        <div style="font-size:11px; font-weight:600; color:#57534e; margin-bottom:6px;">Samples per year</div>
        <table style="width:100%; border-collapse:collapse; font-size:11px; margin-bottom:10px;">
          {samples_year_rows}
        </table>
    """ if years_available else ""

    stats_panel_html = f"""
    <div id="stats-panel" style="position:fixed; bottom:36px; right:12px; z-index:1000;
              background:rgba(255,255,255,0.94); border:1px solid #d6d3d1;
              border-radius:10px; box-shadow:0 8px 18px rgba(0,0,0,0.09);
              font-size:12px; overflow:hidden; min-width:220px;">
      <div id="stats-header" onclick="document.getElementById('stats-body').style.display=
            document.getElementById('stats-body').style.display==='none'?'block':'none'"
           style="padding:10px 14px; font-weight:700; font-size:13px; cursor:pointer;
                  display:flex; justify-content:space-between; align-items:center;
                  border-bottom:1px solid #e5e7eb;">
        <span>Fleet Statistics</span>
        <span style="font-size:10px; color:#78716c;">{total_n} samples &#9660;</span>
      </div>
      <div id="stats-body" style="padding:10px 14px;">
        <div style="font-size:11px; font-weight:600; color:#57534e; margin-bottom:6px;">pH distribution</div>
        <table style="width:100%; border-collapse:collapse; font-size:11px; margin-bottom:10px;">
          {ph_cats_rows}
        </table>
        {samples_year_section}
        <div style="font-size:11px; font-weight:600; color:#57534e; margin-bottom:6px;">Parameter means &plusmn; std</div>
        <table style="width:100%; border-collapse:collapse; font-size:11px;">
          <tr style="color:#78716c; font-size:10px;"><th style="text-align:left">Parameter</th><th style="text-align:right">Value</th></tr>
          {stats_rows_html}
        </table>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_panel_html))

    # --- Unified JavaScript ---
    overlay_vars_js = json.dumps(overlay_vars)            # {param: {subset: var_name}}
    samples_per_subset_js = json.dumps(samples_per_subset)  # {subset: int}
    param_colors_js = "{" + ", ".join(
        f'"{k}": {CMAP_STOPS[k]}' for k in raster_b64
    ) + "}"
    param_ranges_js = "{" + ", ".join(
        f'"{k}": [{raster_ranges[k][0]}, {raster_ranges[k][1]}]' for k in raster_b64
    ) + "}"
    param_labels_js = "{" + ", ".join(
        f'"{k}": "{PARAMS[k][0]}"' for k in raster_b64
    ) + "}"

    spacing_css = """
    <style>
      .leaflet-bottom.leaflet-left { bottom: 12px !important; left: 12px !important; }
    </style>
    """
    m.get_root().html.add_child(folium.Element(spacing_css))

    js_html = f"""
    <script>
    (function() {{
      var _check = setInterval(function() {{
        if (window.{map_var}) {{
          clearInterval(_check);
          var map = window.{map_var};

          // Zoom control at bottom-left
          L.control.zoom({{position: 'bottomleft'}}).addTo(map);

          // Nested: param → subset → Leaflet layer
          var overlayVarNames = {overlay_vars_js};
          var overlayLayers = {{}};
          Object.keys(overlayVarNames).forEach(function(p) {{
            overlayLayers[p] = {{}};
            Object.keys(overlayVarNames[p]).forEach(function(s) {{
              overlayLayers[p][s] = window[overlayVarNames[p][s]];
            }});
          }});

          var samplesPerSubset = {samples_per_subset_js};
          var wmsLayer = window["{wms_var}"];

          var paramColors = {param_colors_js};
          var paramRanges = {param_ranges_js};
          var paramLabels = {param_labels_js};

          var LOW_SAMPLE_THRESHOLD = 15;

          // Optimal range metadata per parameter.
          var paramOptimal = {{
            'ph':    {{ optimalRange: [6.0, 7.5], direction: null,      note: 'Optimal: 6.0 – 7.5' }},
            'mo':    {{ optimalRange: null,        direction: 'higher',  note: 'Higher → better (organic matter)' }},
            'ce':    {{ optimalRange: [0, 2],      direction: null,      note: 'Optimal: < 2 dS/m (low salinity)' }},
            'n_no3': {{ optimalRange: [10, 40],    direction: null,      note: 'Optimal: 10 – 40 mg/kg' }},
            'p':     {{ optimalRange: [15, 50],    direction: null,      note: 'Optimal: 15 – 50 mg/kg' }},
          }};

          function updateLegend(param, year, hasRaster) {{
            var colors = paramColors[param];
            var range  = paramRanges[param] || [0, 1];
            var vmin = range[0], vmax = range[1];
            var span = vmax - vmin || 1;

            document.getElementById('legend-gradient').style.background =
              'linear-gradient(to right, ' + colors.join(',') + ')';
            document.getElementById('legend-min').textContent = vmin;
            document.getElementById('legend-max').textContent = vmax;

            var yearLabel = (year === 'all') ? 'All years' : ('Year ' + year);
            var n = samplesPerSubset[year] || 0;
            document.getElementById('legend-title').textContent =
              paramLabels[param] + ' — ' + yearLabel + ' (n=' + n + ')';

            // Optimal annotation
            var opt = paramOptimal[param];
            var optEl  = document.getElementById('legend-optimal');
            var barEl  = document.getElementById('legend-optimal-bar');
            var warnEl = document.getElementById('legend-warning');

            if (opt && opt.note) {{
              optEl.textContent = '✓ ' + opt.note;
              optEl.style.display = 'block';
            }} else {{
              optEl.style.display = 'none';
            }}

            if (opt && opt.optimalRange) {{
              var lo = opt.optimalRange[0];
              var hi = opt.optimalRange[1];
              var leftPct  = Math.max(0, Math.min(100, (lo - vmin) / span * 100));
              var rightPct = Math.max(0, Math.min(100, (hi - vmin) / span * 100));
              if (rightPct > leftPct) {{
                barEl.style.left  = leftPct + '%';
                barEl.style.width = (rightPct - leftPct) + '%';
                barEl.style.display = 'block';
              }} else {{
                barEl.style.display = 'none';
              }}
            }} else if (opt && opt.direction === 'higher') {{
              barEl.style.left  = '60%';
              barEl.style.width = '40%';
              barEl.style.display = 'block';
            }} else {{
              barEl.style.display = 'none';
            }}

            // Warning for low-sample yearly subsets, or missing raster
            if (!hasRaster) {{
              warnEl.textContent = 'No data for ' + yearLabel + ' — fewer than 3 samples for this parameter';
              warnEl.style.display = 'block';
            }} else if (year !== 'all' && n > 0 && n < LOW_SAMPLE_THRESHOLD) {{
              warnEl.textContent = 'n=' + n + ' samples — limited spatial coverage, interpret with care';
              warnEl.style.display = 'block';
            }} else {{
              warnEl.style.display = 'none';
            }}
          }}

          function switchView(param, year) {{
            // Hide every overlay first
            Object.keys(overlayLayers).forEach(function(p) {{
              Object.keys(overlayLayers[p]).forEach(function(s) {{
                if (overlayLayers[p][s]) map.removeLayer(overlayLayers[p][s]);
              }});
            }});
            // Show selected (param, year) if it exists
            var layer = overlayLayers[param] && overlayLayers[param][year];
            if (layer) {{
              map.addLayer(layer);
              updateLegend(param, year, true);
            }} else {{
              updateLegend(param, year, false);
            }}
          }}

          var paramSelect = document.getElementById('param-select');
          var yearSelect  = document.getElementById('year-select');

          paramSelect.addEventListener('change', function() {{
            switchView(paramSelect.value, yearSelect.value);
          }});
          yearSelect.addEventListener('change', function() {{
            switchView(paramSelect.value, yearSelect.value);
          }});

          document.getElementById('wms-opacity-slider').addEventListener('input', function() {{
            var val = this.value / 100;
            document.getElementById('wms-opacity-label').textContent = this.value + '%';
            if (wmsLayer) wmsLayer.setOpacity(val);
          }});

          // Initialise
          switchView('ph', 'all');
        }}
      }}, 50);
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_html))

    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_MAP))
    print(f"\nMap saved: {OUTPUT_MAP}")
    print(f"  {total_n} samples, {len(raster_b64)} IDW rasters generated")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Soil Data Extraction & Map Generation")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_inputs()

    # Phase 1: Extract PDF data
    records = extract_all_pdfs()

    if not records:
        print("[ERROR] No records extracted. Aborting.")
        sys.exit(1)

    # Phase 2: Geocode via Spain Catastro API
    print("\nGeocoding via Spain Catastro API...")
    records = geocode_records(records)

    # Dump per-record extraction for offline auditing / cross-validation
    dump_path = OUTPUT_DIR / "records.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nRecords dump: {dump_path} ({len(records)} records)")

    # Phase 3: Generate map directly from geocoded records (no CSV written)
    all_rows = records_to_map_rows(records)
    generate_map(all_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
