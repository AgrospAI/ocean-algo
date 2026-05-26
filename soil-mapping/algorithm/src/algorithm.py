#!/usr/bin/env python3

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import httpx
from dotenv import get_key
from functools import lru_cache
from ocean_runner import Algorithm, Config

# --- Paths ------------------------------------------------------------------
INPUT_DIR = Path("/data/inputs")
OUTPUT_DIR = Path("/data/outputs")
RAW_DIR = Path("/tmp/soil-pdfs")

OUTPUT_MAP = OUTPUT_DIR / "soil-characteristics-map.html"

OPENWEBUI_URL = "https://chat.agrospai.udl.cat"
CHAT_COMPLETIONS_URL = f"{OPENWEBUI_URL}/api/chat/completions"

LLM_CONCURRENCY = 4
GEO_CONCURRENCY = 4

algorithm = Algorithm.create(Config())


@lru_cache(maxsize=None)
def _load_env(key: str) -> str:
    path = algorithm.job_details.paths.algorithm
    source = "file" if path.exists() else "environ"
    value = get_key(str(path), key) if path.exists() else os.getenv(key)
    assert value, (
        f"{key} not populated (source={source}, path={path}); "
        f"got value={value!r}"
    )
    return value

# In-memory cache: province_int → {NORMALIZED_NAME: mun_3digit}
_province_muni_cache: dict[int, dict[str, int]] = {}
_province_muni_lock = asyncio.Lock()

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
- "report_date": string or null  (analysis or sampling date as ISO YYYY-MM-DD; extract from
  headers like "Data anàlisi", "Fecha de análisis", "Fecha de muestreo", "Data de mostratge",
  or the report header date. Return null if no date is visible.)

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
# Threading lock (not asyncio): converter is built lazily inside asyncio.to_thread,
# so concurrent worker threads can race on init without this.
_doc_converter_init_lock = threading.Lock()


def _docling_converter():
    # Lab-report PDFs are born-digital: trust the native text layer and skip
    # OCR (the dominant cost — ~15 s/PDF) and table-structure extraction
    # (~40 s/PDF). The markdown is only used as a digit-disambiguation hint
    # for the vision LLM; we don't need structured tables.
    global _doc_converter
    if _doc_converter is None:
        with _doc_converter_init_lock:
            if _doc_converter is None:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
                pipeline_options = PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                )
                _doc_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
    return _doc_converter


def pdf_to_markdown(pdf_path: Path) -> str | None:
    """Extract layout-aware markdown from a PDF via docling. None on failure."""
    try:
        result = _docling_converter().convert(str(pdf_path))
        md = result.document.export_to_markdown()
        return md.strip() or None
    except Exception as e:
        algorithm.logger.warning(
            f"docling failed for {pdf_path.name}: {type(e).__name__}: {e}"
        )
        return None


async def _warmup_llm(client: httpx.AsyncClient) -> None:
    """Send a tiny ping so the first real PDF doesn't pay the full cold-start cost."""
    try:
        await client.post(
            CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {_load_env('OPENWEBUI_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": _load_env("LLM_MODEL"),
                "stream": False,
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.0,
                "chat_id": str(uuid.uuid4()),
                "id": str(uuid.uuid4()),
                "reasoning_effort": "low",
                "chat_template_kwargs": {"enable_thinking": False},
                "thinking": {"type": "disabled"},
            },
            timeout=120,
        )
    except Exception as e:
        algorithm.logger.warning(f"LLM warmup failed: {type(e).__name__}: {e}")


async def extract_with_llm(client: httpx.AsyncClient, pdf_path: Path) -> dict | None:
    """Extract a soil analysis record from a PDF using the local LLM."""
    # PDF rasterisation and docling markdown are CPU/subprocess-bound — run in
    # threads so the event loop can keep dispatching other LLM tasks meanwhile.
    images, markdown = await asyncio.gather(
        asyncio.to_thread(pdf_to_images_base64, pdf_path),
        asyncio.to_thread(pdf_to_markdown, pdf_path),
    )

    if not images or not markdown:
        algorithm.logger.info(
            f"[SKIP] {pdf_path.name}: missing images or markdown for vision+markdown"
        )
        return None

    headers = {
        "Authorization": f"Bearer {_load_env('OPENWEBUI_API_KEY')}",
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
        "model": _load_env("LLM_MODEL"),
        "stream": False,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "chat_id": str(uuid.uuid4()),
        "id": str(uuid.uuid4()),
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }

    raw = None
    try:
        response = await client.post(
            CHAT_COMPLETIONS_URL, headers=headers, json=payload,
        )
        if response.status_code >= 400:
            body = response.text[:400] if response.text else "<empty>"
            algorithm.logger.warning(
                f"vision+markdown failed for {pdf_path.name}: "
                f"{response.status_code} {response.reason_phrase} — {body}"
            )
            return None
        try:
            data = response.json()
        except Exception:
            data = None
        if not isinstance(data, dict):
            algorithm.logger.warning(
                f"vision+markdown failed for {pdf_path.name}: "
                f"non-JSON or null body — status={response.status_code} "
                f"body={response.text[:600] if response.text else '<empty>'}"
            )
            return None
        choices = data.get("choices") or []
        first = choices[0] if choices else None
        message = first.get("message") if isinstance(first, dict) else None
        raw = message.get("content") if isinstance(message, dict) else None
        if not raw:
            algorithm.logger.warning(
                f"vision+markdown failed for {pdf_path.name}: "
                f"unexpected/empty response — {str(data)[:600]}"
            )
            return None
    except Exception as e:
        algorithm.logger.warning(
            f"vision+markdown failed for {pdf_path.name}: {type(e).__name__}: {e}"
        )
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
        algorithm.logger.warning(f"Could not parse LLM JSON for {pdf_path.name}")
        return None

    doc_type = (extracted.get("document_type") or "").strip().lower()
    if doc_type != "soil":
        algorithm.logger.info(
            f"[SKIP] {pdf_path.name}: document_type={doc_type or 'unknown'}"
        )
        return None

    return {
        "pdf_name": pdf_path.name,
        "year": _infer_year(pdf_path, extracted),
        "report_date": extracted.get("report_date"),
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


def _infer_year(pdf_path: Path, extracted: dict | None = None) -> int | None:
    """Infer the sample year from the LLM-extracted date, the path, or a compact filename date."""
    if extracted:
        rd = (extracted.get("report_date") or "").strip()
        m = re.match(r"^(20\d{2})", rd)
        if m:
            return int(m.group(1))

    for part in pdf_path.parts:
        if re.match(r"^20\d{2}$", part):
            return int(part)

    m = re.search(r"20\d{2}", pdf_path.stem)
    if m:
        return int(m.group(0))

    m = re.search(
        r"(?<!\d)(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)",
        pdf_path.stem,
    )
    if m:
        return 2000 + int(m.group(1))

    return None


def _normalize_muni_name(name: str) -> str:
    """Uppercase and strip accents for fuzzy name matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name.upper())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


async def _fetch_province_munis(
    client: httpx.AsyncClient, prov: int
) -> dict[str, int]:
    """
    Query Catastro ConsultaMunicipioCodigos for all municipalities in a province.
    Returns {NORMALIZED_NAME: mun_3digit}. Result is cached in _province_muni_cache.
    """
    if prov in _province_muni_cache:
        return _province_muni_cache[prov]

    async with _province_muni_lock:
        if prov in _province_muni_cache:
            return _province_muni_cache[prov]

        url = (
            "https://ovc.catastro.meh.es/ovcservweb/OVCSWLocalizacionRC/"
            "OVCCallejeroCodigos.asmx/ConsultaMunicipioCodigos"
        )
        params = {
            "CodigoProvincia": f"{prov:02d}",
            "CodigoMunicipio": "",
            "CodigoMunicipioIne": "",
        }
        mapping: dict[str, int] = {}
        try:
            resp = await client.get(
                url, params=params,
                headers={"User-Agent": "SoilMappingResearch/1.0"},
            )
            raw = resp.content
            tree = ET.fromstring(raw)
            munis = list(tree.iter("muni")) or list(
                tree.iter("{http://www.catastro.meh.es/}muni")
            )
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
                algorithm.logger.info(
                    f"[CAT] Loaded {len(mapping)} municipalities for province {prov}"
                )
                _province_muni_cache[prov] = mapping
            else:
                algorithm.logger.warning(
                    f"[CAT] No municipalities parsed for province {prov} — "
                    f"raw response head: {raw[:200]}"
                )
        except Exception as e:
            algorithm.logger.warning(
                f"[CAT] Could not fetch municipalities for province {prov}: "
                f"{type(e).__name__}: {e}"
            )

        return mapping


async def _lookup_muni_code_by_name(
    client: httpx.AsyncClient, muni_name: str, prov: int = 25
) -> tuple[int, int] | None:
    """
    Resolve a municipality name to (province, mun_3digit) via Catastro API.
    Tries exact normalized match, then partial match.
    """
    mapping = await _fetch_province_munis(client, prov)
    if not mapping:
        return None

    normalized = _normalize_muni_name(muni_name)

    if normalized in mapping:
        return prov, mapping[normalized]

    for key, code in mapping.items():
        if normalized in key or key in normalized:
            return prov, code

    return None


async def resolve_catastro_code(
    client: httpx.AsyncClient, muni_name: str, raw_code: int | None
) -> tuple[int, int] | None:
    """
    Return (province, municipality_3digit) for Catastro API lookup.
    Trusts the raw code extracted from the PDF (Catastro DGC code or SIGPAC-derived).
    Falls back to Catastro API name lookup for province 25 when code is absent.
    """
    if raw_code is not None:
        prov = raw_code // 1000
        mun = raw_code % 1000
        if 1 <= prov <= 52:
            return prov, mun

    result = await _lookup_muni_code_by_name(client, muni_name, prov=25)
    if result:
        return result

    algorithm.logger.warning(
        f"Unknown municipality: '{muni_name}' (raw_code={raw_code})"
    )
    return None


def extract_inputs() -> None:
    """Unzip every archive found under INPUT_DIR into RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in INPUT_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        if zipfile.is_zipfile(file_path):
            algorithm.logger.info(f"[INFO] Extracting ZIP: {file_path.name}")
            with zipfile.ZipFile(file_path) as zf:
                zf.extractall(RAW_DIR)


async def extract_all_pdfs(client: httpx.AsyncClient) -> list[dict]:
    """Extract records from all PDFs found under RAW_DIR using the local LLM."""
    pdfs = sorted(
        {p for p in (*RAW_DIR.rglob("*.pdf"), *RAW_DIR.rglob("*.PDF"))
         if not p.name.startswith("._")},
        key=lambda p: p.name,
    )

    algorithm.logger.info(
        f"Processing {len(pdfs)} PDFs from {RAW_DIR} "
        f"(concurrency={LLM_CONCURRENCY})..."
    )

    sem = asyncio.Semaphore(LLM_CONCURRENCY)

    async def _bounded(pdf: Path) -> dict | None:
        async with sem:
            algorithm.logger.info(f"  → start {pdf.name}")
            rec = await extract_with_llm(client, pdf)
            if rec and rec["poligon"] is not None:
                algorithm.logger.info(
                    f"    ← {pdf.name}: Polígon {rec['poligon']}, "
                    f"Parcella {rec['parcela']}, {rec['muni_name']}"
                )
            return rec

    results = await asyncio.gather(*[_bounded(p) for p in pdfs])
    records = [r for r in results if r]
    # Preserve deterministic order for downstream debug dumps.
    records.sort(key=lambda r: r["pdf_name"])

    geocodable = sum(1 for r in records if r["poligon"] is not None)
    algorithm.logger.info(
        f"Extracted {len(records)} total records "
        f"({geocodable} with Polígon/Parcella, "
        f"{len(records) - geocodable} without)"
    )
    return records


# ============================================================================
# PHASE 2: GEOCODING VIA SPAIN CATASTRO API
# ============================================================================

_coord_cache: dict = {}
_coord_cache_lock = asyncio.Lock()
_muni_cache: dict = {}


# Fruilar data region: Lleida lowlands (Segrià/Pla d'Urgell comarcas).
_LLEIDA_BBOX = (39.0, 43.5, -1.5, 1.5)   # (lat_min, lat_max, lng_min, lng_max)


async def catastro_geocode(
    client: httpx.AsyncClient, prov: int, mun: int, pol: int, par: int
) -> tuple[float, float] | None:
    """Query the Spain Catastro Consulta_CPMRC endpoint for parcel centroid."""
    rc = f"{prov:02d}{mun:03d}A{pol:03d}{par:05d}"
    url = (
        "https://ovc.catastro.meh.es/ovcservweb/ovcswlocalizacionrc/"
        "ovccoordenadas.asmx/Consulta_CPMRC"
    )
    params = {
        "SRS": "EPSG:4326",
        "Provincia": "",
        "Municipio": "",
        "RC": rc,
    }
    try:
        resp = await client.get(
            url, params=params,
            headers={"User-Agent": "SoilMappingResearch/1.0"},
        )
        raw = resp.content
        tree = ET.fromstring(raw)
        ns = {"c": "http://www.catastro.meh.es/"}
        xcen = tree.find(".//c:xcen", ns)
        ycen = tree.find(".//c:ycen", ns)
        if xcen is not None and ycen is not None:
            lat = float(ycen.text)
            lng = float(xcen.text)
            lat_min, lat_max, lng_min, lng_max = _LLEIDA_BBOX
            if lat_min < lat < lat_max and lng_min < lng < lng_max:
                return lat, lng
            algorithm.logger.warning(
                f"[CAT] RC={rc} returned coords outside Lleida region "
                f"(lat={lat:.4f}, lng={lng:.4f}) — likely wrong mun code"
            )
    except Exception as e:
        algorithm.logger.warning(
            f"[CAT] Error querying RC={rc}: {type(e).__name__}: {e}"
        )
    return None


async def nominatim_geocode(
    client: httpx.AsyncClient, muni_name: str
) -> tuple[float, float] | None:
    """Fallback: geocode municipality name via Nominatim (OSM)."""
    params = {
        "q": f"{muni_name}, Lleida, Spain",
        "format": "json",
        "limit": "1",
    }
    url = "https://nominatim.openstreetmap.org/search"
    try:
        resp = await client.get(
            url, params=params,
            headers={"User-Agent": "SoilMappingResearch/1.0 (educational)"},
        )
        data = resp.json()
        if data:
            lat = float(data[0]["lat"])
            lng = float(data[0]["lon"])
            algorithm.logger.info(
                f"[NOM] {muni_name} → lat={lat:.5f}, lng={lng:.5f}"
            )
            return lat, lng
    except Exception as e:
        algorithm.logger.warning(
            f"[NOM] Error for '{muni_name}': {type(e).__name__}: {e}"
        )
    return None


async def geocode_record(
    client: httpx.AsyncClient, rec: dict
) -> tuple[float, float] | None:
    """Geocode a record using the Spain Catastro parcel API."""
    if rec.get("poligon") is None or rec.get("parcela") is None:
        algorithm.logger.info(
            f"[GEO] Skipping – no Polígon/Parcella for {rec['pdf_name']}"
        )
        return None

    prov_mun = await resolve_catastro_code(client, rec["muni_name"], rec["raw_ine"])
    if prov_mun is None:
        algorithm.logger.warning(
            f"[GEO] Cannot resolve municipality '{rec['muni_name']}'"
        )
        return None

    prov, mun = prov_mun
    pol = rec["poligon"]
    par = rec["parcela"]

    cache_key = (prov, mun, pol, par)
    if cache_key in _coord_cache:
        return _coord_cache[cache_key]

    algorithm.logger.info(
        f"[GEO] Catastro prov={prov} mun={mun:03d} pol={pol} par={par}"
    )
    result = await catastro_geocode(client, prov, mun, pol, par)

    if result:
        lat, lng = result
        lat_min, lat_max, lng_min, lng_max = _LLEIDA_BBOX
        if lat_min < lat < lat_max and lng_min < lng < lng_max:
            async with _coord_cache_lock:
                _coord_cache[cache_key] = result
            algorithm.logger.info(f"[GEO] → lat={lat:.5f}, lng={lng:.5f}")
            return result
        algorithm.logger.warning(
            f"[GEO] Rejected out-of-region coords lat={lat:.4f}, lng={lng:.4f}"
        )

    algorithm.logger.warning(
        f"[GEO] Could not geocode Polígon {pol} Parcella {par}"
    )
    return None


async def geocode_records(
    client: httpx.AsyncClient, records: list[dict]
) -> list[dict]:
    """Geocode all records via Catastro API in parallel with a small semaphore."""
    sem = asyncio.Semaphore(GEO_CONCURRENCY)

    async def _bounded(rec: dict) -> None:
        async with sem:
            coords = await geocode_record(client, rec)
            if coords:
                rec["lat"], rec["lng"] = coords

    await asyncio.gather(*[_bounded(r) for r in records])
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
    algorithm.logger.info(
        f"  {geocoded}/{len(records)} records geocoded successfully"
    )
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
    """Render interpolated grid as a transparent PNG with edge-fade alpha.

    Returns ``(data_uri, value_grid)`` where ``value_grid`` is a nested list of the
    smoothed interpolated values (the exact values the rendered colours represent),
    with ``None`` wherever the raster is effectively invisible (outside the hull or
    fully faded). This drives the hover tooltip so the read-out matches the colour.
    """
    import numpy as np
    import io
    import base64 as b64mod
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter
    from PIL import Image

    smoothed = gaussian_filter(grid_values, sigma=1.2)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(smoothed)).copy()

    H, W = smoothed.shape
    row_dist = np.minimum(np.arange(H), H - 1 - np.arange(H)).astype(np.float32)
    col_dist = np.minimum(np.arange(W), W - 1 - np.arange(W)).astype(np.float32)
    dist = np.minimum(row_dist[:, None], col_dist[None, :])
    fade_pixels = int(0.08 * min(H, W))
    alpha_fade = np.clip(dist / fade_pixels, 0, 1) if fade_pixels > 0 else np.ones((H, W), dtype=np.float32)

    rgba[..., 3] = 0.85 * alpha_fade

    if hull_mask is not None:
        soft_hull = gaussian_filter(hull_mask.astype(np.float32), sigma=2)
        soft_hull = np.clip(soft_hull, 0, 1)
        rgba[..., 3] *= soft_hull

    # Expose the displayed value where the raster is visible; None elsewhere so the
    # tooltip vanishes off the painted area. Threshold matches near-zero alpha.
    visible = rgba[..., 3] >= 0.05
    rounded = np.round(smoothed, 2)
    value_grid = [
        [float(v) if vis else None for v, vis in zip(row_vals, row_vis)]
        for row_vals, row_vis in zip(rounded.tolist(), visible.tolist())
    ]

    img_array = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    data = b64mod.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{data}", value_grid


def _cluster_hull(lngs_seq, lats_seq, buffer: float = 0.025, eps: float = 0.5):
    """Union of per-cluster buffered convex hulls (DBSCAN-like over coords)."""
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


WRB_LEGEND = [
    ("Acrisols",     "#f7991d", "acidic, weathered"),
    ("Albeluvisols", "#9b9d57", "bleached"),
    ("Alisols",      "#faf7c0", "acidic, Al-rich"),
    ("Andosols",     "#ed3a33", "volcanic"),
    ("Arenosols",    "#f7d8ac", "sandy"),
    ("Calcisols",    "#ffee00", "calcareous"),
    ("Cambisols",    "#fecd67", "moderate"),
    ("Chernozems",   "#e2c837", "fertile steppe"),
    ("Cryosols",     "#756a92", "permafrost"),
    ("Durisols",     "#efe6bf", "silica-cemented"),
    ("Ferralsols",   "#f6872d", "tropical, weathered"),
    ("Fluvisols",    "#01b0ef", "alluvial"),
    ("Gleysols",     "#9291b9", "waterlogged"),
    ("Gypsisols",    "#fbf6a5", "gypsum-rich"),
    ("Histosols",    "#8b898a", "peat"),
    ("Kastanozems",  "#c99580", "grassland"),
    ("Leptosols",    "#d5d6d8", "shallow"),
    ("Lixisols",     "#f9bdbf", "tropical clay"),
    ("Luvisols",     "#f48385", "clay-rich"),
    ("Nitisols",     "#f7a082", "tropical clay"),
    ("Phaeozems",    "#ba6850", "dark, fertile"),
    ("Planosols",    "#f59354", "perched"),
    ("Plinthosols",  "#6f0e41", "iron-rich"),
    ("Podzols",      "#0daf63", "acidic forest"),
    ("Regosols",     "#ffe2ae", "weak"),
    ("Solonchaks",   "#ed3994", "saline"),
    ("Solonetz",     "#f4cde2", "alkaline, Na"),
    ("Stagnosols",   "#40c1eb", "perched, wet"),
    ("Umbrisols",    "#618f82", "dark, acidic"),
    ("Vertisols",    "#9e567c", "shrink-swell"),
]


def generate_map(all_rows: list[dict]) -> None:
    import numpy as np
    import matplotlib.colors as mcolors
    import folium
    import folium.raster_layers

    if not all_rows:
        algorithm.logger.warning("No data to plot")
        return

    lats = [r["lat"] for r in all_rows]
    lngs = [r["lng"] for r in all_rows]
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    PARAMS = {
        "ph":    ("pH",             lambda r: r.get("pH")),
        "mo":    ("OM — Organic Matter (%)",              lambda r: float(r["MO"])        if r.get("MO")        else None),
        "ce":    ("EC — Electrical Conductivity (dS/m)", lambda r: float(r["CE"])        if r.get("CE")        else None),
        "n_no3": ("N-NO₃ (mg/kg)", lambda r: float(r["N_Nitrico"]) if r.get("N_Nitrico") else None),
        "p":     ("P (mg/kg)",      lambda r: float(r["Fosforo"])   if r.get("Fosforo")   else None),
    }
    CMAP_STOPS = {
        "ph":    ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
        "mo":    ["#ffffe5", "#78c679", "#004529"],
        "ce":    ["#f7fbff", "#6baed6", "#08306b"],
        "n_no3": ["#fff5eb", "#fd8d3c", "#7f2704"],
        "p":     ["#fff5eb", "#fd8d3c", "#7f2704"],
    }
    CMAPS = {
        key: mcolors.LinearSegmentedColormap.from_list(key, stops)
        for key, stops in CMAP_STOPS.items()
    }

    PAD = 0.05
    lat_min = min(lats) - PAD
    lat_max = max(lats) + PAD
    lng_min = min(lngs) - PAD
    lng_max = max(lngs) + PAD
    GRID_N = 120
    grid_lng_vals = np.linspace(lng_min, lng_max, GRID_N)
    grid_lat_vals = np.linspace(lat_max, lat_min, GRID_N)
    grid_lng, grid_lat = np.meshgrid(grid_lng_vals, grid_lat_vals)

    from shapely.geometry import Point
    from shapely.prepared import prep

    MIN_SAMPLES_FOR_YEAR = 3
    years_available = sorted({r["year"] for r in all_rows if r["year"] is not None})
    subsets: dict[str, list[dict]] = {"all": all_rows}
    for y in years_available:
        subsets[str(y)] = [r for r in all_rows if r["year"] == y]

    years_available = [y for y in years_available
                       if len(subsets[str(y)]) >= MIN_SAMPLES_FOR_YEAR]
    for y_str in list(subsets):
        if y_str != "all" and int(y_str) not in years_available:
            del subsets[y_str]

    samples_per_subset = {k: len(rows) for k, rows in subsets.items()}

    def _build_hull_mask(rows_subset):
        if not rows_subset:
            return None
        sub_lngs = [r["lng"] for r in rows_subset]
        sub_lats = [r["lat"] for r in rows_subset]
        hull = _cluster_hull(sub_lngs, sub_lats, buffer=0.025, eps=0.5)
        if hull is None:
            return None
        ph = prep(hull)
        return np.array([
            ph.contains(Point(lon, lat))
            for lon, lat in zip(grid_lng.ravel(), grid_lat.ravel())
        ]).reshape(grid_lng.shape)

    subset_hull_masks = {k: _build_hull_mask(rows) for k, rows in subsets.items()}

    raster_b64: dict[str, dict[str, str]] = {}
    raster_grids: dict[str, dict[str, list]] = {}
    raster_ranges: dict[str, tuple[float, float]] = {}
    algorithm.logger.info("Computing IDW rasters (aggregate + per-year)...")
    for key, (label, extractor) in PARAMS.items():
        all_pts = [(r["lat"], r["lng"], extractor(r))
                   for r in all_rows if extractor(r) is not None]
        if len(all_pts) < 3:
            algorithm.logger.info(
                f"  [SKIP] {label}: fewer than 3 valid samples in aggregate"
            )
            continue
        _, _, all_vals = zip(*all_pts)
        all_vals_arr = np.array(all_vals, dtype=float)
        vmin = float(np.percentile(all_vals_arr, 2))
        vmax = float(np.percentile(all_vals_arr, 98))
        if vmin == vmax:
            vmax = vmin + 1
        raster_ranges[key] = (round(vmin, 2), round(vmax, 2))
        raster_b64[key] = {}
        raster_grids[key] = {}

        for subset_key, rows_subset in subsets.items():
            pts = [(r["lat"], r["lng"], extractor(r))
                   for r in rows_subset if extractor(r) is not None]
            if len(pts) < 3:
                continue
            plats, plngs, pvals = zip(*pts)
            grid_vals = idw_grid(plats, plngs, pvals, grid_lat, grid_lng)
            img_uri, value_grid = raster_to_base64(
                grid_vals, CMAPS[key], vmin, vmax,
                hull_mask=subset_hull_masks.get(subset_key),
            )
            raster_b64[key][subset_key] = img_uri
            raster_grids[key][subset_key] = value_grid

        algorithm.logger.info(
            f"  {label}: [{vmin:.2f}, {vmax:.2f}]  "
            f"({len(raster_b64[key])}/{len(subsets)} subsets, "
            f"{len(all_pts)} samples aggregate)"
        )

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
        return (
            f'<tr>'
            f'<td style="padding-right:8px">{label}</td>'
            f'<td style="text-align:right; white-space:nowrap; '
            f'font-variant-numeric:tabular-nums">'
            f'{mean:.2f} &plusmn; {std:.2f}</td>'
            f'</tr>'
        )

    stats_rows_html = "\n".join([
        param_stats_html("pH", "ph", lambda r: r.get("pH")),
        param_stats_html("OM — Organic Matter (%)", "mo", lambda r: safe_float(r.get("MO"))),
        param_stats_html("EC — Electrical Conductivity (dS/m)", "ce", lambda r: safe_float(r.get("CE"))),
        param_stats_html("N-NO₃ (mg/kg)", "n_no3", lambda r: safe_float(r.get("N_Nitrico"))),
        param_stats_html("P (mg/kg)", "p", lambda r: safe_float(r.get("Fosforo"))),
    ])

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=10,
        tiles=None,
        zoom_control=False,
    )
    folium.TileLayer("CartoDB Positron", control=False).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}"
        ),
        name="Hillshade",
        attr='Tiles &copy; <a href="https://www.esri.com">Esri</a>',
        opacity=0.4,
        control=False,
    ).add_to(m)

    wms_layer = folium.WmsTileLayer(
        url="https://maps.isric.org/mapserv?map=/map/wrb.map",
        layers="MostProbable",
        name="Soil Types (WRB)",
        fmt="image/png",
        transparent=True,
        opacity=0.35,
        attr='<a href="https://www.isric.org/explore/soilgrids">ISRIC SoilGrids WRB</a>',
        show=True,
    )
    wms_layer.add_to(m)
    wms_var = wms_layer.get_name()

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
      <div style="font-size:15px; font-weight:700; margin-bottom:10px;">Soil Conditions Dashboard</div>

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

      <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
        <label for="wms-opacity-slider" style="font-size:11px; color:#57534e;">WRB opacity</label>
        <span id="wms-opacity-label" style="font-size:11px; color:#57534e;
                font-variant-numeric:tabular-nums;">35%</span>
      </div>
      <input id="wms-opacity-slider" type="range" min="0" max="100" value="35"
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

    wrb_rows_html = "\n".join(
        f'<div title="{name} — {hint}" '
        f'style="display:flex;align-items:center;margin-bottom:3px;'
        f'break-inside:avoid;page-break-inside:avoid;">'
        f'<span class="wrb-swatch" style="display:inline-block;width:11px;'
        f'height:11px;border-radius:2px;background:{hex_};'
        f'border:1px solid rgba(0,0,0,0.15);margin-right:6px;'
        f'flex-shrink:0;"></span>'
        f'<span style="white-space:nowrap;font-size:11px;"><b>{name}</b></span>'
        f'</div>'
        for (name, hex_, hint) in WRB_LEGEND
    )
    wrb_body_html = f"""
        <div style="color:#57534e; margin-bottom:9px; font-size:11px;">
          ISRIC SoilGrids &mdash; most probable class
        </div>
        <div style="column-count:2; column-gap:14px;">
          {wrb_rows_html}
        </div>
        """

    wrb_legend_html = f"""
    <style>
      .wrb-swatch {{
        filter: saturate(0.35) brightness(1.05);
        opacity: var(--wrb-swatch-opacity, 0.35);
      }}
    </style>
    <div id="wrb-legend" style="position:fixed; top:12px; right:12px; z-index:1000;
                background:rgba(255,255,255,0.94); border:1px solid #d6d3d1;
                border-radius:10px; box-shadow:0 12px 24px rgba(0,0,0,0.10);
                font-size:12px; overflow:hidden;">
      <div id="wrb-toggle" onclick="(function(){{
              var c=document.getElementById('wrb-content');
              var t=document.getElementById('wrb-toggle');
              var open=c.style.display!=='none';
              c.style.display=open?'none':'block';
              t.querySelector('span.arrow').textContent=open?'▼':'▲';
            }})()"
           style="padding:10px 14px; font-size:13px; font-weight:700; cursor:pointer;
                  display:flex; justify-content:space-between; align-items:center;
                  user-select:none;">
        <span>Soil Types (WRB)</span>
        <span class="arrow" style="font-size:10px; color:#78716c; margin-left:8px;">&#9660;</span>
      </div>
      <div id="wrb-content" style="display:none; padding:0 14px 12px 14px; width:240px;">
        {wrb_body_html}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(wrb_legend_html))

    ph_cats_rows = "\n".join(
        f'<tr><td style="padding:2px 6px 2px 0">{label}</td>'
        f'<td style="padding:2px 0; text-align:right">{pct_val}%</td></tr>'
        for label, pct_val in ph_cats.items()
    )
    stats_panel_html = f"""
    <div id="stats-panel" style="position:fixed; bottom:36px; right:12px; z-index:1000;
              background:rgba(255,255,255,0.94); border:1px solid #d6d3d1;
              border-radius:10px; box-shadow:0 8px 18px rgba(0,0,0,0.09);
              font-size:12px; overflow:hidden; width:290px;">
      <div id="stats-header" onclick="document.getElementById('stats-body').style.display=
            document.getElementById('stats-body').style.display==='none'?'block':'none'"
           style="padding:10px 14px; font-weight:700; font-size:13px; cursor:pointer;
                  display:flex; align-items:center; gap:8px;
                  border-bottom:1px solid #e5e7eb;">
        <span style="white-space:nowrap;">Soil Summary &amp; Statistics</span>
        <span style="font-size:10px; color:#78716c; white-space:nowrap;">{total_n} samples &#9660;</span>
      </div>
      <div id="stats-body" style="padding:10px 14px; display:none;">
        <div style="font-size:11px; font-weight:600; color:#57534e; margin-bottom:6px;">pH distribution</div>
        <table style="width:100%; border-collapse:collapse; font-size:11px; margin-bottom:10px;">
          {ph_cats_rows}
        </table>
        <div style="font-size:11px; font-weight:600; color:#57534e; margin-bottom:6px;">Parameter means &plusmn; std</div>
        <table style="width:100%; border-collapse:collapse; font-size:11px;">
          <tr style="color:#78716c; font-size:10px;"><th style="text-align:left">Parameter</th><th style="text-align:right">Value</th></tr>
          {stats_rows_html}
        </table>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_panel_html))

    overlay_vars_js = json.dumps(overlay_vars)
    samples_per_subset_js = json.dumps(samples_per_subset)
    param_colors_js = "{" + ", ".join(
        f'"{k}": {CMAP_STOPS[k]}' for k in raster_b64
    ) + "}"
    param_ranges_js = "{" + ", ".join(
        f'"{k}": [{raster_ranges[k][0]}, {raster_ranges[k][1]}]' for k in raster_b64
    ) + "}"
    param_labels_js = "{" + ", ".join(
        f'"{k}": "{PARAMS[k][0]}"' for k in raster_b64
    ) + "}"
    raster_grids_js = json.dumps(raster_grids)
    grid_bounds_js = json.dumps([lat_min, lat_max, lng_min, lng_max])

    js_html = f"""
    <script>
    (function() {{
      var _check = setInterval(function() {{
        if (window.{map_var}) {{
          clearInterval(_check);
          var map = window.{map_var};

          L.control.zoom({{position: 'topright'}}).addTo(map);

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

          if (!map.getPane('wrbPane')) {{
            map.createPane('wrbPane');
            var wp = map.getPane('wrbPane');
            wp.style.filter = 'saturate(0.35) brightness(1.05)';
            wp.style.zIndex = 250;
          }}
          if (wmsLayer && wmsLayer.options.pane !== 'wrbPane') {{
            map.removeLayer(wmsLayer);
            wmsLayer.options.pane = 'wrbPane';
            wmsLayer.addTo(map);
          }}

          if (!document.getElementById('left-stack')) {{
            var stack = document.createElement('div');
            stack.id = 'left-stack';
            stack.style.cssText =
              'position:fixed; top:12px; left:12px; z-index:1000;' +
              'display:flex; flex-direction:column; gap:10px;' +
              'max-height:calc(100vh - 24px); overflow-y:auto; width:260px;';
            document.body.appendChild(stack);
            ['param-selector-panel', 'wrb-legend'].forEach(function(id) {{
              var el = document.getElementById(id);
              if (el) {{
                el.style.position = 'static';
                el.style.top = 'auto';
                el.style.bottom = 'auto';
                el.style.left = 'auto';
                el.style.right = 'auto';
                el.style.width = 'auto';
                stack.appendChild(el);
              }}
            }});
          }}

          var paramColors = {param_colors_js};
          var paramRanges = {param_ranges_js};
          var paramLabels = {param_labels_js};

          // --- Per-pixel value read-out (hover tooltip) -----------------------
          var gridData = {raster_grids_js};
          var gridBounds = {grid_bounds_js};   // [lat_min, lat_max, lng_min, lng_max]
          var GRID_N = {GRID_N};

          // Compact per-parameter formatting for the read-out.
          var tipFmt = {{
            'ph':    {{ label: 'pH',    dp: 1, unit: ''       }},
            'mo':    {{ label: 'OM',    dp: 1, unit: ' %'     }},
            'ce':    {{ label: 'EC',    dp: 2, unit: ' dS/m'  }},
            'n_no3': {{ label: 'N-NO₃', dp: 0, unit: ' mg/kg' }},
            'p':     {{ label: 'P',     dp: 0, unit: ' mg/kg' }},
          }};

          function hexToRgb(h) {{
            h = h.replace('#', '');
            return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
          }}
          // Colour at normalised position t (0..1) along a parameter's gradient stops,
          // mirroring the matplotlib LinearSegmentedColormap used for the raster.
          function colorAt(param, t) {{
            var stops = paramColors[param];
            if (!stops || !stops.length) return '#888';
            t = Math.max(0, Math.min(1, t));
            var seg = t * (stops.length - 1);
            var i = Math.min(stops.length - 2, Math.floor(seg));
            var f = seg - i;
            var a = hexToRgb(stops[i]), b = hexToRgb(stops[i+1]);
            var r = Math.round(a[0]+(b[0]-a[0])*f);
            var g = Math.round(a[1]+(b[1]-a[1])*f);
            var bl = Math.round(a[2]+(b[2]-a[2])*f);
            return 'rgb(' + r + ',' + g + ',' + bl + ')';
          }}

          var tip = document.createElement('div');
          tip.id = 'px-tooltip';
          tip.style.cssText =
            'position:fixed; pointer-events:none; z-index:1200; display:none;' +
            'align-items:center; gap:8px; padding:6px 10px; font-size:13px;' +
            'background:rgba(255,255,255,0.96); border:1px solid #e5e7eb;' +
            'border-radius:8px; box-shadow:0 6px 16px rgba(0,0,0,0.12);' +
            'font-family:inherit; white-space:nowrap;';
          tip.innerHTML =
            '<span id="px-dot" style="width:11px; height:11px; border-radius:50%;' +
            'border:1px solid rgba(0,0,0,0.15); flex-shrink:0;"></span>' +
            '<span><b id="px-label" style="color:#57534e; font-weight:600;"></b>' +
            '<span id="px-value" style="margin-left:6px; font-variant-numeric:tabular-nums;"></span></span>';
          document.body.appendChild(tip);
          function hideTip() {{ tip.style.display = 'none'; }}

          var LOW_SAMPLE_THRESHOLD = 50;

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

            var yearLabel = (year === 'all') ? 'all years' : year;
            var n = samplesPerSubset[year] || 0;
            document.getElementById('legend-title').textContent =
              paramLabels[param] + ' · ' + yearLabel + ' · samples=' + n;

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

            if (!hasRaster) {{
              warnEl.textContent = 'No data for ' + yearLabel + ' — fewer than 3 samples for this parameter';
              warnEl.style.display = 'block';
            }} else if (year !== 'all' && n > 0 && n < LOW_SAMPLE_THRESHOLD) {{
              warnEl.textContent = 'samples=' + n + ' — limited spatial coverage, interpret with care';
              warnEl.style.display = 'block';
            }} else {{
              warnEl.style.display = 'none';
            }}
          }}

          function switchView(param, year) {{
            Object.keys(overlayLayers).forEach(function(p) {{
              Object.keys(overlayLayers[p]).forEach(function(s) {{
                if (overlayLayers[p][s]) map.removeLayer(overlayLayers[p][s]);
              }});
            }});
            var layer = overlayLayers[param] && overlayLayers[param][year];
            hideTip();
            if (layer) {{
              map.addLayer(layer);
              updateLegend(param, year, true);
              map.getContainer().style.cursor = 'crosshair';
            }} else {{
              updateLegend(param, year, false);
              map.getContainer().style.cursor = '';
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

          // Follow-cursor read-out: map cursor lat/lng → grid cell → interpolated value.
          var lat_min = gridBounds[0], lat_max = gridBounds[1];
          var lng_min = gridBounds[2], lng_max = gridBounds[3];
          var rafPending = false, lastEvt = null;
          function renderTip() {{
            rafPending = false;
            var e = lastEvt;
            if (!e) return;
            var param = paramSelect.value, year = yearSelect.value;
            var grid = gridData[param] && gridData[param][year];
            if (!grid) {{ hideTip(); return; }}
            var lat = e.latlng.lat, lng = e.latlng.lng;
            var col = Math.round((lng - lng_min) / (lng_max - lng_min) * (GRID_N - 1));
            var row = Math.round((lat_max - lat) / (lat_max - lat_min) * (GRID_N - 1));
            if (row < 0 || row >= GRID_N || col < 0 || col >= GRID_N) {{ hideTip(); return; }}
            var val = grid[row][col];
            if (val === null || val === undefined) {{ hideTip(); return; }}

            var fmt = tipFmt[param] || {{ label: paramLabels[param] || param, dp: 2, unit: '' }};
            var range = paramRanges[param] || [0, 1];
            var t = (val - range[0]) / ((range[1] - range[0]) || 1);
            document.getElementById('px-dot').style.background = colorAt(param, t);
            document.getElementById('px-label').textContent = fmt.label;
            document.getElementById('px-value').textContent = val.toFixed(fmt.dp) + fmt.unit;

            tip.style.display = 'flex';
            // Offset from cursor; flip near the viewport edges so it stays visible.
            var oe = e.originalEvent;
            var w = tip.offsetWidth, h = tip.offsetHeight;
            var x = oe.clientX + 14, y = oe.clientY + 14;
            if (x + w > window.innerWidth - 8)  x = oe.clientX - w - 14;
            if (y + h > window.innerHeight - 8) y = oe.clientY - h - 14;
            tip.style.left = x + 'px';
            tip.style.top  = y + 'px';
          }}
          map.on('mousemove', function(e) {{
            lastEvt = e;
            if (!rafPending) {{ rafPending = true; requestAnimationFrame(renderTip); }}
          }});
          map.on('mouseout', hideTip);

          document.getElementById('wms-opacity-slider').addEventListener('input', function() {{
            var val = this.value / 100;
            document.getElementById('wms-opacity-label').textContent = this.value + '%';
            if (wmsLayer) wmsLayer.setOpacity(val);
            document.documentElement.style.setProperty('--wrb-swatch-opacity', val);
          }});

          switchView('ph', 'all');
        }}
      }}, 50);
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_html))

    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_MAP))
    algorithm.logger.info(f"Map saved: {OUTPUT_MAP}")
    algorithm.logger.info(
        f"  {total_n} samples, {len(raster_b64)} IDW rasters generated"
    )


# ============================================================================
# ENTRY POINT (ocean-runner)
# ============================================================================

@algorithm.validate
def validate(algorithm: Algorithm) -> None:
    # soil-mapping reads PDFs directly from /data/inputs and does not depend on
    # OceanProtocol DDO metadata, so we skip the default DDO assertion that
    # would fail when the container is launched outside the Ocean job runner.
    _load_env("OPENWEBUI_API_KEY")
    _load_env("LLM_MODEL")


@algorithm.run
async def run(algorithm: Algorithm) -> list[dict]:
    algorithm.logger.info("=" * 60)
    algorithm.logger.info("Soil Data Extraction & Map Generation")
    algorithm.logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_inputs()

    # Phase 1: LLM extraction (concurrent, bounded by LLM_CONCURRENCY)
    async with httpx.AsyncClient(timeout=300) as llm_client:
        await _warmup_llm(llm_client)
        records = await extract_all_pdfs(llm_client)

    if not records:
        raise RuntimeError("No records extracted")

    # Phase 2: Catastro geocoding (concurrent, bounded by GEO_CONCURRENCY)
    algorithm.logger.info("Geocoding via Spain Catastro API...")
    async with httpx.AsyncClient(timeout=15) as geo_client:
        records = await geocode_records(geo_client, records)

    # Phase 3: map generation (CPU-bound, synchronous)
    rows = records_to_map_rows(records)
    generate_map(rows)

    algorithm.logger.info("Done.")
    return rows


@algorithm.save_results
async def save(algorithm: Algorithm, result: list[dict], base: Path) -> None:
    # generate_map() already writes /data/outputs/soil-characteristics-map.html.
    return


if __name__ == "__main__":
    algorithm()
