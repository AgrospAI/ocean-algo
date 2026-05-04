#!/usr/bin/env python3

import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# --- Paths ------------------------------------------------------------------
INPUT_DIR = Path("/data/inputs")
OUTPUT_DIR = Path("/data/outputs")
RAW_DIR = Path("/tmp/soil-pdfs")

OUTPUT_MAP = OUTPUT_DIR / "soil-characteristics-map.html"

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
# PDF FORMAT DETECTION
# ============================================================================

def detect_pdf_type(text: str) -> str:
    """
    Identify which laboratory report format a PDF contains.
    Returns one of: "butlleti", "agrolab", "skip", "eurofins_xk", "unknown".
    """
    if "BUTLLETÍ D'ANÀLISIS" in text:
        return "butlleti"
    if "AGROLAB" in text or "Análisis de Tierras" in text:
        return "agrolab"
    # Eurofins XK format: only classify as soil if sample type is explicitly soil.
    # This avoids false positives for water, foliar, and other analysis types
    # that share the same Eurofins XK report structure.
    if "XK" in text and ("AR-" in text or "Informe analít" in text):
        if "Suelo" in text or "Sòls" in text or "Soil" in text:
            return "eurofins_xk"
        return "skip"
    return "unknown"


# ============================================================================
# PHASE 1: PDF TEXT EXTRACTION
# ============================================================================

def pdf_to_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pdftotext (no layout mode for clean line-per-cell)."""
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True, timeout=30
        )
        return result.stdout.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] pdftotext failed for {pdf_path.name}: {e}")
        return ""


def _infer_year(pdf_path: Path) -> int | None:
    """Try to infer the sample year from a year-shaped directory part or filename."""
    for part in pdf_path.parts:
        if re.match(r"^20\d{2}$", part):
            return int(part)
    m = re.search(r"20\d{2}", pdf_path.stem)
    return int(m.group(0)) if m else None


def parse_parcella(raw: str) -> tuple[int | None, int]:
    """
    Parse Parcella field into (parcela_num, recinto_num).

    Examples:
      "143-R:1"        → (143, 1)
      "80-R:"          → (80, 1)
      "503-R2"         → (503, 2)
      "235 R1"         → (235, 1)
      "92r-1ha-0,66"   → (92, 1)   # digit after r is part of 'ha' field
      "35r-4ha-1,67"   → (35, 1)   # same
      "10075r-6ha-0,66"→ (10075, 1)
      "121-R2-1.17HA"  → (121, 2)
      "5-R1"           → (5, 1)
      "4-R1"           → (4, 1)
      "5r-1ha"         → (5, 1)
    """
    raw = raw.strip()
    # Extract leading integer (parcela number)
    m = re.match(r'^(\d+)', raw)
    if not m:
        return None, 1
    parcela = int(m.group(1))

    # Try to find recinto number after R or r
    # Patterns: -R:3, R2, -R2, r1, -r:1
    # But NOT when followed immediately by 'ha' (that's hectares, not recinto)
    rec_match = re.search(r'[Rr]:?(\d+)(?!ha)', raw[m.end():], re.IGNORECASE)
    if rec_match:
        recinto = int(rec_match.group(1))
    else:
        recinto = 1

    return parcela, recinto


def _extract_sigpac(text: str) -> tuple[int, int, int, int, int] | None:
    """
    Find a SIGPAC parcel reference embedded in PDF text.
    Format: province:municipality:aggregate:zone:polygon:parcella:enclosure
    Example: 25:287:0:0:6:2:1
    Returns (province, municipality_3digit, polygon, parcella, enclosure) or None.
    """
    m = re.search(r'\b(\d{1,2}):(\d{1,3}):(\d+):(\d+):(\d+):(\d+):(\d+)\b', text)
    if not m:
        return None
    prov = int(m.group(1))
    mun  = int(m.group(2))
    pol  = int(m.group(5))
    par  = int(m.group(6))
    rec  = int(m.group(7))
    if not (1 <= prov <= 52 and 1 <= mun <= 999 and pol > 0 and par > 0):
        return None
    return prov, mun, pol, par, rec


def parse_terme_municipal(raw: str) -> tuple[str, int | None]:
    """
    Parse Terme Municipal field into (municipality_name, ine_code).

    Examples:
      "CORBINS-25094"     → ("CORBINS", 25094)
      "LLEIDA-25900"      → ("LLEIDA", 25900)   # will be overridden later
      "25094 - CORBINS"   → ("CORBINS", 25094)
      "CORBINS"           → ("CORBINS", None)
      "TORRES DE SEGRE-25289" → ("TORRES DE SEGRE", 25289)
    """
    raw = raw.strip()

    # Format: "NAME-XXXXX" or "NAME - XXXXX"
    m = re.match(r'^([A-ZÀ-Ú\s]+?)\s*[-–]\s*(\d{5})\s*$', raw, re.IGNORECASE)
    if m:
        name = m.group(1).strip().upper()
        code = int(m.group(2))
        return name, code

    # Format: "XXXXX - NAME" (reversed)
    m = re.match(r'^(\d{5})\s*[-–]\s*(.+)$', raw)
    if m:
        name = m.group(2).strip().upper()
        code = int(m.group(1))
        return name, code

    # Just a name
    return raw.upper().strip(), None


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


def extract_value_after_label(lines: list[str], label: str) -> str | None:
    """Find a label in lines and return the value on the next non-empty line."""
    for i, line in enumerate(lines):
        if label.lower() in line.lower():
            # Look for next non-empty line
            for j in range(i + 1, min(i + 4, len(lines))):
                val = lines[j].strip()
                if val:
                    return val
    return None


def parse_soil_params(text: str) -> dict:
    """
    Extract soil parameters from the Eurofins PDF text.
    Uses the cleaner summary page (page 3) when available,
    falls back to the analytic page 1/2.
    """
    params = {
        "pH": None, "MO": None, "CE": None,
        "Caliza": None, "N_Nitrico": None,
        "Fosforo": None, "Potasio": None,
        "Cultiu": None, "Texture": None,
    }

    lines = text.split("\n")

    def safe_float(s: str | None) -> float | None:
        if s is None:
            return None
        # Take only first token, strip units
        token = s.strip().split()[0] if s.strip() else ""
        token = token.replace(",", ".")
        token = re.sub(r"[^\d.\-]", "", token)
        try:
            return float(token)
        except ValueError:
            return None

    # ----------------------------------------------------------------
    # Try summary table first (Informe de valors de referència de Sòl)
    # The summary block has lines like:
    #   "pH\n8.2"  or  "pH                   8.2"
    # ----------------------------------------------------------------
    summary_start = -1
    for i, line in enumerate(lines):
        if ("Informe de valors de refer" in line
                or "Propietats bàsiques" in line
                or "Propiedades básicas" in line):
            summary_start = i
            break

    if summary_start >= 0:
        summary_lines = lines[summary_start:]

        def find_val(label: str) -> str | None:
            return extract_value_after_label(summary_lines, label)

        params["pH"] = safe_float(find_val("pH"))
        params["CE"] = safe_float(find_val("Conductivitat el"))
        params["MO"] = safe_float(find_val("orgànica oxidable") or find_val("orgánica oxidable"))
        params["Caliza"] = safe_float(find_val("Carbonat càlcic") or find_val("Carbonato cálcico"))
        params["N_Nitrico"] = safe_float(find_val("Nitrogen nítric") or find_val("Nitrógeno nítrico"))
        params["Fosforo"] = safe_float(find_val("Fòsfor") or find_val("Fósforo"))
        params["Potasio"] = safe_float(find_val("Potassi") or find_val("Potasio"))
        params["Texture"] = find_val("Texture") or find_val("Textura")

    # ----------------------------------------------------------------
    # Fallback: parse from analytic page using XK codes and result values
    # Pattern: after "XK pH" line, the result appears a few lines below
    # ----------------------------------------------------------------
    if params["pH"] is None:
        # Pattern on page 1: "XK007" followed eventually by "pH\n\n8.2"
        m = re.search(r'pH\s+(\d+\.\d+)', text)
        if m:
            params["pH"] = safe_float(m.group(1))

    if params["CE"] is None:
        m = re.search(r'(?:Conductivitat|Conductividad)[^\n]*\n+(\d+[\.,]\d+)\s*dS', text)
        if m:
            params["CE"] = safe_float(m.group(1))

    if params["MO"] is None:
        m = re.search(r'orgànica oxidable[^\n]*\n+(\d+[\.,]\d+)\s*%', text)
        if not m:
            m = re.search(r'orgánica[^\n]*\n+(\d+[\.,]\d+)\s*%', text)
        if m:
            params["MO"] = safe_float(m.group(1))

    if params["Caliza"] is None:
        m = re.search(r'Carbonat càlcic[^\n]*\n+(\d+[\.,]\d*)\s*%', text)
        if not m:
            m = re.search(r'Carbonato c.lcico[^\n]*\n+(\d+[\.,]\d*)\s*%', text)
        if m:
            params["Caliza"] = safe_float(m.group(1))

    if params["N_Nitrico"] is None:
        m = re.search(r'Nitrogen nítric[^\n]*\n+(\d+[\.,]?\d*)\s*mg', text)
        if not m:
            m = re.search(r'Nitr.geno n.trico[^\n]*\n+(\d+[\.,]?\d*)\s*mg', text)
        if m:
            params["N_Nitrico"] = safe_float(m.group(1))

    if params["Fosforo"] is None:
        m = re.search(r'Fòsfor sms[^\n]*\n+(\d+[\.,]\d*)\s*mg', text)
        if m:
            params["Fosforo"] = safe_float(m.group(1))

    if params["Potasio"] is None:
        m = re.search(r'Potassi sms[^\n]*\n+(\d+[\.,]\d*)\s*mg', text)
        if m:
            params["Potasio"] = safe_float(m.group(1))

    # Cultiu (crop type)
    m = re.search(r'Cultiu\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑ][^\n]+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'Cultivo\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑ][^\n]+)', text, re.IGNORECASE)
    if m:
        params["Cultiu"] = m.group(1).strip()

    return params


def extract_eurofins_record(pdf_path: Path, year: int | None) -> dict | None:
    """
    Extract one record from a Eurofins Fruilar PDF.
    Returns None if Polígon/Parcella are not found.
    """
    text = pdf_to_text(pdf_path)
    if not text:
        return None

    lines = [l.strip() for l in text.split("\n")]

    # --- Locate Polígon / Parcella / Terme Municipal ---
    poligon = None
    parcella_raw = None
    terme_raw = None

    for i, line in enumerate(lines):
        if line == "Polígon" or line == "Polígono" or re.match(r'^Pol[ií]g[oó]n$', line, re.IGNORECASE):
            # Value is on the next non-empty line
            for j in range(i + 1, min(i + 5, len(lines))):
                val = lines[j].strip()
                if val and re.match(r'^\d+$', val):
                    poligon = int(val)
                    break

        elif line == "Parcella" or line == "Parcela" or re.match(r'^Parcel+a$', line, re.IGNORECASE):
            for j in range(i + 1, min(i + 5, len(lines))):
                val = lines[j].strip()
                if val and re.match(r'^\d+', val):
                    parcella_raw = val
                    break

        elif "Terme Municipal" in line or "Término Municipal" in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                val = lines[j].strip()
                if val and not any(kw in val for kw in ["Denominació", "Cultiu", "Varietat"]):
                    terme_raw = val
                    break

    # 2021 Catastro format: line "Catastro:" followed by "Polígono X Parcela Y"
    if poligon is None:
        for i, line in enumerate(lines):
            if line.strip() in ("Catastro:", "Catastro"):
                # Next non-empty line should be "Polígono X Parcela Y"
                for j in range(i + 1, min(i + 4, len(lines))):
                    val = lines[j].strip()
                    if re.match(r'Pol[ií]gono?\s+\d+', val, re.IGNORECASE):
                        m2 = re.match(r'Pol[ií]gono?\s+(\d+)\s+Parce[ll]?a?\s+(\S+)', val, re.IGNORECASE)
                        if m2:
                            poligon = int(m2.group(1))
                            parcella_raw = m2.group(2)
                        break
            if line.strip() == "Localidad:" and terme_raw is None:
                for j in range(i + 1, min(i + 3, len(lines))):
                    val = lines[j].strip()
                    if val:
                        terme_raw = val
                        break

    # SIGPAC fallback: look for province:municipality:agg:zone:polygon:parcella:enclosure
    sigpac_ine = None
    sigpac_rec = None
    if poligon is None or parcella_raw is None:
        sigpac = _extract_sigpac(text)
        if sigpac:
            s_prov, s_mun, s_pol, s_par, s_rec = sigpac
            if poligon is None:
                poligon = s_pol
            if parcella_raw is None:
                parcella_raw = str(s_par)
            sigpac_ine = s_prov * 1000 + s_mun
            sigpac_rec = s_rec
            print(f"    [SIGPAC] prov={s_prov} mun={s_mun} pol={s_pol} par={s_par} rec={s_rec}")

    if poligon is None or parcella_raw is None:
        print(f"  [SKIP] No Polígon/Parcella in: {pdf_path.name}")
        return None

    parcela, recinto = parse_parcella(parcella_raw)
    if parcela is None:
        print(f"  [SKIP] Could not parse parcella '{parcella_raw}' in: {pdf_path.name}")
        return None
    if sigpac_rec is not None:
        recinto = sigpac_rec

    muni_name, raw_ine = ("UNKNOWN", None)
    if terme_raw:
        muni_name, raw_ine = parse_terme_municipal(terme_raw)
    if sigpac_ine is not None:
        raw_ine = sigpac_ine

    soil = parse_soil_params(text)

    record = {
        "pdf_name": pdf_path.name,
        "year": year,
        "poligon": poligon,
        "parcella_raw": parcella_raw,
        "parcela": parcela,
        "recinto": recinto,
        "muni_name": muni_name,
        "raw_ine": raw_ine,
        "cultiu": soil["Cultiu"] or "",
        "pH": soil["pH"],
        "MO": soil["MO"],
        "CE": soil["CE"],
        "Caliza": soil["Caliza"],
        "N_Nitrico": soil["N_Nitrico"],
        "Fosforo": soil["Fosforo"],
        "Potasio": soil["Potasio"],
        "Texture": soil["Texture"] or "",
        "lat": None,
        "lng": None,
    }
    return record


def parse_butlleti_params(text: str) -> dict:
    """
    Extract soil parameters from the Eurofins BUTLLETÍ d'Anàlisis format
    (used by Fruits de Ponent and similar clients).
    Each result row looks like: LABEL  value  units  method  interpretation
    """
    params = {
        "pH": None, "MO": None, "CE": None,
        "Caliza": None, "N_Nitrico": None,
        "Fosforo": None, "Potasio": None,
        "Cultiu": None, "Texture": None,
    }

    def safe_float(s: str | None) -> float | None:
        if s is None:
            return None
        token = s.strip().split()[0] if s.strip() else ""
        token = token.replace(",", ".")
        token = re.sub(r"[^\d.\-]", "", token)
        try:
            return float(token)
        except ValueError:
            return None

    # Each parameter is on one line: NAME ... value units ...
    patterns = [
        ("CE",        r'COND\.ELEC\.[^\n]*([\d,]+)\s*dS'),
        ("MO",        r'MAT\.ORGANICA[^\n]*([\d,]+)\s*%'),
        ("Fosforo",   r'FOSFOR[^\n]*([\d,]+)\s*mg'),
        ("Potasio",   r'POTASSI[^\n]*([\d,]+)\s*mg'),
        ("N_Nitrico", r'NITROGEN-NITRIC[^\n]*([\d,]+)\s*mg'),
        ("pH",        r'\bpH\b[^\n]*([\d,]+)'),
        ("Caliza",    r'CARBONAT[^\n]*([\d,]+)\s*%'),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            params[key] = safe_float(m.group(1))

    # Cultiu
    m = re.search(r'CULTIU[:\s]+([A-ZÁÉÍÓÚÀÈÌÒÙÑ][^\n]+)', text, re.IGNORECASE)
    if m:
        params["Cultiu"] = m.group(1).strip()

    return params


def extract_butlleti_record(pdf_path: Path) -> dict | None:
    """
    Extract one record from a Eurofins BUTLLETÍ d'Anàlisis format PDF.
    These have POL.: and PARCEL·LA: fields with municipality in T.M.:
    """
    text = pdf_to_text(pdf_path)
    if not text:
        return None

    lines = [l.strip() for l in text.split("\n")]

    # --- Municipality from T.M.: ---
    muni_name = "UNKNOWN"
    for i, line in enumerate(lines):
        if re.match(r'^T\.M\.:?$', line, re.IGNORECASE):
            for j in range(i + 1, min(i + 4, len(lines))):
                val = lines[j].strip()
                if val and not re.match(r'^(LOCALITZ|LOCALIZ|POL|PARCEL)', val, re.IGNORECASE):
                    muni_name = val.upper()
                    break
            break

    # --- POL. / PARCEL·LA via column-reordering logic ---
    # pdftotext outputs labels in one column, then values in another.
    # After "POL.:" appear all label lines (PARCEL·LA:, CULTIU:, VARIETAT:),
    # then blank lines, then the numeric values in order.
    poligon = None
    parcela = None
    pol_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^POL\.?:?$', line, re.IGNORECASE):
            pol_idx = i
            break

    if pol_idx is not None:
        integers_found = []
        for j in range(pol_idx + 1, min(pol_idx + 20, len(lines))):
            val = lines[j].strip()
            if re.match(r'^\d+$', val):
                integers_found.append(int(val))
            if len(integers_found) == 2:
                break
        if integers_found:
            poligon = integers_found[0]
        if len(integers_found) >= 2:
            parcela = integers_found[1]

    sigpac_ine = None
    if poligon is None or parcela is None:
        sigpac = _extract_sigpac(text)
        if sigpac:
            s_prov, s_mun, s_pol, s_par, _rec = sigpac
            if poligon is None:
                poligon = s_pol
            if parcela is None:
                parcela = s_par
            sigpac_ine = s_prov * 1000 + s_mun
            print(f"    [SIGPAC] prov={s_prov} mun={s_mun} pol={s_pol} par={s_par}")

    if poligon is None or parcela is None:
        print(f"  [INFO] No POL/PARCEL·LA in {pdf_path.name} – will not be geocoded")

    # --- Year from DATA INICI: ---
    year = None
    m = re.search(r'DATA INICI:\s*\d{2}/\d{2}/(\d{4})', text)
    if m:
        year = int(m.group(1))

    # --- Resolve municipality code ---
    muni_name_resolved, raw_ine = parse_terme_municipal(muni_name)
    if sigpac_ine is not None:
        raw_ine = sigpac_ine
    soil = parse_butlleti_params(text)

    return {
        "pdf_name": pdf_path.name,
        "year": year,
        "poligon": poligon,
        "parcella_raw": str(parcela) if parcela is not None else "N/A",
        "parcela": parcela,
        "recinto": 1,
        "muni_name": muni_name_resolved,
        "raw_ine": raw_ine,
        "cultiu": soil["Cultiu"] or "",
        "pH": soil["pH"],
        "MO": soil["MO"],
        "CE": soil["CE"],
        "Caliza": soil["Caliza"],
        "N_Nitrico": soil["N_Nitrico"],
        "Fosforo": soil["Fosforo"],
        "Potasio": soil["Potasio"],
        "Texture": soil["Texture"] or "",
        "lat": None,
        "lng": None,
    }


def parse_agrolab_params(text: str) -> dict:
    """Extract soil parameters from AGROLAB Analítica format PDFs."""
    params = {
        "pH": None, "MO": None, "CE": None,
        "Caliza": None, "N_Nitrico": None,
        "Fosforo": None, "Potasio": None,
        "Cultiu": None, "Texture": None,
    }

    def safe_float(s: str | None) -> float | None:
        if s is None:
            return None
        token = s.strip().split()[0] if s.strip() else ""
        token = token.replace(",", ".")
        token = re.sub(r"[^\d.\-]", "", token)
        try:
            return float(token)
        except ValueError:
            return None

    patterns = [
        ("pH",        r'pH agua[^\n]*\n[^\n]*\n\s*([\d.,]+)'),
        ("MO",        r'Materia Org.nica\s+Oxidable\s+([\d.,]+)'),
        ("CE",        r'Conductividad El.ctrica[^\n]*\n[^\n]*([\d.,]+)\s*dS'),
        ("Caliza",    r'Carbonatos Totales[^\n]*\n[^\n]*([\d.,]+)\s*g/100'),
        ("Fosforo",   r'F.sforo.*?Olsen\)[^\n]*\n[^\n]*([\d.,]+)\s*mg'),
        ("Potasio",   r'Potasio.*?Acet[^\n]*\n[^\n]*([\d.,]+)\s*mg'),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            params[key] = safe_float(m.group(1))

    # Texture: appears after "Clasificación TEXTURAL"
    m = re.search(r'Clasificaci.n TEXTURAL\s+([\w ]+)', text, re.IGNORECASE)
    if m:
        params["Texture"] = m.group(1).strip()

    # Cultiu: from "Referencia:" line, e.g. "VIÑEDO (9 Ha)"
    m = re.search(r'Referencia:\s*\n([^\n]+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'Referencia:\s+([^\n]+)', text, re.IGNORECASE)
    if m:
        cultiu_raw = m.group(1).strip()
        # Strip parenthetical hectares/area info
        params["Cultiu"] = re.sub(r'\s*\(.*?\)', '', cultiu_raw).strip()

    return params


def extract_agrolab_record(pdf_path: Path) -> dict:
    """
    Extract a record from an AGROLAB Analítica format PDF.
    These have no Polígon/Parcella – geocoding is not possible.
    """
    text = pdf_to_text(pdf_path)

    # Extract year from reception date
    year = None
    m = re.search(r'Fecha Recepci.n\s+(\d{2}/\d{2}/(\d{4}))', text, re.IGNORECASE)
    if m:
        year = int(m.group(2))

    # Municipality hint from Observaciones
    muni_name = "UNKNOWN"
    m = re.search(r'Observaciones:\s*\n([^\n]+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'Observaciones:\s+([^\n]+)', text, re.IGNORECASE)
    if m:
        obs = m.group(1).strip().upper()
        # "ZONA SOMONTANO (HUESCA)" → keep as location hint
        muni_name = obs

    print(f"  [INFO] No Polígon/Parcella in {pdf_path.name} – will not be geocoded")
    soil = parse_agrolab_params(text)

    return {
        "pdf_name": pdf_path.name,
        "year": year,
        "poligon": None,
        "parcella_raw": "N/A",
        "parcela": None,
        "recinto": None,
        "muni_name": muni_name,
        "raw_ine": None,
        "cultiu": soil["Cultiu"] or "",
        "pH": soil["pH"],
        "MO": soil["MO"],
        "CE": soil["CE"],
        "Caliza": soil["Caliza"],
        "N_Nitrico": soil["N_Nitrico"],
        "Fosforo": soil["Fosforo"],
        "Potasio": soil["Potasio"],
        "Texture": soil["Texture"] or "",
        "lat": None,
        "lng": None,
    }


def extract_cota220_soil_record(pdf_path: Path) -> dict:
    """
    Extract a record from a Eurofins soil analysis PDF without Polígon/Parcella.
    Geocoding is not possible for these records.
    """
    text = pdf_to_text(pdf_path)

    # Extract year from reception date
    year = None
    m = re.search(r'Fecha de recepci.n\s*:\s*\n*\s*(\d{2}/\d{2}/(\d{4}))', text, re.IGNORECASE)
    if m:
        year = int(m.group(2))

    # Description from client field
    muni_name = "UNKNOWN"
    m = re.search(r'Descripci.n por el cliente\s*\n([^\n]+)', text, re.IGNORECASE)
    if m:
        muni_name = m.group(1).strip().upper()

    print(f"  [INFO] No Polígon/Parcella in {pdf_path.name} – will not be geocoded")
    soil = parse_soil_params(text)

    return {
        "pdf_name": pdf_path.name,
        "year": year,
        "poligon": None,
        "parcella_raw": "N/A",
        "parcela": None,
        "recinto": None,
        "muni_name": muni_name,
        "raw_ine": None,
        "cultiu": soil["Cultiu"] or "",
        "pH": soil["pH"],
        "MO": soil["MO"],
        "CE": soil["CE"],
        "Caliza": soil["Caliza"],
        "N_Nitrico": soil["N_Nitrico"],
        "Fosforo": soil["Fosforo"],
        "Potasio": soil["Potasio"],
        "Texture": soil["Texture"] or "",
        "lat": None,
        "lng": None,
    }


def extract_inputs() -> None:
    """Unzip every archive found under INPUT_DIR into RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for zip_path in INPUT_DIR.rglob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)


def extract_all_pdfs() -> list[dict]:
    """Extract records from all PDFs found under RAW_DIR."""
    records = []
    pdfs = sorted(
        {*RAW_DIR.rglob("*.pdf"), *RAW_DIR.rglob("*.PDF")},
        key=lambda p: p.name,
    )

    print(f"\nProcessing {len(pdfs)} PDFs from {RAW_DIR}...")
    for pdf in pdfs:
        print(f"  {pdf.name}")
        text = pdf_to_text(pdf)
        if not text:
            continue
        fmt = detect_pdf_type(text)

        if fmt == "skip":
            print(f"  [SKIP] Water/foliar: {pdf.name}")
        elif fmt == "butlleti":
            rec = extract_butlleti_record(pdf)
            if rec:
                print(f"    → Polígon {rec['poligon']}, Parcella {rec['parcela']}, "
                      f"{rec['muni_name']}")
                records.append(rec)
        elif fmt == "agrolab":
            rec = extract_agrolab_record(pdf)
            records.append(rec)
        elif fmt == "eurofins_xk":
            year = _infer_year(pdf)
            rec = extract_eurofins_record(pdf, year)
            if rec is None:
                # No Polígon/Parcella — capture soil params without geocoding
                rec = extract_cota220_soil_record(pdf)
            if rec:
                if rec["poligon"] is not None:
                    print(f"    → Polígon {rec['poligon']}, Parcella {rec['parcela']} "
                          f"(recinto {rec['recinto']}), {rec['muni_name']}")
                records.append(rec)
        else:
            print(f"  [SKIP] Unrecognised format ({fmt}): {pdf.name}")

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
    if rec.get("poligon") is None:
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

    # --- Per-cluster hull mask ---
    # Each geographic cluster gets its own buffered convex hull so the IDW
    # raster is never shown in the empty space between distant regions.
    from shapely.geometry import Point
    from shapely.prepared import prep

    combined_hull = _cluster_hull(lngs, lats, buffer=0.05, eps=0.5)
    prepared_hull = prep(combined_hull)
    hull_mask = np.array([
        prepared_hull.contains(Point(lon, lat))
        for lon, lat in zip(grid_lng.ravel(), grid_lat.ravel())
    ]).reshape(grid_lng.shape)

    raster_b64 = {}
    raster_ranges = {}
    print("  Computing IDW rasters...")
    for key, (label, extractor) in PARAMS.items():
        pts = [(r["lat"], r["lng"], extractor(r)) for r in all_rows if extractor(r) is not None]
        if len(pts) < 3:
            print(f"    [SKIP] {label}: fewer than 3 valid samples")
            continue
        plats, plngs, pvals = zip(*pts)
        pvals_arr = np.array(pvals, dtype=float)
        vmin = float(np.percentile(pvals_arr, 2))
        vmax = float(np.percentile(pvals_arr, 98))
        if vmin == vmax:
            vmax = vmin + 1  # prevent degenerate colormap
        grid_vals = idw_grid(plats, plngs, pvals, grid_lat, grid_lng)
        raster_b64[key] = raster_to_base64(grid_vals, CMAPS[key], vmin, vmax, hull_mask=hull_mask)
        raster_ranges[key] = (round(vmin, 2), round(vmax, 2))
        print(f"    {label}: [{vmin:.2f}, {vmax:.2f}]  ({len(pts)} samples, 2–98th pct)")

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

    # IDW raster overlays — only pH shown by default
    overlay_vars = {}
    for key in raster_b64:
        overlay = folium.raster_layers.ImageOverlay(
            image=raster_b64[key],
            bounds=[[lat_min, lng_min], [lat_max, lng_max]],
            name=f"IDW: {PARAMS[key][0]}",
            opacity=1.0,
            show=(key == "ph"),
            interactive=False,
            cross_origin=False,
        )
        overlay.add_to(m)
        overlay_vars[key] = overlay.get_name()

    map_var = m.get_name()

    # --- Parameter selector + WMS slider panel (top-left) ---
    param_options_html = "\n".join(
        f'<option value="{k}">{PARAMS[k][0]}</option>'
        for k in raster_b64
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
              border-radius:6px; font-size:13px; background:#fff; cursor:pointer; margin-bottom:14px;">
        {param_options_html}
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
              border:1px solid #bbf7d0; border-radius:5px; padding:3px 7px; display:none;"></div>
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
    overlay_vars_js = "{" + ", ".join(
        f'"{k}": "{v}"' for k, v in overlay_vars.items()
    ) + "}"
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

          // Overlay layer variable name → actual Leaflet layer object
          var overlayVarNames = {overlay_vars_js};
          var overlayLayers = {{}};
          Object.keys(overlayVarNames).forEach(function(k) {{
            overlayLayers[k] = window[overlayVarNames[k]];
          }});

          var wmsLayer = window["{wms_var}"];

          var paramColors = {param_colors_js};
          var paramRanges = {param_ranges_js};
          var paramLabels = {param_labels_js};

          // Optimal range metadata per parameter.
          // optimalRange: [lo, hi] within the parameter's value range (null = no numeric range).
          // direction: 'higher' | 'lower' | null (used when there's no numeric range).
          // note: short descriptive text shown in the green badge.
          var paramOptimal = {{
            'ph':    {{ optimalRange: [6.0, 7.5], direction: null,      note: 'Optimal: 6.0 – 7.5' }},
            'mo':    {{ optimalRange: null,        direction: 'higher',  note: 'Higher → better (organic matter)' }},
            'ce':    {{ optimalRange: [0, 2],      direction: null,      note: 'Optimal: < 2 dS/m (low salinity)' }},
            'n_no3': {{ optimalRange: [10, 40],    direction: null,      note: 'Optimal: 10 – 40 mg/kg' }},
            'p':     {{ optimalRange: [15, 50],    direction: null,      note: 'Optimal: 15 – 50 mg/kg' }},
          }};

          function updateLegend(key) {{
            var colors = paramColors[key];
            var range  = paramRanges[key] || [0, 1];
            var vmin = range[0], vmax = range[1];
            var span = vmax - vmin || 1;

            document.getElementById('legend-gradient').style.background =
              'linear-gradient(to right, ' + colors.join(',') + ')';
            document.getElementById('legend-min').textContent = vmin;
            document.getElementById('legend-max').textContent = vmax;
            document.getElementById('legend-title').textContent = paramLabels[key];

            // Optimal annotation
            var opt = paramOptimal[key];
            var optEl  = document.getElementById('legend-optimal');
            var barEl  = document.getElementById('legend-optimal-bar');

            if (opt && opt.note) {{
              optEl.textContent = '✓ ' + opt.note;
              optEl.style.display = 'block';
            }} else {{
              optEl.style.display = 'none';
            }}

            // Position the translucent highlight bar over the optimal zone
            if (opt && opt.optimalRange) {{
              var lo = opt.optimalRange[0];
              var hi = opt.optimalRange[1];
              // Clamp to visible range
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
              // Highlight rightmost 40% to suggest "higher is better"
              barEl.style.left  = '60%';
              barEl.style.width = '40%';
              barEl.style.display = 'block';
            }} else {{
              barEl.style.display = 'none';
            }}
          }}

          function switchParam(key) {{
            Object.keys(overlayLayers).forEach(function(k) {{
              if (overlayLayers[k]) {{
                if (k === key) map.addLayer(overlayLayers[k]);
                else           map.removeLayer(overlayLayers[k]);
              }}
            }});
            updateLegend(key);
          }}

          document.getElementById('param-select').addEventListener('change', function() {{
            switchParam(this.value);
          }});

          document.getElementById('wms-opacity-slider').addEventListener('input', function() {{
            var val = this.value / 100;
            document.getElementById('wms-opacity-label').textContent = this.value + '%';
            if (wmsLayer) wmsLayer.setOpacity(val);
          }});

          // Initialise
          switchParam('ph');
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

    # Phase 3: Generate map directly from geocoded records (no CSV written)
    all_rows = records_to_map_rows(records)
    generate_map(all_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
