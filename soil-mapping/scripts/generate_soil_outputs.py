#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean" / "soil-csv"
MAP_DIR = ROOT / "output"

SAMPLE_CSV = CLEAN_DIR / "soil-data-extracted-raw.csv"
SUMMARY_CSV = CLEAN_DIR / "soil-data-municipality-summary.csv"
MAP_HTML = MAP_DIR / "soil-characteristics-map.html"


MUNICIPALITY_CENTROIDS = {
    "ALCARRAS": (41.5663, 0.5245),
    "ALFES": (41.5226, 0.6204),
    "BELLVIS": (41.6726, 0.8186),
    "BENAVENT DE SEGRIA": (41.6969, 0.6332),
    "CORBINS": (41.6903, 0.6976),
    "HUESCA": (42.1362, -0.4087),
    "LLEIDA": (41.6176, 0.6200),
    "MIRALCAMP": (41.6040, 0.8811),
    "ONDA": (39.9649, -0.2633),
    "SARROCA DE LLEIDA": (41.4590, 0.5618),
    "SUDANELL": (41.5567, 0.5668),
    "TORREGROSSA": (41.5815, 0.8254),
    "TORRELAMEU": (41.7053, 0.7050),
    "TORRES DE SEGRE": (41.5337, 0.5143),
    "TORRE-SERONA": (41.6718, 0.6293),
    "VILANOVA DE LA BARCA": (41.6887, 0.7298),
    "VILANOVA DE SEGRIA": (41.7119, 0.6209),
}


NUMERIC_FIELDS = [
    "pH",
    "MO (%)",
    "CE (dS/m)",
    "Caliza (%)",
    "Nitrogeno (%)",
    "N_Nitrico (mg/kg)",
    "Fosforo (mg/kg)",
    "Potasio (mg/kg)",
    "Calcio (mg/kg)",
    "Magnesio (mg/kg)",
    "Sodio (mg/kg)",
]

CONDUCTIVITY_LABELS = [
    "XK CONDUCT. ELECTRICA 25OC (EXTR. 1:5 H2O)",
    "XK CONDUCT. ELECTRICA 25C (EXTR. 1:5 H2O)",
    "XK CONDUCT. ELECTRICA 25OC (EXTRACTE 1:5 H2O)",
    "XK CONDUCT. ELECTRICA 25C (EXTRACTE 1:5 H2O)",
]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.upper().strip()


def squash_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_number(raw: str | None) -> float | None:
    if not raw:
        return None
    cleaned = raw.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def pretty_label(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip()
    if text.isupper():
        return text.title()
    return text


def source_group_for(pdf_path: Path) -> str:
    relative = pdf_path.relative_to(ROOT)
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "raw":
        return parts[2]
    return pdf_path.parent.name


def sanitize_municipality(raw: str | None) -> str | None:
    if not raw:
        return None
    value = squash_whitespace(raw)
    value = re.sub(r"^\d{5}\s*-\s*", "", value)
    value = re.sub(r"\s*-\s*\d{5}$", "", value)
    value = re.sub(r"\d{5}$", "", value).strip(" -")
    if any(token in value for token in ["HA:", "_", "/"]):
        return None
    return normalize_text(value)


def find_line_value(text: str, labels: Iterable[str]) -> str | None:
    for label in labels:
        match = re.search(rf"{label}\s+([^\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_code_value(text: str, code: str, unit_hint: str | None = None) -> float | None:
    pattern = rf"{re.escape(code)}(?:(?!XK\d{{3}}).){{0,220}}?([-+]?\d+(?:[.,]\d+)?)"
    if unit_hint:
        pattern = rf"{re.escape(code)}(?:(?!XK\d{{3}}).){{0,220}}?([-+]?\d+(?:[.,]\d+)?)\s*{unit_hint}"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return parse_number(match.group(1)) if match else None


def extract_label_value(text: str, labels: Iterable[str], unit_hint: str, max_chars: int = 260) -> float | None:
    for label in labels:
        match = re.search(re.escape(label), text, re.IGNORECASE)
        if not match:
            continue
        window = text[match.start(): match.start() + max_chars]
        value_match = re.search(rf"([-+]?\d+(?:[.,]\d+)?)\s*{unit_hint}", window, re.IGNORECASE)
        if value_match:
            return parse_number(value_match.group(1))
    return None


def extract_segment_value(
    text: str,
    start_labels: Iterable[str],
    end_labels: Iterable[str],
    unit_hint: str | None = None,
    strategy: str = "last",
) -> float | None:
    for label in start_labels:
        start = text.find(label)
        if start == -1:
            continue

        end_positions = [text.find(end_label, start + len(label)) for end_label in end_labels]
        end_candidates = [position for position in end_positions if position != -1]
        end = min(end_candidates) if end_candidates else start + 400
        segment = text[start:end]

        if unit_hint:
            matches = re.findall(rf"([-+]?\d+(?:[.,]\d+)?)\s*{unit_hint}", segment, re.IGNORECASE)
        else:
            matches = re.findall(r"(?<![:/])\b([-+]?\d+(?:[.,]\d+)?)\b", segment)

        if not matches:
            continue
        raw = matches[0] if strategy == "first" else matches[-1]
        return parse_number(raw)
    return None


def infer_municipality_from_known_places(text: str) -> str | None:
    matches = [name for name in MUNICIPALITY_CENTROIDS if name in text]
    if not matches:
        return None
    return max(matches, key=len)


def read_pdf_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_eurofins(pdf_path: Path, text: str) -> dict[str, object] | None:
    first_page = text.split("\f", 1)[0]
    municipality = sanitize_municipality(
        find_line_value(first_page, ["Terme Municipal", "Municipio"])
    )
    if not municipality or municipality not in MUNICIPALITY_CENTROIDS:
        municipality = infer_municipality_from_known_places(first_page) or municipality

    crop = find_line_value(first_page, ["Cultiu", "Cultivo"])
    client_description = find_line_value(first_page, ["Descripcio pel client", "Descripcion por el cliente"]) or ""

    sample_id_match = re.search(r"(AR-\d{2}-XK-\d{6}-\d{2})", first_page)
    report_date_match = re.search(r"\n(?:Data|Fecha)\s+(\d{2}/\d{2}/\d{4})", first_page)
    year_match = re.search(r"/(20\d{2})/", pdf_path.as_posix())

    nutrient_start = re.search(r"NITROGEN(?:O)? NITRIC[O]? \(N-NO3\)", first_page)
    nutrient_window = first_page[nutrient_start.start(): nutrient_start.start() + 1200] if nutrient_start else ""
    nutrient_values = [
        parse_number(match)
        for match in re.findall(r"([-+]?\d+(?:[.,]\d+)?)\s*MG/KG", nutrient_window, re.IGNORECASE)
    ]

    def nutrient_value(index: int) -> float | None:
        return nutrient_values[index] if len(nutrient_values) > index else None

    def first_page_value(label: str, unit_hint: str, span: int = 240) -> float | None:
        idx = first_page.find(label)
        if idx == -1:
            return None
        window = first_page[idx:idx + span]
        match = re.search(rf"([-+]?\d+(?:[.,]\d+)?)\s*{unit_hint}", window, re.IGNORECASE)
        return parse_number(match.group(1)) if match else None

    ph_match = re.search(r"\bPH\b(?:\s+XK008)?\s+([-+]?\d+(?:[.,]\d+)?)\b", first_page, re.IGNORECASE)
    ce_match = re.search(
        r"(?:CONDUCTIVITAT|CONDUCTIVIDAD)\s+ELECTRICA\s+25(?: ?°?C|OC)\s+([-+]?\d+(?:[.,]\d+)?)\s*DS/M",
        first_page,
        re.IGNORECASE,
    )
    mo_match = re.search(
        r"MATERIA\s+ORGANICA\s+OXIDABLE(?:(?!\d).){0,40}([-+]?\d+(?:[.,]\d+)?)\s*%\s*S\.M\.S\.",
        first_page,
        re.IGNORECASE | re.DOTALL,
    )
    caliza_match = re.search(
        r"CARBONAT[OA]\s+CALCIC[OA]?\s+EQUIVALENT(?:E)?(?:(?!\d).){0,80}([-+]?\d+(?:[.,]\d+)?)\s*%\s*S\.M\.S\.",
        first_page,
        re.IGNORECASE | re.DOTALL,
    )
    nitrogeno_total_match = re.search(
        r"NITROGEN(?:O)?\s+TOTAL\s*\(N\)(?:(?!\d).){0,80}([-+]?\d+(?:[.,]\d+)?)\s*%\s*S\.M\.S\.",
        first_page,
        re.IGNORECASE | re.DOTALL,
    )

    row = {
        "source_file": pdf_path.relative_to(ROOT).as_posix(),
        "source_group": source_group_for(pdf_path),
        "sample_id": sample_id_match.group(1) if sample_id_match else pdf_path.stem,
        "year": int(year_match.group(1)) if year_match else None,
        "report_date": report_date_match.group(1) if report_date_match else "",
        "municipio": municipality or "",
        "cultivo": pretty_label(crop),
        "descripcion": client_description,
        "pH": parse_number(ph_match.group(1)) if ph_match else None,
        "MO (%)": parse_number(mo_match.group(1)) if mo_match else first_page_value("MATERIA ORGANICA OXIDABLE", r"%\s*S\.M\.S\."),
        "CE (dS/m)": parse_number(ce_match.group(1)) if ce_match else first_page_value("CONDUCTIVIDAD ELECTRICA 25", r"DS/M"),
        "Caliza (%)": parse_number(caliza_match.group(1)) if caliza_match else first_page_value("CARBONATO CALCICO EQUIVALENTE", r"%\s*S\.M\.S\."),
        "Nitrogeno (%)": parse_number(nitrogeno_total_match.group(1)) if nitrogeno_total_match else first_page_value("NITROGENO TOTAL (N)", r"%\s*S\.M\.S\."),
        "N_Nitrico (mg/kg)": nutrient_value(0),
        "Fosforo (mg/kg)": nutrient_value(1),
        "Potasio (mg/kg)": nutrient_value(2),
        "Calcio (mg/kg)": nutrient_value(3),
        "Magnesio (mg/kg)": nutrient_value(4),
        "Sodio (mg/kg)": nutrient_value(5),
    }

    return row


def parse_agrolab(pdf_path: Path, text: str) -> dict[str, object] | None:
    municipality = "HUESCA" if "HUESCA" in normalize_text(text) else ""
    crop = "Vinedo"
    report_date_match = re.search(r"Fecha Recepcion\s+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    sample_id_match = re.search(r"Registro N[oº]:\s*([A-Z0-9-]+)", text)
    first_page = text.split("\f", 1)[0]

    def agrolab_value(label: str, unit_hint: str, span: int = 220) -> float | None:
        idx = first_page.find(label.upper())
        if idx == -1:
            return None
        window = first_page[idx:idx + span]
        match = re.search(rf"([-+]?\d+(?:,\d+)?)\s*{unit_hint}", window, re.IGNORECASE)
        return parse_number(match.group(1)) if match else None

    return {
        "source_file": pdf_path.relative_to(ROOT).as_posix(),
        "source_group": source_group_for(pdf_path),
        "sample_id": sample_id_match.group(1) if sample_id_match else pdf_path.stem,
        "year": 2023,
        "report_date": report_date_match.group(1) if report_date_match else "",
        "municipio": municipality,
        "cultivo": pretty_label(crop),
        "descripcion": "ZONA SOMONTANO (HUESCA)",
        "pH": agrolab_value("PH AGUA", r"UNID\.\s*PH"),
        "MO (%)": agrolab_value("MATERIA ORGANICA", r"G/100G"),
        "CE (dS/m)": None,
        "Caliza (%)": agrolab_value("CALCICO EQUIVALENTE", r"G/100G"),
        "Nitrogeno (%)": agrolab_value("NITROGENO (N)", r"G/100G"),
        "N_Nitrico (mg/kg)": None,
        "Fosforo (mg/kg)": agrolab_value("FOSFORO (P) ASIMILABLE", r"MG/KG"),
        "Potasio (mg/kg)": agrolab_value("POTASIO (K) ASIMILABLE", r"MG/KG"),
        "Calcio (mg/kg)": agrolab_value("CALCIO (CA) ASIMILABLE", r"MG/KG"),
        "Magnesio (mg/kg)": agrolab_value("MAGNESIO (MG) ASIMILABLE", r"MG/KG"),
        "Sodio (mg/kg)": agrolab_value("SODIO (NA) ASIMILABLE", r"MG/KG"),
    }


def parse_pdf(pdf_path: Path) -> dict[str, object] | None:
    text = read_pdf_text(pdf_path)
    normalized = normalize_text(text)
    first_page = normalized.split("\f", 1)[0]

    if "SUELOS" in normalized and "AGROLAB" in normalized:
        return parse_agrolab(pdf_path, normalized)

    description_window = ""
    for marker in ["DESCRIPCION DE LA MUESTRA", "DESCRIPCIO DE LA MOSTRA"]:
        index = first_page.find(marker)
        if index != -1:
            description_window = first_page[index:index + 220]
            break

    is_soil_report = bool(re.search(r"\bSUELO\b|\bSOIL\b|\bSOL\b", description_window)) or bool(
        re.search(r"\bSUELOS\b", first_page)
    )
    if is_soil_report:
        return parse_eurofins(pdf_path, normalized)

    return None


def collect_rows() -> list[dict[str, object]]:
    pdfs = sorted({*RAW_DIR.rglob("*.pdf"), *RAW_DIR.rglob("*.PDF")})
    rows = []
    for pdf_path in pdfs:
        try:
            row = parse_pdf(pdf_path)
        except subprocess.CalledProcessError as exc:
            print(f"Skipping unreadable PDF: {pdf_path} ({exc})", file=sys.stderr)
            continue
        if row:
            rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        municipality = str(row.get("municipio", "")).strip()
        if municipality:
            grouped[municipality].append(row)

    summary_rows = []
    for municipality, items in sorted(grouped.items()):
        centroid = MUNICIPALITY_CENTROIDS.get(municipality)
        summary = {
            "Municipio": municipality.title(),
            "Latitud": centroid[0] if centroid else "",
            "Longitud": centroid[1] if centroid else "",
            "N_muestras": len(items),
        }
        for field in NUMERIC_FIELDS:
            values = [row[field] for row in items if isinstance(row.get(field), (int, float))]
            summary[field] = round(statistics.median(values), 2) if values else ""
        summary_rows.append(summary)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_coordinate(summary_rows: list[dict[str, object]]) -> tuple[float, float]:
    valid = [
        (float(row["Latitud"]), float(row["Longitud"]))
        for row in summary_rows
        if row["Latitud"] != "" and row["Longitud"] != ""
    ]
    if not valid:
        return 41.6, 0.62
    lat = sum(item[0] for item in valid) / len(valid)
    lon = sum(item[1] for item in valid) / len(valid)
    return lat, lon


def ph_profile(ph_value: object) -> dict[str, object]:
    if ph_value == "":
        return {
            "label": "Unavailable",
            "effect": "No pH-based interpretation available.",
            "color": "#6b7280",
            "score": 0.25,
        }

    ph = float(ph_value)
    if ph < 5.5:
        return {
            "label": "Strongly acidic",
            "effect": "Risk of aluminum and manganese toxicity. Phosphorus becomes less available.",
            "color": "#9b2226",
            "score": 0.2,
        }
    if ph < 6.0:
        return {
            "label": "Moderately acidic",
            "effect": "Suitable for acid-loving crops such as potatoes, blueberries, and coffee.",
            "color": "#ca6702",
            "score": 0.55,
        }
    if ph <= 7.0:
        return {
            "label": "Optimal / neutral",
            "effect": "Best overall nutrient availability with strong microbial activity.",
            "color": "#2a9d8f",
            "score": 1.0,
        }
    if ph <= 8.5:
        return {
            "label": "Slightly alkaline",
            "effect": "Iron and zinc availability starts to decline, increasing chlorosis risk.",
            "color": "#457b9d",
            "score": 0.6,
        }
    return {
        "label": "Strongly alkaline",
        "effect": "Typical of saline or sodic soils. Many micronutrients become difficult for plants to access.",
        "color": "#6a040f",
        "score": 0.2,
    }


def build_popup_html(row: dict[str, object]) -> str:
    ph_info = ph_profile(row.get("pH", ""))
    lines = [
        f"<strong>{row['Municipio']}</strong>",
        f"Aggregated samples: {row['N_muestras']}",
        f"pH class: {ph_info['label']}",
        f"pH effect: {ph_info['effect']}",
    ]
    for field in [
        "pH",
        "MO (%)",
        "CE (dS/m)",
        "Caliza (%)",
        "Nitrogeno (%)",
        "N_Nitrico (mg/kg)",
        "Fosforo (mg/kg)",
        "Potasio (mg/kg)",
        "Calcio (mg/kg)",
        "Magnesio (mg/kg)",
        "Sodio (mg/kg)",
    ]:
        value = row.get(field, "")
        if value != "":
            lines.append(f"{field}: {value}")
    return "<br>".join(lines)


def marker_color(ph_value: object) -> str:
    return str(ph_profile(ph_value)["color"])


def generate_map(summary_rows: list[dict[str, object]]) -> None:
    try:
        import folium
        from folium.plugins import HeatMap
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Folium is not installed. Create a virtualenv and install folium before running this script."
        ) from exc

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    center_lat, center_lon = mean_coordinate(summary_rows)
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None, control_scale=True)

    folium.TileLayer(
        "CartoDB positron",
        name="Base",
        show=True,
    ).add_to(fmap)
    folium.TileLayer(
        "CartoDB Voyager",
        name="Base alternativa",
        show=False,
    ).add_to(fmap)

    heat_layer = folium.FeatureGroup(name="Heatmap", show=False)
    marker_layer = folium.FeatureGroup(name="Muestras agregadas", show=True)

    heat_points = []

    for row in summary_rows:
        lat = row.get("Latitud")
        lon = row.get("Longitud")
        if lat == "" or lon == "":
            continue
        lat_f = float(lat)
        lon_f = float(lon)
        samples = max(int(row["N_muestras"]), 1)
        ph_info = ph_profile(row.get("pH", ""))
        radius = 7 + 3.5 * math.log2(samples)
        heat_points.append([lat_f, lon_f, samples * float(ph_info["score"])])

        folium.CircleMarker(
            location=[lat_f, lon_f],
            radius=radius,
            color=str(ph_info["color"]),
            fill=True,
            weight=2,
            fill_color=str(ph_info["color"]),
            fill_opacity=0.82,
            popup=folium.Popup(build_popup_html(row), max_width=320),
            tooltip=f"{row['Municipio']} ({row['N_muestras']} muestras)",
        ).add_to(marker_layer)

    if heat_points:
        HeatMap(
            heat_points,
            min_opacity=0.28,
            max_zoom=11,
            radius=34,
            blur=26,
            gradient={
                0.2: "#9b2226",
                0.4: "#ca6702",
                0.65: "#457b9d",
                0.85: "#2a9d8f",
                1.0: "#2a9d8f",
            },
        ).add_to(heat_layer)

    marker_layer.add_to(fmap)
    heat_layer.add_to(fmap)
    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    legend = """
    <div style="position: fixed; top: 18px; left: 18px; z-index: 1000;
                background: rgba(255,255,255,0.94); border: 1px solid #d6d3d1;
                padding: 14px 16px; font-size: 13px; line-height: 1.45;
                border-radius: 10px; box-shadow: 0 12px 24px rgba(0,0,0,0.10);">
      <div style="font-size: 15px; font-weight: 700; margin-bottom: 4px;">Soil Characteristics Overview</div>
      <div style="color: #57534e; margin-bottom: 8px;">Municipality-level aggregation with pH suitability heatmap</div>
      <span style="color: #9b2226;">●</span> pH &lt; 5.5: Strongly acidic<br>
      <span style="color: #ca6702;">●</span> pH 5.5 to 6.0: Moderately acidic<br>
      <span style="color: #2a9d8f;">●</span> pH 6.0 to 7.0: Optimal / neutral<br>
      <span style="color: #457b9d;">●</span> pH 7.1 to 8.5: Slightly alkaline<br>
      <span style="color: #6a040f;">●</span> pH &gt; 8.5: Strongly alkaline<br>
      <span style="color: #6b7280;">●</span> pH unavailable
    </div>
    <div style="position: fixed; bottom: 24px; left: 18px; z-index: 1000;
                background: rgba(255,255,255,0.92); border: 1px solid #d6d3d1;
                padding: 10px 12px; font-size: 12px; line-height: 1.4;
                border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.08);">
      Heatmap intensity = sample count weighted by pH suitability for general soil use
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))
    fmap.save(MAP_HTML)


def main() -> None:
    rows = collect_rows()
    sample_fieldnames = [
        "source_file",
        "source_group",
        "sample_id",
        "year",
        "report_date",
        "municipio",
        "cultivo",
        "descripcion",
        *NUMERIC_FIELDS,
    ]
    write_csv(SAMPLE_CSV, rows, sample_fieldnames)

    summary_rows = aggregate_rows(rows)
    summary_fieldnames = ["Municipio", "Latitud", "Longitud", "N_muestras", *NUMERIC_FIELDS]
    write_csv(SUMMARY_CSV, summary_rows, summary_fieldnames)
    generate_map(summary_rows)

    print(f"Wrote {len(rows)} sample rows to {SAMPLE_CSV}")
    print(f"Wrote {len(summary_rows)} municipality summaries to {SUMMARY_CSV}")
    print(f"Wrote privacy-preserving map to {MAP_HTML}")


if __name__ == "__main__":
    main()
