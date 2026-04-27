# Soil Characteristics Mapping Guide

This repository contains the soil mapping algorithm (`algorithm.py`) that processes laboratory soil analysis PDF reports, geocodes each sample to its cadastral parcel, and generates an interactive HTML map visualising key soil parameters across the field fleet.

## 1. How It Works

The algorithm runs in three sequential phases:

1. **PDF extraction** — unzips the input archive, detects the laboratory report format (Eurofins XK, Eurofins BUTLLETÍ, or AGROLAB), and parses soil parameters plus cadastral identifiers (Polígon/Parcella/Terme Municipal) from each PDF using `pdftotext`.
2. **Geocoding** — resolves each sample to GPS coordinates via the Spain Catastro `Consulta_CPMRC` API using the cadastral reference (province + municipality + polygon + parcel). Falls back to Nominatim (OpenStreetMap) for municipality-level coordinates when the exact parcel is not found.
3. **Map generation** — computes IDW (Inverse Distance Weighting) interpolation rasters for each soil parameter and renders them as a Folium interactive HTML map with a WRB soil-type WMS overlay, a parameter selector, and a fleet statistics panel.

### Supported Laboratory Report Formats

| Format | Detection criterion | Geocodable |
|---|---|---|
| Eurofins XK | Contains `"XK"` and soil sample keywords | Yes (when Polígon/Parcella present) |
| Eurofins BUTLLETÍ | Contains `"BUTLLETÍ D'ANÀLISIS"` | Yes |
| AGROLAB | Contains `"AGROLAB"` or `"Análisis de Tierras"` | No (no cadastral reference) |

### Extracted Soil Parameters

| Parameter | Key | Unit |
|---|---|---|
| pH | `pH` | — |
| Organic Matter | `MO` | % |
| Electrical Conductivity | `CE` | dS/m |
| Calcium Carbonate | `Caliza` | % |
| Nitric Nitrogen | `N_Nitrico` | mg/kg |
| Phosphorus | `Fosforo` | mg/kg |
| Potassium | `Potasio` | mg/kg |

---

## 2. Input Data Specifications

The algorithm expects a **ZIP archive** containing soil analysis PDFs from any of the supported laboratory formats listed above.

### Directory Structure

```
/data/inputs/
  └── <any-subdirectory>/
        └── <your-archive>.zip   ← one or more ZIP files containing PDFs
```

The algorithm recursively searches `inputs/` for `.zip` files and extracts all PDFs it finds inside. PDFs can be nested at any depth within the archive.

### ZIP Archive Example

```
my-soil-reports-2024.zip
  ├── 2024/
  │   ├── field_report_001.pdf    ← Eurofins XK format
  │   ├── field_report_002.pdf    ← Eurofins BUTLLETÍ format
  │   └── lab_result_agrolab.pdf  ← AGROLAB format
  └── archived/
      └── field_report_2023.pdf
```

PDFs that are not recognised as soil analysis reports (e.g. water or foliar analysis) are automatically skipped.

> [!IMPORTANT]
> The geocoding step queries the Spain Catastro API and requires an active internet connection. PDFs without a Polígon/Parcella cadastral reference (e.g. AGROLAB format) will still have their soil parameters extracted but **will not appear on the map**, as their geographic position cannot be determined.

---

## 3. Output

A single file is written to `/data/outputs/`:

| File | Description |
|---|---|
| `soil-characteristics-map.html` | Self-contained interactive map (no server needed — open in any browser) |

### Map Features

- **Parameter selector** — switch between pH, Organic Matter, Electrical Conductivity, N-NO₃, and Phosphorus IDW rasters.
- **WRB Soil Type overlay** — ISRIC SoilGrids WMS layer with adjustable opacity.
- **Fleet statistics panel** — pH distribution breakdown and mean ± std for each parameter across all samples.
- **Optimal range indicators** — a highlighted zone on the legend gradient marks the agronomically optimal range for each parameter.

---

## 4. Running Locally with Docker

### Prerequisites

- Docker and Docker Compose installed.
- Input ZIP archive(s) placed under `_data/inputs/`.

### Run

```bash
docker compose up --build
```

The output map will be written to `_data/outputs/soil-characteristics-map.html`.

### Rebuild after code changes

The `docker-compose.yaml` mounts `algorithm/src/` into the container, so Python source changes are picked up without rebuilding the image:

```bash
docker compose up
```

To force a full image rebuild (e.g. after changing `pyproject.toml` or `Dockerfile`):

```bash
docker compose up --build
```
