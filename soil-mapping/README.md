# Soil Characteristics Mapping Guide

This repository contains the soil mapping algorithm (`algorithm.py`) that processes laboratory soil analysis PDF reports, geocodes each sample to its cadastral parcel, and generates an interactive HTML map visualising key soil parameters across the field fleet.

## 1. How It Works

The algorithm runs in three sequential phases:

1. **PDF extraction (LLM-based)** — unzips the input archive and, for each PDF, sends both rendered page images and a layout-aware markdown transcription (extracted with `docling`) to a vision-language model. The model:
   - Classifies the document as `soil`, `water`, `foliar` or `other`. Only `soil` reports are kept; water and foliar analyses are skipped automatically.
   - Extracts cadastral identifiers (Polígon / Parcella / Terme Municipal / INE code) and soil parameters from layout-agnostic content.
   - Enforces strict unit/method rules: **Nitrogen** is kept only when reported as *Nítrico* in mg/kg; **Phosphorus** only when extracted by the **Olsen** method (or expressed in mg/kg); **Potassium** only in mg/kg. Values from acid-extract methods, foliar ppm, or % s.m.s. are discarded.
2. **Geocoding** — resolves each sample to GPS coordinates via the Spain Catastro `Consulta_CPMRC` API using the cadastral reference (province + municipality + polygon + parcel). For records missing an explicit INE code, the municipality name is resolved against the Catastro municipality registry for the target province.
3. **Map generation** — computes IDW (Inverse Distance Weighting) interpolation rasters per soil parameter, both for the full aggregate and for each individual year present in the dataset. Renders them as a Folium interactive HTML map with a WRB soil-type WMS overlay, parameter + year selectors, optimal-range indicators, and a fleet statistics panel including a per-year sample breakdown.

### Supported Laboratory Report Formats

The LLM reads layout directly from page images, so format support is driven by content rather than rigid template matching. The pipeline has been validated against the following formats:

| Format | Description | Geocodable |
|---|---|---|
| Eurofins XK | Eurofins soil bulletins identified by the `AR-…-XK-…` reference scheme | Yes (when Polígon/Parcella present) |
| Eurofins BUTLLETÍ | Catalan `BUTLLETÍ D'ANÀLISIS` soil bulletins | Yes |

Other soil-analysis layouts may also be processed successfully as long as they contain the required cadastral fields and at least one of the supported soil parameters.

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
  │   └── field_report_002.pdf    ← Eurofins BUTLLETÍ format
  └── archived/
      └── field_report_2023.pdf
```

PDFs that are not recognised as soil analysis reports (e.g. water or foliar analysis) are automatically skipped.

> [!IMPORTANT]
> Both the LLM extraction step (vision-language model hosted on OpenWebUI) and the geocoding step (Spain Catastro API) require an active internet connection. PDFs without a Polígon/Parcella cadastral reference will still have their soil parameters extracted into `records.json` but **will not appear on the map**, as their geographic position cannot be determined.

---

## 3. Output

One file is written to `/data/outputs/`:

| File | Description |
|---|---|
| `soil-characteristics-map.html` | Self-contained interactive map (no server needed — open in any browser) |

### Map Features

- **Parameter selector** — switch between pH, Organic Matter, Electrical Conductivity, N-NO₃, and Phosphorus IDW rasters.
- **Year selector** — view the full aggregate (default) or any single year present in the dataset. Per-year rasters share the aggregate's colour scale so cross-year comparisons remain visually honest.
- **Low-sample warning** — when a selected year has fewer than ~15 samples, the legend shows an advisory note so users do not over-interpret sparse interpolations. Areas not sampled in the chosen year stay transparent (per-year coverage hull).
- **WRB Soil Type overlay** — ISRIC SoilGrids WMS layer with adjustable opacity.
- **Fleet statistics panel** — pH distribution breakdown, samples-per-year histogram, and mean ± std for each parameter across all samples.
- **Optimal range indicators** — a highlighted zone on the legend gradient marks the agronomically optimal range for each parameter.

> [!NOTE]
> No individual sample markers are rendered on the map. The IDW interpolation is the only visualisation, which preserves the geographic privacy of contributing growers — exact parcel locations are never exposed.

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
