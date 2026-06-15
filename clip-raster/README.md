# Clip Raster — Sentinel-2 Vegetation Indices

This algorithm takes a Spanish cadastral reference, resolves its parcel geometry, downloads the most recent Sentinel-2 L2A product covering it, clips the relevant bands to the parcel boundary, and computes a set of vegetation and water indices. The result is a single image showing each index over the parcel.

## How It Works

Given a cadastral reference (`refcat`), the algorithm:

1. Resolves the parcel geometry from the Spanish Catastro [INSPIRE WFS](http://ovc.catastro.meh.es/INSPIRE/wfsCP.aspx) (`GetParcel` stored query).
2. Queries the Copernicus Data Space [OData API](https://catalogue.dataspace.copernicus.eu/odata/v1/Products) for the latest `SENTINEL-2` `S2MSI2A` product intersecting the parcel's bounding box.
3. Downloads the product's bands from the Copernicus `eodata` S3 store using temporary credentials.
4. Clips each required band to the parcel geometry (reprojecting the geometry to the band's UTM CRS).
5. Computes the indices listed below from the clipped bands.
6. Renders all indices as a 2×2 grid and saves it to `/data/outputs/indices.png`.

---

## Computed Indices

| Index | Description | Bands |
|---|---|---|
| `ndvi` | Normalized Difference Vegetation Index | Red (B04), NIR (B08) |
| `gndvi` | Green Normalized Difference Vegetation Index | Green (B03), NIR (B08) |
| `ndwi` | Normalized Difference Water Index | Green (B03 20 m), SWIR-1 (B11) |
| `ndmi` | Normalized Difference Moisture Index | Narrow NIR (B8A), SWIR-1 (B11) |

---

## Input Parameters

Passed as `algoCustomData`:

| Parameter | Description | Example |
|---|---|---|
| `refcat` | Cadastral reference (alphanumeric) of the target parcel | `"25168A010000060000IZ"` |

---

## Output

| File | Description |
|---|---|
| `indices.png` | 2×2 grid plotting NDVI, GNDVI, NDWI and NDMI clipped to the parcel |

---

## Running Locally with Docker

### Prerequisites

- Docker and Docker Compose installed.
- Input parameter set in `_data/inputs/algoCustomData.json`.
- Copernicus Data Space credentials available at `_data/transformations/algorithm` as `username` and `password` keys (used to obtain an access token and temporary S3 credentials for the download).

> [!IMPORTANT]
> Both the geometry lookup (Catastro INSPIRE WFS) and the product download (Copernicus Data Space) require an active internet connection. The product query only returns scenes acquired after the configured start date — if no Sentinel-2 product matches the parcel, the algorithm fails with an error.

### Run

```bash
docker compose up --build
```

The output image will be written to `_data/outputs/indices.png`.

### Rebuild after code changes

The `docker-compose.yaml` mounts `algorithm/src/` into the container, so Python source changes are picked up without rebuilding the image:

```bash
docker compose up
```

To force a full image rebuild (e.g. after changing `pyproject.toml` or `Dockerfile`):

```bash
docker compose up --build
```

---

## Build and Push

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t {ALGORITHM_TAG}:{ALGORITHM_VERSION} . --push
```
