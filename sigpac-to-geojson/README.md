# SIGPAC to GeoJSON

This algorithm queries the Spanish [SIGPAC HubCloud OGC API](https://sigpac-hubcloud.es) for a specific cadastral precinct and returns its geometry and attributes as a GeoJSON file.

## How It Works

Given a set of cadastral identifiers (province, municipality, polygon, parcel, and precinct), the algorithm:

1. Builds a CQL2 filter from the input parameters.
2. Issues a single GET request to the SIGPAC `recintos` collection endpoint.
3. Saves the GeoJSON response to `/data/outputs/sigpac_data.json`.

---

## Input Parameters

Passed as `algoCustomData` (all values must be positive integers as strings):

| Parameter | Description | Example |
|---|---|---|
| `province` | INE province code | `"25"` |
| `municipality` | INE municipality code | `"168"` |
| `aggregation` | Aggregation code (`0` if none) | `"0"` |
| `zone` | Zone code (`0` if none) | `"0"` |
| `polygon` | Polygon number | `"10"` |
| `parcel` | Parcel number | `"6"` |
| `precint` | Precinct number within the parcel | `"1"` |

---

## Output

| File | Description |
|---|---|
| `sigpac_data.json` | GeoJSON FeatureCollection returned by SIGPAC for the requested precinct |

---

## Running Locally with Docker

### Prerequisites

- Docker and Docker Compose installed.
- Input parameters set in `_data/inputs/algoCustomData.json`.

### Run

```bash
docker compose up --build
```

The output will be written to `_data/outputs/sigpac_data.json`.

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
