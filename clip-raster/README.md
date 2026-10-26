# Clip Raster — Sentinel-2 Vegetation Indices

This algorithm takes the geometry of a parcel from its input dataset, finds the most recent Sentinel-2 L2A scene in which the parcel is clear of clouds, reads only the pixels around the parcel, and computes a set of vegetation and water indices. The result is an interactive HTML report plus a small JSON summary.

It is designed to run on Ocean Node Compute-to-Data together with the **SIGPAC (Spanish LPIS): Land Parcel Codes** dataset. The consumer passes the SIGPAC codes (province, municipality, polygon, parcel, precinct) as dataset parameters. The node appends them to the dataset URL and saves the returned GeoJSON in `/data/inputs/`.

## How It Works

1. Reads every input file as GeoJSON (FeatureCollection, Feature or geometry), or as SIGPAC `format=json` records with a `wkt` geometry. The polygons are merged into one parcel.
2. Searches the public [Earth Search](https://earth-search.aws.element84.com/v1) STAC catalogue (`sentinel-2-c1-l2a`, the ESA Sentinel-2 L2A archive on AWS Open Data) for the scenes of the last `lookbackDays` days that intersect the parcel, newest first. When neighbouring tiles overlap, it keeps the tile that covers the whole parcel.
3. For each candidate, it reads the Scene Classification Layer (SCL) over the parcel. A scene qualifies when the share of pixels that are no data, saturated, cloud shadow, cloud, cirrus or snow is at most `maxCloudCover`. The newest qualifying scene is used. If none qualifies, the clearest one is used and the report says so.
4. Reads the needed bands of that scene as windowed reads of the Cloud-Optimized GeoTIFFs. Only a few KB to MB are downloaded, so no credentials and no full product are needed. The bands are resampled to a common 10 m grid and converted to reflectance (`DN × scale + offset`). Unusable pixels are masked.
5. Computes the indices and their statistics. It then writes `report.html` and `indices.json`.

---

## Computed Indices

| Index | Description | Bands |
|---|---|---|
| `ndvi` | Normalized Difference Vegetation Index | NIR (B08), Red (B04) |
| `gndvi` | Green Normalized Difference Vegetation Index | NIR (B08), Green (B03) |
| `ndwi` | Normalized Difference Water Index, green–SWIR variant (also known as MNDWI) | Green (B03), SWIR-1 (B11) |
| `ndmi` | Normalized Difference Moisture Index | Narrow NIR (B8A), SWIR-1 (B11) |

---

## Inputs

| Input | Description |
|---|---|
| `/data/inputs/*` | GeoJSON of the parcel, e.g. the SIGPAC dataset's response (`recinfo`). |
| `algoCustomData.json` | Algorithm parameters, see below. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `lookbackDays` | How many days back to search for scenes (1–365). | `60` |
| `maxCloudCover` | Maximum % of unusable pixels inside the parcel (0–100). | `20` |

---

## Output

| File | Description |
|---|---|
| `report.html` | Self-contained report, in Spanish. It has a Leaflet map with one layer per index plus true colour over orthophoto, per-index statistics and histograms, scene details and the SIGPAC attributes. Viewing it needs internet access for Leaflet and the base maps. |
| `indices.json` | Scene, parcel and statistics per index: mean, median, P10, P90, std, min, max and valid pixels. |

---

## Requirements on Ocean Node

- The compute environment must allow network access (`enableNetwork: true`), to reach Earth Search and the AWS COGs.
- A job with the free environment (1 CPU, 1 GB RAM, ≤ 600 s) takes about 15 s.
- `oceanprotocol-job-details` ≥ 0.5.2 is required for the Ocean Node `/data` layout, which has no `DIDS` variable and an empty `ddos/`.

---

## Running Locally

`_data/inputs/` mimics what Ocean Node provisions: `recinfo` is the SIGPAC GeoJSON of precinct 25:168:0:0:10:6:1.

With uv:

```bash
cd algorithm
uv run ocean-execute --base-dir ../_data src.algorithm
uv run pytest
```

With Docker:

```bash
docker compose up --build
```

The outputs are written to `_data/outputs/`. To try another parcel, replace `_data/inputs/recinfo`:

```bash
curl -o _data/inputs/recinfo 'https://sigpac.agrospai.udl.cat/query/recinfo?province=25&municipality=23&aggregation=0&zone=0&polygon=3&parcel=132&precint=1&format=geojson'
```

---

## Build and Push

Ocean Node pulls the image by digest (`image@checksum`), so the published algorithm needs the digest of the pushed image:

```bash
docker buildx build --platform linux/amd64 -t registry.agrospai.udl.cat/library/clip-raster:{VERSION} . --push
docker buildx imagetools inspect registry.agrospai.udl.cat/library/clip-raster:{VERSION}  # Digest: sha256:…
```
