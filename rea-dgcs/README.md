# Algorithm implementation based on the OceanProtocol ecosystem

Build and push

```bash
$ docker buildx build --platform linux/amd64,linux/arm64 -t {ALGORITHM_TAG}:{ALGORITHM_VERSION} . --push 
```

## Algorithm details

This algorithm reads a REA (Spanish Register of Agricultural Holdings) dataset produced by the
SIEX/IACS system, extracts each field's (DGC) geometry via the SIGPAC API and generates an
interactive **Leaflet** HTML map (`parcel_map.html`) browsable by year and crop, together with
per-field GeoJSON files, a parcel log and per-parcel detail tables.

### Crop catalogue

Crop codes are shown with their names using the SIEX **product** catalogue bundled in the
algorithm image at `algorithm/src/cultivos.csv`, loaded at runtime by `algorithm.py`.

To update the crop names with the latest official catalogue, replace that file with the current
version downloaded from (keep the same filename and columns):

```
https://www3.sede.fega.gob.es/bdcsixpor/descarga-csv/PRODUCTOS/csv
```

The file is copied into the image as part of `algorithm/src`, so no code change is needed — just
replace the CSV and rebuild.