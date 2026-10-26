import json

import pytest
from ocean_runner import Algorithm

from src.geometry import load_parcel

FEATURE = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[0.5, 41.3], [0.51, 41.3], [0.51, 41.31], [0.5, 41.31], [0.5, 41.3]]],
    },
    "properties": {"provincia": 25, "recinto": 1},
}


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_reads_a_sigpac_feature_collection(tmp_path):
    parcel = load_parcel([write(tmp_path, "recinfo", {"type": "FeatureCollection", "features": [FEATURE]})])
    assert parcel.geometry.bounds == (0.5, 41.3, 0.51, 41.31)
    assert parcel.properties == [{"provincia": 25, "recinto": 1}]


def test_merges_features_and_files(tmp_path):
    other = json.loads(json.dumps(FEATURE))
    other["geometry"]["coordinates"][0] = [[x + 0.01, y] for x, y in other["geometry"]["coordinates"][0]]
    parcel = load_parcel([write(tmp_path, "a", FEATURE), write(tmp_path, "b", other["geometry"])])
    assert parcel.geometry.bounds == (0.5, 41.3, 0.52, 41.31)
    assert len(parcel.properties) == 2


def test_reads_sigpac_json_records_with_wkt(tmp_path):
    record = {"provincia": 25, "srid": 4258, "wkt": "POLYGON((0.5 41.3,0.51 41.3,0.51 41.31,0.5 41.3))"}
    parcel = load_parcel([write(tmp_path, "recinfo", [record])])
    assert parcel.properties == [{"provincia": 25}]
    assert parcel.geometry.area > 0


def test_explains_an_empty_sigpac_answer(tmp_path):
    with pytest.raises(Algorithm.Error, match="No SIGPAC parcel matches"):
        load_parcel([write(tmp_path, "recinfo", {"type": "FeatureCollection", "features": [], "crs": ""})])


def test_rejects_inputs_that_are_not_geojson(tmp_path):
    with pytest.raises(Algorithm.Error, match="neither GeoJSON"):
        load_parcel([write(tmp_path, "data.json", {"rows": 3})])
    bad = tmp_path / "data.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(Algorithm.Error, match="not JSON"):
        load_parcel([bad])


def test_requires_an_input():
    with pytest.raises(Algorithm.Error, match="No input files"):
        load_parcel([])
