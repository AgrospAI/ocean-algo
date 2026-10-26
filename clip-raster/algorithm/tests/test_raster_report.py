import json
import re

import numpy as np

from src.geometry import Parcel
from src.indices import compute_indices
from src.raster import RESOLUTION, scene_grid, unusable_share, usable
from src.report import render_html, summary
from src.scenes import Scene

from .conftest import make_item


def test_scene_grid_snaps_to_the_tile_and_masks_the_parcel(parcel):
    grid = scene_grid(make_item("a", "2026-10-21T10:00:00Z"), parcel)
    assert grid.crs == "EPSG:32631"
    assert (grid.transform.c - 300000) % RESOLUTION == 0
    assert (4700040 - grid.transform.f) % RESOLUTION == 0
    # ~170 m x ~165 m parcel: about 280 pixels of 10 m inside, plus a margin around it.
    assert 250 < grid.inside.sum() < 310
    assert not grid.inside[0].any() and not grid.inside[:, 0].any()


def test_scl_classes_decide_usable_pixels(parcel):
    grid = scene_grid(make_item("a", "2026-10-21T10:00:00Z"), parcel)
    scl = np.full((grid.height, grid.width), 4, dtype="uint8")  # vegetation
    scl[: grid.height // 2] = 9  # high-probability cloud
    share = unusable_share(scl, grid)
    assert 0.3 < share < 0.7
    assert (usable(scl, grid) == (grid.inside & (scl == 4))).all()


def test_summary_and_report(parcel):
    item = make_item("S2B_TEST", "2026-10-21T10:59:45Z")
    item["properties"].update({"grid:code": "MGRS-31TCG", "eo:cloud_cover": 5.4})
    grid = scene_grid(item, parcel)
    shape = (grid.height, grid.width)
    mask = ~grid.inside
    bands = {
        name: np.ma.masked_array(np.full(shape, value, dtype="float32"), mask=mask)
        for name, value in {"green": 0.08, "red": 0.05, "nir": 0.35, "nir08": 0.33, "swir16": 0.2}.items()
    }
    indices = compute_indices(bands)
    sigpac = {"provincia": 25, "municipio": 23, "agregado": 0, "zona": 0, "poligono": 3, "parcela": 132, "recinto": 1}
    result = summary(Parcel(parcel, [sigpac]), grid, Scene(item, 1.0, 0.0, True), indices, 60, 20)

    assert result["parcel"]["sigpac"] == ["25:23:0:0:3:132:1"]
    assert 2.0 < result["parcel"]["areaHa"] < 3.2
    assert result["scene"]["tile"] == "MGRS-31TCG"
    assert abs(result["indices"]["ndvi"]["mean"] - 0.75) < 1e-3
    assert result["indices"]["ndvi"]["validShare"] == 1.0
    json.dumps(result, allow_nan=False)

    rgb = np.full((3, *shape), 120, dtype="uint8")
    html = render_html(result, Parcel(parcel, [sigpac]), grid, indices, rgb)
    assert "Recinto SIGPAC 25:23:0:0:3:132:1" in html
    data = json.loads(re.search(r"const DATA = (.*?);\n", html).group(1))
    assert [layer["id"] for layer in data["layers"]] == ["rgb", "ndvi", "gndvi", "ndwi", "ndmi"]
    assert all(layer["image"].startswith("data:image/png;base64,") for layer in data["layers"])
    (south, west), (north, east) = data["layers"][1]["bounds"]
    minx, miny, maxx, maxy = parcel.bounds
    assert west <= minx and south <= miny and east >= maxx and north >= maxy
