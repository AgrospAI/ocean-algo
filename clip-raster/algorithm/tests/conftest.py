from typing import Any

import pytest
from shapely.geometry import Polygon

# A ~2 ha parcel near Lleida (UTM zone 31N), in EPSG:4326.
PARCEL = Polygon(
    [(0.6200, 41.6100), (0.6220, 41.6100), (0.6220, 41.6115), (0.6200, 41.6115), (0.6200, 41.6100)]
)


def make_item(
    item_id: str,
    datetime: str,
    footprint: Polygon | None = None,
    platform: str = "sentinel-2b",
) -> dict[str, Any]:
    """A minimal Earth Search item in EPSG:32631 (tile origin of 31TCG)."""
    footprint = footprint or PARCEL.buffer(0.1)
    band = {"href": f"https://example.test/{item_id}/B.tif", "proj:transform": [10, 0, 300000, 0, -10, 4700040]}
    return {
        "id": item_id,
        "geometry": {"type": "Polygon", "coordinates": [list(footprint.exterior.coords)]},
        "properties": {"datetime": datetime, "platform": platform, "proj:epsg": 32631},
        "assets": {"red": band},
    }


@pytest.fixture
def parcel() -> Polygon:
    return PARCEL
