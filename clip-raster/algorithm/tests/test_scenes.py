from datetime import UTC, datetime

import pytest
import requests
from ocean_runner import Algorithm
from shapely.geometry import box

from src.scenes import candidates, search_items, select_scene

from .conftest import make_item


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.body


def test_search_asks_for_the_newest_scenes_in_the_window(parcel):
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, body=json)
        return FakeResponse({"features": [make_item("a", "2026-10-21T10:00:00Z")]})

    items = search_items(parcel, 30, now=datetime(2026, 10, 26, tzinfo=UTC), post=post)
    assert [i["id"] for i in items] == ["a"]
    assert sent["body"]["collections"] == ["sentinel-2-c1-l2a"]
    assert sent["body"]["datetime"] == "2026-09-26T00:00:00Z/2026-10-26T00:00:00Z"
    assert sent["body"]["sortby"] == [{"field": "properties.datetime", "direction": "desc"}]
    assert sent["body"]["intersects"]["type"] == "Polygon"


def test_search_fails_clearly(parcel):
    with pytest.raises(Algorithm.Error, match="No Sentinel-2 scene covers"):
        search_items(parcel, 10, post=lambda *a, **k: FakeResponse({"features": []}))
    with pytest.raises(Algorithm.Error, match="catalogue search failed"):
        search_items(parcel, 10, post=lambda *a, **k: FakeResponse({}, status=502))


def test_candidates_keep_one_tile_per_acquisition_and_prefer_full_coverage(parcel):
    half = box(parcel.bounds[0] - 1, parcel.bounds[1] - 1, parcel.centroid.x, parcel.bounds[3] + 1)
    items = [
        make_item("new-partial", "2026-10-24T10:00:00Z", footprint=half),
        make_item("old-tile-a", "2026-10-21T10:00:00Z", footprint=half),
        make_item("old-tile-b", "2026-10-21T10:00:01Z"),
        make_item("older", "2026-10-18T10:00:00Z", platform="sentinel-2a"),
    ]
    ranked = candidates(items, parcel)
    assert [(i["id"], round(c, 2)) for i, c in ranked] == [("old-tile-b", 1.0), ("older", 1.0)]


def test_candidates_fall_back_to_partial_scenes(parcel):
    half = box(parcel.bounds[0] - 1, parcel.bounds[1] - 1, parcel.centroid.x, parcel.bounds[3] + 1)
    ranked = candidates([make_item("p", "2026-10-24T10:00:00Z", footprint=half)], parcel)
    assert ranked[0][0]["id"] == "p"
    assert 0.4 < ranked[0][1] < 0.6


def test_select_scene_takes_the_newest_clear_enough_scene(parcel):
    items = [
        make_item("cloudy", "2026-10-24T10:00:00Z"),
        make_item("clear", "2026-10-21T10:00:00Z", platform="sentinel-2c"),
        make_item("clearer", "2026-10-18T10:00:00Z", platform="sentinel-2a"),
    ]
    clouds = {"cloudy": 0.9, "clear": 0.15, "clearer": 0.0}
    scene = select_scene(items, parcel, 20, lambda item: clouds[item["id"]])
    assert (scene.id, scene.meets_threshold, scene.cloud_fraction) == ("clear", True, 0.15)


def test_select_scene_falls_back_to_the_clearest_and_skips_broken_assets(parcel):
    items = [
        make_item("broken", "2026-10-24T10:00:00Z"),
        make_item("a", "2026-10-21T10:00:00Z", platform="sentinel-2c"),
        make_item("b", "2026-10-18T10:00:00Z", platform="sentinel-2a"),
    ]

    def clouds(item):
        if item["id"] == "broken":
            raise OSError("HTTP 404")
        return {"a": 0.7, "b": 0.4}[item["id"]]

    scene = select_scene(items, parcel, 10, clouds)
    assert (scene.id, scene.meets_threshold) == ("b", False)

    with pytest.raises(Algorithm.Error, match="could be read"):
        select_scene(items[:1], parcel, 10, clouds)
