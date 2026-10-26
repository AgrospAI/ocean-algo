import numpy as np

from src.indices import INDEXES, compute_indices, normalized_difference


def test_normalized_difference_masks_zero_sums_and_keeps_masks():
    a = np.ma.masked_array([0.4, 0.0, 0.3], mask=[False, False, True])
    b = np.ma.masked_array([0.1, 0.0, 0.1], mask=[False, False, False])
    out = normalized_difference(a, b)
    assert abs(out[0] - 0.6) < 1e-6
    assert out.mask.tolist() == [False, True, True]


def test_compute_indices_uses_the_documented_bands():
    ones = np.ma.masked_array(np.full((2, 2), 0.1, dtype="float32"))
    bands = {name: ones for name in ("green", "red", "nir08", "swir16")}
    bands["nir"] = ones * 3
    out = compute_indices(bands)
    assert set(out) == {index.name for index in INDEXES}
    assert np.allclose(out["ndvi"], 0.5)
    assert np.allclose(out["gndvi"], 0.5)
    assert np.allclose(out["ndwi"], 0)
    assert np.allclose(out["ndmi"], 0)
