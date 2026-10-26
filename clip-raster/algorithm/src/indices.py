from dataclasses import dataclass

import numpy as np
from numpy.ma import MaskedArray


def normalized_difference(a: MaskedArray, b: MaskedArray) -> MaskedArray:
    """(a - b) / (a + b), masked where the sum is zero."""
    total = a + b
    total = np.ma.masked_where(np.ma.getdata(total) == 0, total)
    return np.ma.clip((a - b) / total, -1, 1)


@dataclass(frozen=True)
class Index:
    name: str
    label: str
    a: str
    """Band of the minuend."""
    b: str
    """Band of the subtrahend."""
    formula: str
    colormap: str
    vmin: float
    vmax: float
    meaning: str


INDEXES = (
    Index(
        "ndvi",
        "NDVI",
        "nir",
        "red",
        "(B08 − B04) / (B08 + B04)",
        "RdYlGn",
        -0.2,
        1,
        "Vigor y densidad de la vegetación. < 0.2: suelo desnudo o agua; 0.2–0.5: "
        "vegetación escasa o en estrés; > 0.5: vegetación densa y sana.",
    ),
    Index(
        "gndvi",
        "GNDVI",
        "nir",
        "green",
        "(B08 − B03) / (B08 + B03)",
        "RdYlGn",
        -0.2,
        1,
        "Contenido de clorofila. Más sensible que el NDVI en cultivos densos, donde el "
        "NDVI se satura.",
    ),
    Index(
        "ndwi",
        "NDWI",
        "green",
        "swir16",
        "(B03 − B11) / (B03 + B11)",
        "RdBu",
        -1,
        1,
        "Agua superficial (variante verde–SWIR, también llamada MNDWI). > 0: lámina de "
        "agua o suelo encharcado; < 0: suelo o vegetación.",
    ),
    Index(
        "ndmi",
        "NDMI",
        "nir08",
        "swir16",
        "(B8A − B11) / (B8A + B11)",
        "BrBG",
        -1,
        1,
        "Humedad de la vegetación. < 0: estrés hídrico o suelo seco; > 0.2: vegetación "
        "con buen contenido de agua.",
    ),
)


def compute_indices(bands: dict[str, MaskedArray]) -> dict[str, MaskedArray]:
    return {
        index.name: normalized_difference(bands[index.a], bands[index.b]) for index in INDEXES
    }
