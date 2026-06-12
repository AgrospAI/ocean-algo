from numpy.ma import MaskedArray

def ndvi(red: MaskedArray, infrared: MaskedArray) -> MaskedArray:
    out = (infrared - red) / (red + infrared)
    return out


def gndvi(green: MaskedArray, infrared: MaskedArray) -> MaskedArray:
    out = (infrared - green) / (green + infrared)
    return out


def ndwi(green: MaskedArray, swir_1: MaskedArray) -> MaskedArray:
    out = (green - swir_1) / (green + swir_1)
    return out


def ndmi(narrow_infrared: MaskedArray, swir_1: MaskedArray) -> MaskedArray:
    out = (narrow_infrared - swir_1) / (narrow_infrared + swir_1)
    return out
