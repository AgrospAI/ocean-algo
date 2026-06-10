import logging
from pathlib import Path
from ocean_runner import Algorithm

logger = logging.getLogger(__name__)

def require(band: Path | None) -> Path:
    if not band:
        logger.error('Missing band path')
        raise Algorithm.Error(f'Missing band path')
    return band


def format_filename(filename, length=40):
    if len(filename) > length:
        return filename[:length - 3] + '...'
    else:
        return filename.ljust(length)


def get_band_path(band: str, res: str = '10m') -> Path | None:
    prefix = Path('/tmp/downloads/eodata/Sentinel-2/GRANULE')

    band_path = next(prefix.rglob(f'*{band}_{res}.jp2'), None)

    return band_path