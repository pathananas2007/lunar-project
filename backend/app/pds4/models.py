from dataclasses import dataclass
from typing import Optional


@dataclass
class PDS4Metadata:
    product_id: str
    instrument: str  # TMC-2 | OHRC | IIRS | DFSAR
    bit_depth: int
    width: int
    height: int
    scaling_factor: float = 1.0
    offset: float = 0.0
    corner_lat_lon: Optional[list] = None  # [(lat,lon) x4]
    solar_zenith: Optional[float] = None
    resolution_mpp: Optional[float] = None  # meters per pixel
    bands: Optional[int] = None  # IIRS hyperspectral: number of bands
