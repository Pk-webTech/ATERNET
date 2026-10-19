"""
PV_Live regions (PES / DNO licence areas) and the weather sample points
used to represent each one.

PV_Live reports the *aggregate* generation of every PV system in a
region, so a single weather gridpoint is a noisy proxy for it. Each
region is represented by three points spread across its area; weather
from the points is averaged per model before the inter-model spread is
taken. Points are approximate population/PV-density centres, not exact
geographic centroids -- good enough at NWP grid resolution (~10-25 km).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Region:
    region_id: int
    name: str
    points: Tuple[Tuple[float, float], ...]   # (lat, lon)

    @property
    def latitude(self) -> float:
        return float(np.mean([p[0] for p in self.points]))

    @property
    def longitude(self) -> float:
        return float(np.mean([p[1] for p in self.points]))


REGIONS = {
    0: Region(0, "National (GB)", ((51.5, -0.5), (52.5, -1.5), (50.9, -3.2))),
    10: Region(10, "UKPN East", ((52.2, 0.1), (52.6, 1.2), (51.8, 0.6))),
    11: Region(11, "WPD East Midlands", ((52.95, -1.15), (52.6, -1.1), (53.2, -0.5))),
    12: Region(12, "UKPN London", ((51.5, -0.12), (51.45, -0.3), (51.55, 0.05))),
    13: Region(13, "SPEN Manweb (Merseyside & N Wales)", ((53.4, -2.95), (53.2, -3.6), (52.9, -3.0))),
    14: Region(14, "WPD West Midlands", ((52.48, -1.9), (52.2, -2.2), (52.7, -2.5))),
    15: Region(15, "NPG Northern", ((54.97, -1.6), (54.6, -1.3), (54.9, -2.2))),
    16: Region(16, "ENWL North West", ((53.48, -2.24), (53.8, -2.7), (54.4, -2.9))),
    17: Region(17, "SSE North Scotland", ((57.15, -2.1), (56.5, -3.4), (57.5, -4.2))),
    18: Region(18, "SPEN South Scotland", ((55.86, -4.25), (55.95, -3.2), (55.5, -3.0))),
    19: Region(19, "UKPN South East", ((51.2, 0.6), (51.0, 0.0), (51.3, 1.1))),
    20: Region(20, "SSE Southern", ((51.2, -1.3), (50.9, -1.4), (51.5, -1.0))),
    21: Region(21, "WPD South Wales", ((51.6, -3.4), (51.7, -4.6), (51.8, -3.0))),
    22: Region(22, "WPD South West", ((50.7, -3.5), (50.4, -4.6), (51.2, -2.6))),
    23: Region(23, "NPG Yorkshire", ((53.8, -1.55), (53.4, -1.45), (53.75, -0.4))),
}


def get_region(region_id: int) -> Region:
    if region_id not in REGIONS:
        raise KeyError(f"Unknown PV_Live region {region_id}. Known: {sorted(REGIONS)}.")
    return REGIONS[region_id]


def nearest_region(latitude: float, longitude: float) -> Region:
    """Closest region (by mean sample point) to a lat/lon. Excludes national."""
    candidates: List[Region] = [r for r in REGIONS.values() if r.region_id != 0]
    lat_scale = np.cos(np.deg2rad(latitude))
    return min(
        candidates,
        key=lambda r: (r.latitude - latitude) ** 2 + ((r.longitude - longitude) * lat_scale) ** 2,
    )
