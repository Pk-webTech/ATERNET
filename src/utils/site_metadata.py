"""
Static metadata describing the PV site(s) used to synthesize / process
data. Centralizing this makes it trivial to swap in a real site later
without touching the pipeline logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteMetadata:
    site_id: str
    latitude: float
    longitude: float
    altitude_m: float
    capacity_kw: float
    tilt_deg: float
    azimuth_deg: float
    timezone: str


# Default demonstration site (Chennai, India -- matches user context)
DEFAULT_SITE = SiteMetadata(
    site_id="SITE_001",
    latitude=13.0827,
    longitude=80.2707,
    altitude_m=6.0,
    capacity_kw=1000.0,
    tilt_deg=13.0,
    azimuth_deg=180.0,
    timezone="Asia/Kolkata",
)
