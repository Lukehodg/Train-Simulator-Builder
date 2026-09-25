"""Geometry helpers: local metric projection, along-track sampling, bearings, nearest-neighbour joins."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

WGS84 = CRS.from_epsg(4326)


def local_crs(lon: float, lat: float, country: str | None = None) -> CRS:
    """Metric CRS for the route: British National Grid for GB, otherwise the UTM zone of the centroid."""
    if (country or "").upper() in {"GB", "UK"}:
        return CRS.from_epsg(27700)
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


@dataclass
class Projector:
    crs: CRS

    def __post_init__(self) -> None:
        self._fwd = Transformer.from_crs(WGS84, self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, WGS84, always_xy=True)

    def to_xy(self, lon, lat):
        return self._fwd.transform(lon, lat)

    def to_lonlat(self, x, y):
        return self._inv.transform(x, y)

    def line_to_xy(self, line: LineString) -> LineString:
        xs, ys = zip(*line.coords)
        px, py = self.to_xy(np.asarray(xs), np.asarray(ys))
        return LineString(np.column_stack([px, py]))


def sample_line(line_xy: LineString, spacing_m: float) -> dict[str, np.ndarray]:
    """Fixed-interval samples along a projected line: x, y, distance_m, bearing_deg (clockwise from north)."""
    length = line_xy.length
    n = int(np.floor(length / spacing_m)) + 1
    dist = np.arange(n, dtype=np.float64) * spacing_m
    dist[-1] = min(dist[-1], length)
    pts = [line_xy.interpolate(d) for d in dist]
    x = np.fromiter((p.x for p in pts), dtype=np.float64, count=n)
    y = np.fromiter((p.y for p in pts), dtype=np.float64, count=n)
    return {"x": x, "y": y, "distance_m": dist, "bearing_deg": bearings(x, y)}


def bearings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.gradient(x)
    dy = np.gradient(y)
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def nearest_sample_index(sx: np.ndarray, sy: np.ndarray, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query point return (index of nearest sample, distance)."""
    tree = STRtree([Point(a, b) for a, b in zip(sx, sy)])
    idx = tree.nearest([Point(a, b) for a, b in zip(np.atleast_1d(px), np.atleast_1d(py))])
    idx = np.asarray(idx, dtype=np.int64)
    d = np.hypot(sx[idx] - np.atleast_1d(px), sy[idx] - np.atleast_1d(py))
    return idx, d


def substring_between(line_xy: LineString, start: Point, end: Point) -> LineString:
    """Clip a line to the section between the projections of two points (ordered along the line)."""
    from shapely.ops import substring

    a, b = line_xy.project(start), line_xy.project(end)
    if a > b:
        a, b = b, a
    return substring(line_xy, a, b)


def offset_xy(x: np.ndarray, y: np.ndarray, bearing_deg: np.ndarray, offset_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Offset points perpendicular to travel (positive = right-hand side)."""
    b = np.radians(bearing_deg)
    return x + np.cos(b) * offset_m, y - np.sin(b) * offset_m
