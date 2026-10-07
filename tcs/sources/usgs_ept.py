"""USGS 3DEP lidar point clouds (keyless, AWS Open Data `usgs-lidar-public`, Entwine Point Tiles) as a surface model
near the track: trees, buildings and bridges, which the 3DEP bare-earth service does not have.

Each project is an octree of LAZ tiles in Web Mercator. For a window we read every tile down to a depth whose voxel
grid is ~1.3 m on the ground (enough for a 2 m surface, far less than the full density), keep the highest point per
2 m cell (noise classes dropped), and hang the result on the 3DEP bare earth: heights above the project's own ground
points are added to the 3DEP terrain, so the two vertical datums never have to agree. Tiles are cached on disk and
shared by every window and route; projects are chosen per window, newest survey first.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer

from .base import USER_AGENT, SourceUnavailable

INDEX = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
GROUND, NOISE = 2, (7, 18)
TARGET_RES_M = 1.4                      # deepest octree level read: voxel size on the ground (Web Mercator metres x cos(lat))


def _year(name: str) -> int:
    ys = [int(y) for y in re.findall(r"(?<!\d)(20\d\d|19\d\d)(?!\d)", name)]
    return max(ys) if ys else 0


class _Project:
    def __init__(self, url: str, http: requests.Session, cache: Path):
        self.base = url.rsplit("/", 1)[0]
        self.name = self.base.rsplit("/", 1)[1]
        self.http, self.cache = http, cache / self.name
        info = self._json("ept.json")
        if info.get("dataType") != "laszip" or int((info.get("srs") or {}).get("horizontal", 3857)) != 3857:
            raise SourceUnavailable(f"{self.name}: unsupported EPT ({info.get('dataType')}, {info.get('srs')})")
        b = info["bounds"]
        self.x0, self.y0, self.width = float(b[0]), float(b[1]), float(b[3]) - float(b[0])
        self.span = int(info["span"])
        self.hier: dict[str, int] = {}
        self.loaded: set[str] = set()
        self.lock = threading.Lock()
        self._load_hierarchy("0-0-0-0")

    def _json(self, rel: str) -> dict:
        p = self.cache / rel
        if p.exists():
            return json.loads(p.read_text())
        r = self.http.get(f"{self.base}/{rel}", timeout=60)
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(r.text)
        return r.json()

    def _load_hierarchy(self, key: str) -> None:
        if key in self.loaded:
            return
        self.hier.update(self._json(f"ept-hierarchy/{key}.json"))
        self.loaded.add(key)

    def nodes(self, bbox: tuple[float, float, float, float], depth: int) -> list[str]:
        """Tiles (any Z) at depths 0..depth whose footprint meets bbox (Web Mercator)."""
        out, frontier = [], ["0-0-0-0"]
        while frontier:
            key = frontier.pop()
            with self.lock:
                n = self.hier.get(key)
                if n == -1:
                    self._load_hierarchy(key)
                    n = self.hier.get(key, 0)
            if not n:
                continue
            d, x, y, z = map(int, key.split("-"))
            w = self.width / (1 << d)
            if self.x0 + (x + 1) * w < bbox[0] or self.x0 + x * w > bbox[2] or self.y0 + (y + 1) * w < bbox[1] or self.y0 + y * w > bbox[3]:
                continue
            out.append(key)
            if d < depth:
                frontier += [f"{d + 1}-{2 * x + i}-{2 * y + j}-{2 * z + k}" for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        return out

    def points(self, key: str) -> np.ndarray:
        """(n, 4) x, y, z, class for one tile (Web Mercator), cached on disk as LAZ."""
        import laspy

        p = self.cache / "ept-data" / f"{key}.laz"
        if not p.exists():
            r = self.http.get(f"{self.base}/ept-data/{key}.laz", timeout=120)
            r.raise_for_status()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".{threading.get_ident()}.part")     # two windows may fetch the same tile at once
            tmp.write_bytes(r.content)
            tmp.replace(p)
        las = laspy.read(p)
        return np.c_[np.asarray(las.x), np.asarray(las.y), np.asarray(las.z), np.asarray(las.classification)]


class Ept:
    """Surface heights above ground from the USGS lidar point clouds, for windows in the route's grid."""

    def __init__(self, epsg: int, cache: Path):
        from shapely.geometry import shape
        from shapely.strtree import STRtree

        self.cache = cache
        self.http = requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT
        p = cache / "resources.geojson"
        if not p.exists():
            r = self.http.get(INDEX, timeout=120)
            r.raise_for_status()
            cache.mkdir(parents=True, exist_ok=True)
            p.write_text(r.text)
        feats = json.loads(p.read_text())["features"]
        self.meta = [f["properties"] for f in feats]
        self.tree = STRtree([shape(f["geometry"]) for f in feats])
        self.to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
        self.to_wm = Transformer.from_crs(epsg, 3857, always_xy=True)
        self.from_wm = Transformer.from_crs(3857, epsg, always_xy=True)
        self.projects: dict[str, _Project | None] = {}
        self.lock = threading.Lock()
        self.memo: dict[tuple[str, str], np.ndarray] = {}

    def _project(self, url: str) -> _Project | None:
        with self.lock:
            if url in self.projects:
                return self.projects[url]
        try:
            pr = _Project(url, self.http, self.cache)
        except (requests.RequestException, SourceUnavailable, KeyError, ValueError):
            pr = None
        with self.lock:
            self.projects[url] = pr
        return pr

    def _tile(self, pr: _Project, key: str) -> np.ndarray:
        k = (pr.name, key)
        with self.lock:
            if k in self.memo:
                return self.memo[k]
        pts = pr.points(key)
        with self.lock:
            if len(self.memo) > 150:                     # tiles are reused by neighbouring windows; keep the recent ones
                self.memo.pop(next(iter(self.memo)))
            self.memo[k] = pts
        return pts

    def heights(self, grid, dtm: np.ndarray) -> tuple[np.ndarray, str | None]:
        """Surface model for the window of `grid` (route grid, 2 m): 3DEP terrain + the point cloud's height above its
        own ground, highest point per cell; NaN where no project covers the cell. Returns (surface, project name)."""
        from shapely.geometry import box

        x0, y0, x1, y1 = grid.bbox
        lon, lat = self.to_ll.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        cands = [self.meta[i] for i in self.tree.query(box(min(lon), min(lat), max(lon), max(lat)))]
        cands.sort(key=lambda m: _year(m["name"]), reverse=True)
        wx, wy = self.to_wm.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        wbox = (min(wx) - 2, min(wy) - 2, max(wx) + 2, max(wy) + 2)
        h, w = dtm.shape
        out = np.full((h, w), np.nan, dtype=np.float32)
        used = None
        for m in cands:
            if not np.isnan(out).any():
                break
            pr = self._project(m["url"])
            if pr is None:
                continue
            res0 = pr.width / pr.span * np.cos(np.radians(np.mean(lat)))
            depth = max(0, int(np.ceil(np.log2(res0 / TARGET_RES_M))))
            parts = [self._tile(pr, k) for k in pr.nodes(wbox, depth)]
            if not parts:
                continue
            p = np.vstack(parts)
            p = p[(p[:, 0] >= wbox[0]) & (p[:, 0] <= wbox[2]) & (p[:, 1] >= wbox[1]) & (p[:, 1] <= wbox[3]) & ~np.isin(p[:, 3], NOISE)]
            if len(p) < 50:
                continue
            gx, gy = self.from_wm.transform(p[:, 0], p[:, 1])
            c = np.floor((np.asarray(gx) - grid.x0) / grid.res).astype(np.int64)
            r = np.floor((grid.y1 - np.asarray(gy)) / grid.res).astype(np.int64)
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            c, r, z, cls = c[ok], r[ok], p[ok, 2], p[ok, 3]
            g = (cls == GROUND) & np.isfinite(dtm[r, c])
            if g.sum() < 20:                             # no ground to tie the cloud to the 3DEP terrain
                continue
            offset = float(np.median(z[g] - dtm[r[g], c[g]]))
            top = np.full(h * w, -np.inf, dtype=np.float64)
            np.maximum.at(top, r * w + c, z - offset)
            top = top.reshape(h, w).astype(np.float32)
            top[~np.isfinite(top)] = np.nan
            fill = np.isnan(out) & np.isfinite(top)
            out[fill] = top[fill]
            used = used or pr.name
        return out, used
