"""Open LiDAR along the track (GB): bare-earth terrain (DTM) and the first-return surface (DSM: trees, buildings,
bridges), read only in a corridor either side of the line, at 2 m.

England   Environment Agency LIDAR Composite, DTM 1 m and first-return DSM 1 m, over WCS (GetCoverage at scale 0.5,
          deflate). Covers ~99 % of England.
Wales     Welsh Government LiDAR 2020-23, 1 km GeoTIFF tiles at 1 m, found through the DataMapWales tile catalogue (WFS).
Scotland  Scottish public-sector LiDAR on the srsp-open-data bucket: the national programme (1 km tiles, 50 cm) and the
          earlier phases (10 km tiles at 1 m, 5 km quarter tiles at 50 cm), newest first. Coverage has gaps.

All Open Government Licence v3. Sources are tried in turn (the one that served the last window first) and each fills
only what is still missing, so a window on a border is mosaicked. A read that fails (a busy service) marks the window,
so the caller can try it again rather than take the gap for "no survey". Rasters are British National Grid (EPSG:27700), the grid GB routes are sampled in.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from .base import USER_AGENT, SourceUnavailable, console

RES = 2.0                                       # metres per pixel everywhere

EA_LAYERS = {
    "dtm": ("https://environment.data.gov.uk/geoservices/datasets/13787b9a-26a4-4775-8523-806d13af58fc/wcs",
            "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"),
    "dsm": ("https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs",
            "df4e3ec3-315e-48aa-aaaf-b5ae74d7b2bb__Lidar_Composite_Elevation_FZ_DSM_1m"),
}
EA_EXTENT = {"dtm": (80000, 4000, 656000, 665000), "dsm": (133000, 11000, 656000, 657600)}   # outside these the service answers 500
WALES_WFS = "https://datamap.gov.wales/geoserver/ows"
WALES_LAYER = "geonode:welsh_government_lidar_tile_catalogue_2020_2023"
SCOT_BUCKET = "https://srsp-open-data.s3.eu-west-2.amazonaws.com/"
SCOT_SOURCES = ["national-lidar-programme", "phase-6", "phase-5", "phase-4", "phase-3", "phase-2", "phase-1",
                "hes", "coastal", "outer-hebrides", "orkney-islands-council-23"]          # newest first
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
            "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "VSI_CACHE": "TRUE",
            "GDAL_HTTP_USERAGENT": USER_AGENT, "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1"}


# ---------------------------------------------------------------- grid
@dataclass
class Grid:
    """A raster window in British National Grid: arr[row, col] covers x0 + col*res .. , y1 - row*res .. (NaN = no data)."""
    x0: float
    y1: float
    arr: np.ndarray
    res: float = RES
    failed: bool = False                        # a source could not be read (service error): what is missing may be there

    @classmethod
    def blank(cls, bbox: tuple[float, float, float, float], res: float = RES) -> "Grid":
        x0, y0, x1, y1 = bbox
        return cls(x0, y1, np.full((int(round((y1 - y0) / res)), int(round((x1 - x0) / res))), np.nan, dtype=np.float32), res)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        h, w = self.arr.shape
        return self.x0, self.y1 - h * self.res, self.x0 + w * self.res, self.y1

    def missing(self) -> float:
        return float(np.isnan(self.arr).mean()) if self.arr.size else 1.0

    def place(self, data: np.ndarray, left: float, top: float) -> None:
        """Copy `data` (same resolution, NaN = no data) into the cells still missing."""
        r0, c0 = int(round((self.y1 - top) / self.res)), int(round((left - self.x0) / self.res))
        h, w = self.arr.shape
        rs, cs = max(r0, 0), max(c0, 0)
        re_, ce = min(r0 + data.shape[0], h), min(c0 + data.shape[1], w)
        if re_ <= rs or ce <= cs:
            return
        src = data[rs - r0:re_ - r0, cs - c0:ce - c0]
        dst = self.arr[rs:re_, cs:ce]
        fill = np.isnan(dst) & np.isfinite(src)
        dst[fill] = src[fill]

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = np.floor((np.asarray(x) - self.x0) / self.res).astype(np.int64)
        r = np.floor((self.y1 - np.asarray(y)) / self.res).astype(np.int64)
        h, w = self.arr.shape
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        out = np.full(np.shape(x), np.nan, dtype=np.float32)
        out[ok] = self.arr[r[ok], c[ok]]
        return out


def median3(a: np.ndarray) -> np.ndarray:
    """3x3 median that ignores missing cells: thin things (masts, gantries, wires) vanish, trees and bridges stay."""
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    stack = np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3)])
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(stack, axis=0).astype(np.float32)


# ---------------------------------------------------------------- OS grid references
_LETTERS = {}
for _e in range(0, 8):
    for _n in range(0, 13):
        _l1 = (19 - _n) - (19 - _n) % 5 + (_e + 10) // 5
        _l2 = (19 - _n) * 5 % 25 + _e % 5
        _l1 += _l1 > 7
        _l2 += _l2 > 7
        _LETTERS[chr(65 + _l1) + chr(65 + _l2)] = (_e, _n)
_SQUARE = {v: k for k, v in _LETTERS.items()}


def grid_ref(x: float, y: float, digits: int = 1) -> str:
    """OS grid reference of the square containing (x, y): digits=1 -> 10 km ('NT27'), 2 -> 1 km ('NT2573')."""
    sq = _SQUARE[(int(x // 100000), int(y // 100000))]
    div = 10 ** (5 - digits)
    return f"{sq}{int(x % 100000 // div):0{digits}d}{int(y % 100000 // div):0{digits}d}"


def tile_bounds(name: str) -> tuple[float, float, float, float] | None:
    """Bounds of a Scottish tile from its file name: NT27_… (10 km), NT27SE_… (5 km quarter), NT2573_… (1 km)."""
    m = re.match(r"^([A-Z]{2})(\d{2}|\d{4})(NE|NW|SE|SW)?_", name)
    if not m or m.group(1) not in _LETTERS:
        return None
    e100, n100 = _LETTERS[m.group(1)]
    d = m.group(2)
    k = len(d) // 2
    size = 10000 if k == 1 else 1000
    x0 = e100 * 100000 + int(d[:k]) * size
    y0 = n100 * 100000 + int(d[k:]) * size
    if m.group(3):
        size //= 2
        x0 += size if m.group(3)[1] == "E" else 0
        y0 += size if m.group(3)[0] == "N" else 0
    return x0, y0, x0 + size, y0 + size


# ---------------------------------------------------------------- sources
class _Http:
    def __init__(self):
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
            self._local.s = s
        return s


def _read_tiff(content: bytes) -> tuple[np.ndarray, float, float, float] | None:
    from rasterio.io import MemoryFile

    try:
        with MemoryFile(content) as mf, mf.open() as ds:
            a = ds.read(1).astype(np.float32)
            nd = ds.nodata
            a[(a < -1000) | (a > 2000) | (np.isclose(a, nd) if nd is not None else False)] = np.nan
            return a, ds.transform.c, ds.transform.f, ds.transform.a
    except Exception:  # an XML exception report instead of a coverage (outside England, server trouble)
        return None


def _resample(a: np.ndarray, src_res: float, how: str) -> np.ndarray:
    """Block-aggregate to RES (src_res divides RES): max for surfaces (keeps tree tops), mean for terrain."""
    k = int(round(RES / src_res))
    if k <= 1:
        return a
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:h, :w].reshape(h // k, k, w // k, k)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmax(b, axis=(1, 3)) if how == "max" else np.nanmean(b, axis=(1, 3))).astype(np.float32)


class EnglandWCS:
    name = "lidar_ea"

    def __init__(self, http: _Http):
        self.http = http

    def fill(self, grid: Grid, layer: str) -> None:
        url, cov = EA_LAYERS[layer]
        ex0, ey0, ex1, ey1 = EA_EXTENT[layer]
        gx0, gy0, gx1, gy1 = grid.bbox
        x0, y0 = np.ceil(max(gx0, ex0) / RES) * RES, np.ceil(max(gy0, ey0) / RES) * RES
        x1, y1 = np.floor(min(gx1, ex1) / RES) * RES, np.floor(min(gy1, ey1) / RES) * RES
        if x1 <= x0 or y1 <= y0:                         # wholly outside England's survey: nothing to ask for
            return
        q = (f"{url}?service=WCS&version=2.0.1&request=GetCoverage&CoverageId={cov}&format=image/tiff"
             f"&subset=E({x0:.0f},{x1:.0f})&subset=N({y0:.0f},{y1:.0f})&SCALEFACTOR={1 / RES:g}&geotiff:compression=Deflate")
        try:
            r = self.http.session.get(q, timeout=90)
        except requests.RequestException:                # retried already (HTTP adapter): a busy or failing service
            grid.failed = True
            return
        if r.status_code != 200 or not r.content.startswith((b"II", b"MM")):
            grid.failed = True
            return
        got = _read_tiff(r.content)
        if got is None:                                  # a truncated or corrupt answer
            grid.failed = True
            return
        a, left, top, res = got
        a[a == 0.0] = np.nan                             # outside England the service returns 0, not its no-data value
        grid.place(a if abs(res - RES) < 1e-6 else _resample(a, res, "max" if layer == "dsm" else "mean"), left, top)


class _TileSource:
    """Tiles found through an index, read as windows with rasterio (cloud-optimised where the publisher made them so)."""

    name = "lidar"

    def tiles(self, grid: Grid, layer: str) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Tiles overlapping the grid; sets grid.failed when the index could not be read."""
        raise NotImplementedError

    def fill(self, grid: Grid, layer: str) -> None:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds

        x0, y0, x1, y1 = grid.bbox
        for url, (tx0, ty0, tx1, ty1) in self.tiles(grid, layer):
            if grid.missing() == 0:
                return
            bx0, by0, bx1, by1 = max(x0, tx0), max(y0, ty0), min(x1, tx1), min(y1, ty1)
            if bx1 <= bx0 or by1 <= by0:
                continue
            try:
                with rasterio.Env(**GDAL_ENV), rasterio.open(f"/vsicurl/{url}") as ds:
                    win = from_bounds(bx0, by0, bx1, by1, ds.transform)
                    shape = (max(1, int(round((by1 - by0) / RES))), max(1, int(round((bx1 - bx0) / RES))))
                    a = ds.read(1, window=win, out_shape=shape, boundless=True, fill_value=ds.nodata if ds.nodata is not None else -9999,
                                resampling=Resampling.max if layer == "dsm" else Resampling.average).astype(np.float32)
                    nd = ds.nodata
                    a[(a < -1000) | (a > 2000) | (np.isclose(a, nd) if nd is not None else False)] = np.nan
            except Exception as exc:  # leave the cells; a dead link in the index is final, anything else worth another try
                if not re.search(r"\b404\b|does not exist|No such file", str(exc)):
                    grid.failed = True
                continue
            grid.place(a, bx0, by1)


class WalesTiles(_TileSource):
    name = "lidar_wales"

    def __init__(self, http: _Http, cache: Path):
        self.http, self.cache = http, cache
        self._index: dict[str, list] = {}
        self._lock = threading.Lock()

    def _square(self, ref10: str, sx0: float, sy0: float) -> list:
        with self._lock:
            if ref10 in self._index:
                return self._index[ref10]
        f = self.cache / f"wales_{ref10}.json"
        if f.exists():
            feats = json.loads(f.read_text())
        else:
            params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": WALES_LAYER, "outputFormat": "application/json",
                      "bbox": f"{sx0},{sy0},{sx0 + 10000},{sy0 + 10000},urn:ogc:def:crs:EPSG::27700", "count": "500"}
            try:
                r = self.http.session.get(WALES_WFS, params=params, timeout=90)
                r.raise_for_status()
                feats = [{"dtm": p.get("dtm_link"), "dsm": p.get("dsm_link"), "ref": p.get("british_gr")}
                         for p in (ft.get("properties") or {} for ft in r.json().get("features", []))]
            except (requests.RequestException, ValueError):
                return None                                 # not cached: try again next time
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(feats))
        with self._lock:
            self._index[ref10] = feats
        return feats

    def tiles(self, grid, layer):
        x0, y0, x1, y1 = grid.bbox
        out = []
        for sx in range(int(x0 // 10000) * 10000, int(x1 // 10000) * 10000 + 1, 10000):
            for sy in range(int(y0 // 10000) * 10000, int(y1 // 10000) * 10000 + 1, 10000):
                try:
                    ref10 = grid_ref(sx + 1, sy + 1)
                except KeyError:
                    continue
                sq = self._square(ref10, sx, sy)
                if sq is None:
                    grid.failed = True
                    continue
                for t in sq:
                    link, ref = t.get(layer), t.get("ref")
                    b = tile_bounds(f"{ref}_") if ref else None
                    if link and b and b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0:
                        out.append(("https://" + link.removeprefix("https://"), b))
        return out


class ScotlandTiles(_TileSource):
    name = "lidar_scotland"

    def __init__(self, http: _Http, cache: Path):
        self.http, self.cache = http, cache
        self._index: dict[tuple, list] = {}
        self._lock = threading.Lock()

    def _list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000", **({"continuation-token": token} if token else {})}
            r = self.http.session.get(SCOT_BUCKET, params=params, timeout=60)
            r.raise_for_status()
            keys += re.findall(r"<Key>([^<]*)</Key>", r.text)
            m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", r.text)
            if not m:
                return keys
            token = m.group(1)

    def _square(self, ref10: str, layer: str) -> list:
        key = (ref10, layer)
        with self._lock:
            if key in self._index:
                return self._index[key]
        f = self.cache / f"scotland_{layer}_{ref10}.json"
        if f.exists():
            found = json.loads(f.read_text())
        else:
            found = []
            try:
                for src in SCOT_SOURCES:
                    # 10 km and quarter tiles start with the 10 km reference; 1 km tiles (national programme) share
                    # only the letters and first easting digit, so list those and keep the matching northing digit.
                    pre = ref10 if src != "national-lidar-programme" else ref10[:3]
                    for k in self._list(f"lidar/{src}/{layer}/27700/gridded/{pre}"):
                        name = k.rsplit("/", 1)[-1]
                        b = tile_bounds(name)
                        if b and grid_ref(b[0] + 1, b[1] + 1) == ref10:
                            found.append([SCOT_BUCKET + k, list(b), SCOT_SOURCES.index(src)])
            except requests.RequestException:
                return None                                 # not cached: try again next time
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(found))
        found = sorted(found, key=lambda t: (t[2], t[1][2] - t[1][0]))      # newest source first, then finer tiles
        with self._lock:
            self._index[key] = found
        return found

    def tiles(self, grid, layer):
        x0, y0, x1, y1 = grid.bbox
        out = []
        for sx in range(int(x0 // 10000) * 10000, int(x1 // 10000) * 10000 + 1, 10000):
            for sy in range(int(y0 // 10000) * 10000, int(y1 // 10000) * 10000 + 1, 10000):
                try:
                    ref10 = grid_ref(sx + 1, sy + 1)
                except KeyError:
                    continue
                sq = self._square(ref10, layer)
                if sq is None:
                    grid.failed = True
                    continue
                for url, b, _ in sq:
                    if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0:
                        out.append((url, tuple(b)))
        return out


class Lidar:
    """DTM / DSM windows for GB from the three national surveys, each filling only what the previous left missing."""

    def __init__(self, cache_dir: Path, offline: bool = False):
        if offline:
            raise SourceUnavailable("LiDAR needs the network")
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        http = _Http()
        self.sources = [EnglandWCS(http), WalesTiles(http, cache_dir), ScotlandTiles(http, cache_dir)]
        self._lock = threading.Lock()

    def read(self, bbox: tuple[float, float, float, float], layer: str) -> tuple[Grid, str | None]:
        """The window and the source that supplied most of it (None: nothing there). The source that served the
        last window is asked first: along a route the next window is almost always in the same nation."""
        g = Grid.blank(bbox)
        best, best_n = None, 0
        with self._lock:
            order = list(self.sources)
        for src in order:
            before = int(np.isnan(g.arr).sum())
            if before == 0:
                break
            src.fill(g, layer)
            got = before - int(np.isnan(g.arr).sum())
            if got > best_n:
                best, best_n = src, got
        if best is not None and best is not order[0]:
            with self._lock:
                self.sources.remove(best)
                self.sources.insert(0, best)
        return g, best.name if best is not None else None


# ---------------------------------------------------------------- per-sample features
SIDE = np.arange(4.0, 42.0, 2.0)                 # lateral offsets for cutting walls and embankment slopes (m)
ALONG = np.arange(-20.0, 21.0, 4.0)              # along-track offsets averaged over a sample's stretch (m)
RAIL_LAT = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
OVER = np.arange(-24.0, 25.0, 2.0)               # along the centreline, for structures over the line
OVERHEAD_M = 5.5                                 # surface this far above the rail is a structure over the line (trains are ~4 m)


def corridor_features(x: np.ndarray, y: np.ndarray, bearing_deg: np.ndarray, on_bridge: np.ndarray, roofed: np.ndarray,
                      lidar: Lidar, *, corridor_m: float = 60.0, antenna_h_m: float = 4.0, azimuths: int = 16,
                      segment: int = 6, workers: int = 8, retry_pause_s: float = 5.0, log=console.log) -> dict[str, np.ndarray]:
    """For every route sample: rail level, cutting walls / embankment fall either side, the near-field skyline
    (DSM, per azimuth, out to corridor_m), the share of the stretch under a structure, and the share of the ground
    within 30 m under trees or buildings. lidar_ok is False where the surveys have no data (keep the DEM values there).

    on_bridge: the line itself is on a structure (OSM), so its rail level is the deck (DSM), not the ground below.
    roofed: a station roof spans the line here, so a long run of high surface is a roof, not a viaduct."""
    from concurrent.futures import ThreadPoolExecutor

    n = len(x)
    az = np.radians(np.linspace(0, 360, azimuths, endpoint=False))
    rr = np.arange(6.0, corridor_m + 1e-6, RES)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ("rail_level_m", "wall_left_m", "wall_right_m", "fall_left_m", "fall_right_m",
                                                             "overhead_fraction", "obstruction_share")}
    out["horizon_near_deg"] = np.full((n, azimuths), np.nan, dtype=np.float32)
    out["lidar_ok"] = np.zeros(n, dtype=bool)
    out["lidar_source"] = np.array([""] * n, dtype=object)
    out["lidar_failed"] = np.zeros(n, dtype=bool)        # a survey could not be read here (service error), not "no survey"
    pad = corridor_m + 4 * RES

    def run(s0: int) -> bool:
        """Features for one stretch of samples; False when a read failed, so the stretch is worth another try."""
        s1 = min(n, s0 + segment)
        xs, ys = x[s0:s1], y[s0:s1]
        bbox = tuple(float(v) for v in (np.floor((xs.min() - pad) / RES) * RES, np.floor((ys.min() - pad) / RES) * RES,
                                        np.ceil((xs.max() + pad) / RES) * RES, np.ceil((ys.max() + pad) / RES) * RES))
        dtm, src = lidar.read(bbox, "dtm")
        if src is None or dtm.missing() > 0.9:
            return not dtm.failed
        dsm, _ = lidar.read(bbox, "dsm")
        if dsm.failed:
            return False
        dsmf = median3(dsm.arr)
        dsmg = Grid(dsm.x0, dsm.y1, dsmf)
        import warnings

        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(s0, s1):
            b = np.radians(bearing_deg[i])
            ax_, ay_ = np.sin(b), np.cos(b)                  # along track
            lx_, ly_ = np.cos(b), -np.sin(b)                 # to the right of travel
            px = x[i] + ALONG[:, None] * ax_ + RAIL_LAT[None, :] * lx_
            py = y[i] + ALONG[:, None] * ay_ + RAIL_LAT[None, :] * ly_
            grid_for_rail = dsmg if on_bridge[i] else dtm
            rail = float(np.nanmedian(grid_for_rail.sample(px, py)))
            if not np.isfinite(rail):
                continue
            # a long run of surface well above the ground along the centreline: the line is on a structure the map
            # did not tag (a viaduct), unless a station roof spans it here
            ox, oy = x[i] + OVER * ax_, y[i] + OVER * ay_
            top = dsmg.sample(ox, oy)
            high = (top - rail) > OVERHEAD_M
            valid = np.isfinite(top)
            if not on_bridge[i] and not roofed[i] and valid.sum() >= 0.8 * len(OVER) and high[valid].all():
                rail = float(np.nanmedian(dsmg.sample(px, py)))
                high = (top - rail) > OVERHEAD_M
            out["rail_level_m"][i] = rail
            out["overhead_fraction"][i] = float(high[valid].mean()) if valid.any() else 0.0
            for side, sgn in (("left", -1.0), ("right", 1.0)):
                qx = x[i] + ALONG[:, None] * ax_ + sgn * SIDE[None, :] * lx_
                qy = y[i] + ALONG[:, None] * ay_ + sgn * SIDE[None, :] * ly_
                prof = np.nanmedian(dtm.sample(qx, qy), axis=0) - rail       # per lateral offset, over the stretch
                if np.isfinite(prof).sum() >= len(SIDE) // 2:
                    out[f"wall_{side}_m"][i] = max(0.0, float(np.nanmax(prof)))
                    out[f"fall_{side}_m"][i] = max(0.0, float(-np.nanmin(prof[SIDE >= 6])))
            # skyline: surface (trees, buildings, walls) seen from the roof antenna, per azimuth
            hx = x[i] + np.sin(az)[:, None] * rr[None, :]
            hy = y[i] + np.cos(az)[:, None] * rr[None, :]
            ang = np.degrees(np.arctan2(dsmg.sample(hx, hy) - (rail + antenna_h_m), rr[None, :]))
            out["horizon_near_deg"][i] = np.nanmax(np.where(np.isfinite(ang), ang, -90.0), axis=1)
            # ground within 30 m either side with something standing above the roof antenna (trees, buildings, walls);
            # measured from the rail, so a viaduct carrying the line is not counted against it
            tx = x[i] + ALONG[::2, None] * ax_ + np.r_[-SIDE[SIDE <= 30], SIDE[SIDE <= 30]][None, :] * lx_
            ty = y[i] + ALONG[::2, None] * ay_ + np.r_[-SIDE[SIDE <= 30], SIDE[SIDE <= 30]][None, :] * ly_
            above = dsmg.sample(tx, ty) - (rail + antenna_h_m)
            ok = np.isfinite(above)
            out["obstruction_share"][i] = float((above[ok] > 0.0).mean()) if ok.any() else np.nan
            out["lidar_ok"][i] = np.isfinite(out["wall_left_m"][i]) or np.isfinite(out["wall_right_m"][i])
            out["lidar_source"][i] = src
        return not dtm.failed

    starts = list(range(0, n, segment))
    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s0, good in zip(starts, ex.map(run, starts)):
            done += 1
            if not good:
                failed.append(s0)
            if done % 200 == 0:
                log(f"LiDAR: {done * segment:,} of {n:,} samples")
    if failed:                                           # busy services: one slower pass over what failed
        log(f"LiDAR: {len(failed) * segment:,} samples hit a service error; trying them again")
        time.sleep(retry_pause_s)
        with ThreadPoolExecutor(max_workers=2) as ex:
            failed = [s0 for s0, good in zip(failed, ex.map(run, failed)) if not good]
        for s0 in failed:
            out["lidar_failed"][s0:s0 + segment] = True
    # a rail level far from its neighbours is a misplaced centreline or a gap in the survey, not a step in the railway
    rl = out["rail_level_m"]
    ok = out["lidar_ok"] & np.isfinite(rl)
    if ok.sum() > 5:
        import pandas as pd

        med = pd.Series(np.where(ok, rl, np.nan)).rolling(5, center=True, min_periods=3).median().to_numpy()
        bad = ok & np.isfinite(med) & (np.abs(rl - med) > 3.0) & ~on_bridge
        out["lidar_ok"][bad] = False
    return out


# ---------------------------------------------------------------- United States
USGS_3DEP = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"


class Usgs3dep:
    """US elevation from the USGS 3D Elevation Program (keyless). 'dtm': the national best-available bare-earth
    mosaic, lidar-derived at 1 m wherever 3DEP lidar has been flown (all of the Northeast Corridor), read at 2 m in the
    route's own grid. 'dsm': with `surface` (a cache folder), trees, buildings and bridges from the 3DEP lidar point
    clouds (sources/usgs_ept.py) on top of that bare earth; without it, or where no point cloud covers a cell, the bare
    earth again (cutting walls, embankments and the ground skyline only). Same interface as Lidar."""

    name = "usgs_3dep"

    def __init__(self, epsg: int, offline: bool = False, surface: Path | None = None):
        if offline:
            raise SourceUnavailable("3DEP needs the network")
        self.epsg = epsg
        self.ept = None
        if surface is not None:
            from .usgs_ept import Ept

            try:
                self.ept = Ept(epsg, surface)
            except (requests.RequestException, OSError, ValueError) as exc:
                console.log(f"[yellow]USGS point clouds unavailable ({exc}); bare earth only")
        self.surface_cells = [0, 0]                      # cells with a point-cloud surface, cells read
        self.http = _Http()
        self._last: dict[int, tuple[tuple, Grid]] = {}
        self._lock = threading.Lock()

    def read(self, bbox: tuple[float, float, float, float], layer: str) -> tuple[Grid, str | None]:
        tid = threading.get_ident()
        with self._lock:
            hit = self._last.get(tid)
        if hit is not None and hit[0] == bbox:              # 'dsm' right after 'dtm' for the same window: one request
            g = hit[1]
            arr = g.arr.copy()
            if layer == "dsm" and self.ept is not None and not g.failed and g.missing() < 1.0:
                try:
                    top, _ = self.ept.heights(g, g.arr)
                except (requests.RequestException, OSError, ValueError) as exc:
                    console.log(f"[yellow]USGS point cloud read failed ({exc}); bare earth for this window")
                    top = np.full_like(arr, np.nan)
                with self._lock:
                    self.surface_cells[0] += int(np.isfinite(top).sum())
                    self.surface_cells[1] += int(np.isfinite(arr).sum())
                arr = np.fmax(arr, top)                  # never below the ground; bare earth where the cloud is silent
            return Grid(g.x0, g.y1, arr, g.res, g.failed), (self.name if g.missing() < 1.0 else None)
        g = Grid.blank(bbox)
        h, w = g.arr.shape
        params = {"bbox": ",".join(f"{v:.1f}" for v in bbox), "bboxSR": self.epsg, "imageSR": self.epsg, "size": f"{w},{h}", "format": "tiff",
                  "pixelType": "F32", "noDataInterpretation": "esriNoDataMatchAny", "interpolation": "RSP_BilinearInterpolation", "f": "image"}
        try:
            r = self.http.session.get(USGS_3DEP, params=params, timeout=120)
            got = _read_tiff(r.content) if r.ok else None
        except requests.RequestException:
            got = None
        if got is None:
            g.failed = True
        else:
            a, left, top, _ = got
            if a.shape == g.arr.shape:
                g.place(a, left, top)
        with self._lock:
            self._last[tid] = (bbox, g)
        return Grid(g.x0, g.y1, g.arr.copy(), g.res, g.failed), (self.name if g.missing() < 1.0 else None)
