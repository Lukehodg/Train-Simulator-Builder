"""Elevation sources and the sky-visibility (horizon) model.

Default: Copernicus DEM GLO-30 (keyless Cloud-Optimised GeoTIFFs on AWS, global, 30 m). Only the corridor windows
are read, via HTTP range requests, so a 600 km route costs tens of MB rather than the full tiles.
Alternative for GB: OS Terrain 50 (OS OpenData) from a local folder of tiles.
Offline: a procedural surface, flagged synthetic.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rich.console import Console

console = Console(stderr=True)
COPERNICUS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


class Elevation(Protocol):
    source: str
    synthetic: bool

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------------------------------
@dataclass
class _Window:
    west: float
    south: float
    east: float
    north: float
    data: np.ndarray        # rows top->bottom
    res_x: float
    res_y: float

    def contains(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return (lon >= self.west) & (lon < self.east) & (lat > self.south) & (lat <= self.north)

    def bilinear(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        cx = (lon - self.west) / self.res_x - 0.5
        cy = (self.north - lat) / self.res_y - 0.5
        h, w = self.data.shape
        x0 = np.clip(np.floor(cx).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(cy).astype(int), 0, h - 2)
        fx = np.clip(cx - x0, 0, 1)
        fy = np.clip(cy - y0, 0, 1)
        d = self.data
        return (d[y0, x0] * (1 - fx) * (1 - fy) + d[y0, x0 + 1] * fx * (1 - fy)
                + d[y0 + 1, x0] * (1 - fx) * fy + d[y0 + 1, x0 + 1] * fx * fy)


def _tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


class CopernicusDEM:
    """GLO-30 sampled through corridor windows. Windows are requested for bounding boxes (in degrees) and cached."""

    source = "copernicus_glo30"
    synthetic = False

    def __init__(self, raw_dir: Path, offline: bool = False):
        self.raw_dir = raw_dir / "dem"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.windows: list[_Window] = []
        self._ds: dict[str, object] = {}

    def _open(self, name: str):
        import rasterio

        if name not in self._ds:
            url = f"/vsicurl/{COPERNICUS_BASE}/{name}/{name}.tif"
            self._ds[name] = rasterio.open(url)
        return self._ds[name]

    def prepare(self, west: float, south: float, east: float, north: float) -> None:
        """Load (or read from cache) every DEM window needed to cover a bbox, tile by tile."""
        from rasterio.windows import from_bounds

        for lat in range(math.floor(south), math.ceil(north)):
            for lon in range(math.floor(west), math.ceil(east)):
                w, s = max(west, lon), max(south, lat)
                e, n = min(east, lon + 1), min(north, lat + 1)
                if e <= w or n <= s:
                    continue
                key = f"{_tile_name(lat, lon)}_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}.npz"
                cache = self.raw_dir / key
                if cache.exists():
                    z = np.load(cache)
                    self.windows.append(_Window(float(z["w"]), float(z["s"]), float(z["e"]), float(z["n"]), z["data"], float(z["rx"]), float(z["ry"])))
                    continue
                if self.offline:
                    raise RuntimeError("offline: DEM window not cached")
                ds = self._open(_tile_name(lat, lon))
                win = from_bounds(w, s, e, n, ds.transform).round_offsets().round_lengths()
                data = ds.read(1, window=win).astype(np.float32)
                data[data == ds.nodata] = 0.0
                bounds = ds.window_bounds(win)
                rx, ry = ds.res
                np.savez_compressed(cache, w=bounds[0], s=bounds[1], e=bounds[2], n=bounds[3], data=data, rx=rx, ry=ry)
                self.windows.append(_Window(bounds[0], bounds[1], bounds[2], bounds[3], data, rx, ry))

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        out = np.full(lon.shape, np.nan, dtype=np.float32)
        for w in self.windows:
            m = w.contains(lon, lat) & np.isnan(out)
            if m.any():
                out[m] = w.bilinear(lon[m], lat[m])
        return np.nan_to_num(out, nan=0.0)


class OSTerrain50:
    """OS Terrain 50 (OS OpenData, 50 m grid, GB) from the "ASCII Grid and GML (Grid)" download.

    Drop the downloaded zip (terr50_gagg_gb.zip) or its extracted contents into the folder. The product nests one
    zip per 10 km tile (data/<sq>/<sq><nn>_OST50GRID_<date>.zip -> .asc); those are extracted on first use, and
    only tiles that intersect the corridor are opened. Tiles are read into memory (10 km at 50 m = 200x200 floats).
    """

    source = "os_terrain50"
    synthetic = False

    def __init__(self, folder: Path, bbox_bng: tuple[float, float, float, float] | None = None):
        import zipfile
        from pyproj import Transformer

        folder = Path(folder)
        extracted = folder / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        outer = sorted(folder.glob("*.zip"))
        for z in outer:                                   # outer product zip -> nested tile zips -> .asc
            stamp = extracted / f".{z.stem}.done"
            if stamp.exists():
                continue
            console.log(f"OS Terrain 50: extracting {z.name} (one-off)")
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".zip"):
                        inner = zipfile.ZipFile(io.BytesIO(zf.read(n)))
                        for m in inner.namelist():
                            if m.lower().endswith((".asc", ".prj")):
                                (extracted / Path(m).name).write_bytes(inner.read(m))
                    elif n.lower().endswith(".asc"):
                        (extracted / Path(n).name).write_bytes(zf.read(n))
            stamp.touch()
        files = sorted(set(folder.rglob("*.asc")) | set(folder.rglob("*.tif")))
        if not files:
            raise FileNotFoundError(f"no OS Terrain 50 tiles under {folder} (expected terr50_gagg_gb.zip or extracted .asc tiles)")
        self._to_bng = Transformer.from_crs(4326, 27700, always_xy=True)
        self.tiles: list[tuple[float, float, float, float, Path]] = []      # (left, bottom, right, top, path)
        for f in files:
            b = _asc_bounds(f)
            if b is None:
                continue
            if bbox_bng and (b[2] < bbox_bng[0] or b[0] > bbox_bng[2] or b[3] < bbox_bng[1] or b[1] > bbox_bng[3]):
                continue
            self.tiles.append((*b, f))
        if not self.tiles:
            raise FileNotFoundError("no OS Terrain 50 tiles intersect the route corridor")
        self._cache: dict[Path, tuple[np.ndarray, float, float, float]] = {}
        console.log(f"OS Terrain 50: {len(self.tiles)} tiles cover the corridor")

    def _grid(self, path: Path):
        if path not in self._cache:
            import rasterio

            with rasterio.open(path) as ds:
                data = ds.read(1).astype(np.float32)
                data[data == ds.nodata] = 0.0
                self._cache[path] = (data, ds.bounds.left, ds.bounds.top, float(ds.res[0]))
        return self._cache[path]

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x, y = self._to_bng.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        out = np.zeros(x.shape, dtype=np.float32)
        done = np.zeros(x.shape, dtype=bool)
        for left, bottom, right, top, path in self.tiles:
            m = ~done & (x >= left) & (x < right) & (y >= bottom) & (y < top)
            if not m.any():
                continue
            data, l, t, res = self._grid(path)
            h, w = data.shape
            cx = np.clip(((x[m] - l) / res).astype(int), 0, w - 1)
            cy = np.clip(((t - y[m]) / res).astype(int), 0, h - 1)
            out[m] = data[cy, cx]
            done |= m
        return out


def _asc_bounds(path: Path) -> tuple[float, float, float, float] | None:
    """Read an ESRI ASCII grid header (or a GeoTIFF's bounds) without loading the data."""
    if path.suffix.lower() == ".tif":
        import rasterio

        with rasterio.open(path) as ds:
            return (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    hdr: dict[str, float] = {}
    with open(path, "r", encoding="ascii", errors="ignore") as fh:
        for _ in range(6):
            parts = fh.readline().split()
            if len(parts) == 2:
                hdr[parts[0].lower()] = float(parts[1])
    try:
        cs = hdr["cellsize"]
        left = hdr.get("xllcorner", hdr.get("xllcenter", 0) - cs / 2)
        bottom = hdr.get("yllcorner", hdr.get("yllcenter", 0) - cs / 2)
        return (left, bottom, left + hdr["ncols"] * cs, bottom + hdr["nrows"] * cs)
    except KeyError:
        return None


class SyntheticDEM:
    """Procedural stand-in so the pipeline runs offline. Never presented as terrain truth."""

    source = "synthetic_terrain"
    synthetic = True

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0

    @staticmethod
    def _hash(n):
        s = np.sin(n) * 43758.5453
        return s - np.floor(s)

    def _noise2(self, x, y):
        ix, iy = np.floor(x), np.floor(y)
        fx, fy = x - ix, y - iy
        ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        h = lambda a, b: self._hash(a * 127.1 + b * 311.7)
        return (h(ix, iy) * (1 - ux) + h(ix + 1, iy) * ux) * (1 - uy) + (h(ix, iy + 1) * (1 - ux) + h(ix + 1, iy + 1) * ux) * uy

    def _fbm(self, x, y, oct_):
        a, s, f, n = 0.5, 0.0, 1.0, 0.0
        for o in range(oct_):
            s += a * self._noise2(x * f + o * 3.1, y * f - o * 7.3)
            n += a
            a *= 0.55
            f *= 2
        return s / n

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x = (np.asarray(lon) - self.lon0) * 111.32 * math.cos(math.radians(self.lat0 + 2))
        y = (np.asarray(lat) - self.lat0) * 110.57
        hills = 420 * np.clip(1 - np.hypot((x + 150) / 80, (y - 300) / 150), 0, None)
        base = 15 + 50 * self._fbm(x / 45, y / 45, 3) + 150 * self._fbm(x / 11, y / 11, 5) * (0.35 + 0.65 * self._fbm(x / 70, y / 70, 2))
        return np.maximum(2, base + hills).astype(np.float32)


# --------------------------------------------------------------------------------------------------
def sky_visibility(dem: Elevation, proj, x: np.ndarray, y: np.ndarray, elev: np.ndarray, *, azimuths: int = 16,
                   reach_m: float = 3000, step_m: float = 50, antenna_h_m: float = 4.0, min_elevation_deg: float = 20.0,
                   chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Fraction (by solid angle) of the usable sky dome above the terminal's minimum elevation that terrain leaves open.

    Returns (sky_visibility 0..1, mean horizon angle in degrees). Casts `azimuths` rays per sample up to `reach_m`.
    """
    n = len(x)
    az = np.radians(np.linspace(0, 360, azimuths, endpoint=False))
    r = np.arange(step_m, reach_m + step_m, step_m)
    sky = np.zeros(n, dtype=np.float32)
    horizon = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        xs, ys = x[s:e], y[s:e]
        # (chunk, az, r)
        px = xs[:, None, None] + np.sin(az)[None, :, None] * r[None, None, :]
        py = ys[:, None, None] + np.cos(az)[None, :, None] * r[None, None, :]
        lon, lat = proj.to_lonlat(px.ravel(), py.ravel())
        z = dem.sample(np.asarray(lon), np.asarray(lat)).reshape(px.shape)
        obs = (elev[s:e] + antenna_h_m)[:, None, None]
        ang = np.degrees(np.arctan2(z - obs, r[None, None, :]))
        hz = ang.max(axis=2)                                    # horizon per azimuth
        # Solid-angle weighting: the usable dome above the terminal's minimum elevation e0 has area ~ (1 - sin e0);
        # a horizon at h leaves (1 - sin h) of it. Average the visible fraction over azimuths.
        e0 = np.radians(min_elevation_deg)
        h = np.radians(np.clip(hz, min_elevation_deg, 90.0))
        sky[s:e] = ((1.0 - np.sin(h)) / (1.0 - np.sin(e0))).mean(axis=1)
        horizon[s:e] = hz.mean(axis=1)
    return sky, horizon
