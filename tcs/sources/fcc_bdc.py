"""FCC National Broadband Map (Broadband Data Collection) mobile coverage as the US coverage prior.

The US counterpart of Ofcom's predictions: each operator files where it provides 4G LTE and 5G-NR, as polygons per
state, for two environments: outdoor stationary and in-vehicle mobile. A rooftop train antenna sits between the two,
so in-vehicle coverage scores highest, stationary-only coverage lower, none lowest (`fcc_bdc` in the country's
networks file).

Access: the BDC public data API needs a free broadbandmap.fcc.gov account. FCC_BDC_USERNAME (the login email) and
FCC_BDC_TOKEN (Manage API Access -> Generate) go in the request headers. `tcs probe-fcc` lists what the API offers.

Files are large, so each is clipped to the route corridor once and only the clip is kept
(data/raw/<route>/fcc_bdc/), and the download is deleted.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import SourceUnavailable, console, http_get

API = "https://broadbandmap.fcc.gov/api/public/map"
SOURCE = "fcc_bdc"
TECH = {"4G": "400", "5G": "500"}
DEFAULT_SCORES = {"invehicle_5g": 0.82, "invehicle_4g": 0.72, "stationary_5g": 0.58, "stationary_4g": 0.50, "none": 0.10}
FIPS = {"AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "ID": "16",
        "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
        "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
        "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
        "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"}


def _headers(settings) -> dict[str, str]:
    user, token = settings.key("FCC_BDC_USERNAME"), settings.key("FCC_BDC_TOKEN")
    if not user or not token:
        raise SourceUnavailable("FCC_BDC_USERNAME / FCC_BDC_TOKEN not set")
    return {"username": user, "hash_value": token, "Accept": "application/json"}


def _get_json(settings, path: str, raw_dir: Path, name: str, params: dict | None = None, ttl_days: float = 7) -> dict:
    p = http_get(f"{API}/{path}", raw_dir=raw_dir, name=name, params=params, headers=_headers(settings), ext="json", ttl_days=ttl_days)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        p.unlink(missing_ok=True)
        raise SourceUnavailable(f"FCC BDC sent something other than JSON for {path}") from exc
    if not isinstance(doc, dict) or str(doc.get("status_code", 200)) != "200":
        p.unlink(missing_ok=True)
        raise SourceUnavailable(f"FCC BDC refused {path}: {str(doc)[:200]}")
    return doc


def latest_as_of(settings, raw_dir: Path) -> str:
    doc = _get_json(settings, "listAsOfDates", raw_dir, "fcc_bdc_dates", ttl_days=1)
    dates = sorted(str(d.get("as_of_date"))[:10] for d in doc.get("data", []) if str(d.get("data_type", "availability")).lower() == "availability")
    if not dates:
        raise SourceUnavailable(f"FCC BDC lists no availability dates: {str(doc)[:200]}")
    return dates[-1]


def list_files(settings, raw_dir: Path, as_of: str, provider_id: str, state_fips: str) -> list[dict]:
    """Mobile coverage files one provider filed for one state. The API filters by technology_type (not subcategory);
    the rows are filtered again here in case a filter is ignored."""
    params = {"category": "Provider", "technology_type": "Mobile Broadband", "provider_id": provider_id, "state_fips": state_fips}
    doc = _get_json(settings, f"downloads/listAvailabilityData/{as_of}", raw_dir, "fcc_bdc_list", params=params)
    rows = doc.get("data", []) or []
    mine = [r for r in rows if str(r.get("provider_id", provider_id)) == str(provider_id) and str(r.get("state_fips", state_fips)).zfill(2) == state_fips
            and "mobile" in str(r.get("technology_type", "mobile")).lower()]
    if not mine:
        console.log(f"[yellow]FCC BDC: no mobile files for provider {provider_id} in state {state_fips} ({len(rows)} rows listed"
                    + (f"; first: {json.dumps(rows[0])[:300]}" if rows else "") + ")")
    return mine


def classify(row: dict) -> tuple[str | None, str]:
    """(technology '4G'/'5G' or None, environment 'invehicle'/'stationary') for a listed file, from its fields and name."""
    text = " ".join(str(v) for v in row.values()).lower()
    code = str(row.get("technology_code", ""))
    tech = "5G" if code == "500" or "5g" in text else "4G" if code == "400" or "4g" in text or "lte" in text else None
    if tech == "5G" and re.search(r"35\s*/\s*3", text):
        tech = None                           # 7/1 coverage already contains 35/3
    env = "invehicle" if re.search(r"in[\s_-]?vehicle|mobile[\s_-]?env", text) else "stationary"
    return tech, env


def _file_type_rank(row: dict) -> int:
    """Lower is better: the operator's own coverage polygons (GeoPackage, then shapefile, then file geodatabase), then
    the hexagon files (H3 cells, as polygons or as a CSV of cell ids); 9 = unusable."""
    t = (str(row.get("file_type", "")) + " " + str(row.get("file_name", ""))).lower()
    hexagon = "hexagon" in str(row.get("subcategory", "")).lower()
    fmt = 0 if "gpkg" in t or "geopackage" in t else 1 if "shp" in t or "shape" in t else 2 if "gdb" in t else 3 if hexagon and "csv" in t else 9
    return 9 if fmt == 9 else fmt + (4 if hexagon else 0)


def _download(settings, raw_dir: Path, row: dict) -> Path:
    fid = row.get("file_id")
    last: Exception | None = None
    for suffix in ("", "/1", "/2", "/3"):
        try:
            p = http_get(f"{API}/downloads/downloadFile/availability/{fid}{suffix}", raw_dir=raw_dir, name="fcc_bdc_files", headers=_headers(settings),
                         ext="zip", timeout=1800, ttl_days=60)
        except SourceUnavailable as exc:
            last = exc
            continue
        if zipfile.is_zipfile(p):
            return p
        last = SourceUnavailable(f"file {fid}{suffix} is not a zip: {p.read_bytes()[:160]!r}")
        p.unlink(missing_ok=True)
    raise SourceUnavailable(f"FCC BDC download failed for file {fid} ({last})")


def _clip(path: Path, bbox: tuple[float, float, float, float]):
    """The part of a coverage file inside bbox: a GeoDataFrame of polygons, or a DataFrame with an `h3` column of cell ids."""
    import geopandas as gpd
    from shapely.geometry import box

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        inner = [n for n in names if n.lower().endswith((".gpkg", ".shp"))]
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if not inner and csvs:
            return _clip_h3(pd.read_csv(z.open(csvs[0]), dtype=str), bbox)
    if not inner:
        raise SourceUnavailable(f"{path.name} holds no GeoPackage, shapefile or hexagon CSV ({', '.join(names[:5])})")
    mask = gpd.GeoSeries([box(*bbox)], crs=4326)
    gdf = gpd.read_file(f"zip://{path.as_posix()}!{inner[0]}", bbox=mask)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)[["geometry"]]


def _clip_h3(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    import h3

    col = next((c for c in df.columns if "h3" in c.lower()), None)
    if col is None:
        col = next((c for c in df.columns if df[c].astype(str).str.fullmatch(r"[0-9a-f]{15}").mean() > 0.9), None)
    if col is None:
        raise SourceUnavailable(f"hexagon CSV has no H3 cell column ({', '.join(df.columns[:8])})")
    ids = df[col].dropna().astype(str).str.lower().unique()
    ll = np.array([h3.cell_to_latlng(c) for c in ids])
    lon0, lat0, lon1, lat1 = bbox
    keep = (ll[:, 1] >= lon0) & (ll[:, 1] <= lon1) & (ll[:, 0] >= lat0) & (ll[:, 0] <= lat1)
    return pd.DataFrame({"h3": ids[keep]})


def _hits(clip, lon: np.ndarray, lat: np.ndarray, pts) -> np.ndarray:
    """Indices of the samples inside a clipped coverage file."""
    import geopandas as gpd

    if not len(clip):
        return np.array([], dtype=int)
    if "h3" in clip.columns:
        import h3

        res = h3.get_resolution(str(clip["h3"].iloc[0]))
        cells = np.array([h3.latlng_to_cell(a, o, res) for a, o in zip(lat, lon)])
        return np.flatnonzero(np.isin(cells, clip["h3"].to_numpy()))
    return np.asarray(gpd.sjoin(pts, clip, predicate="within", how="inner").index.unique(), dtype=int)


def coverage_flags(settings, samples: pd.DataFrame, raw_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """{network id: {'invehicle_5g': bool array per sample, ...}} for every network with an FCC provider id."""
    import geopandas as gpd

    cfg = settings.networks.get("fcc_bdc") or {}
    states = [str(s).upper() for s in settings.route.get("fcc_states") or []]
    if not states:
        raise SourceUnavailable("route file lists no fcc_states")
    as_of = cfg.get("as_of") or latest_as_of(settings, raw_dir)
    lon, lat = samples["longitude"].to_numpy(float), samples["latitude"].to_numpy(float)
    pad = 0.05
    bbox = (lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    clips = raw_dir / "fcc_bdc"
    clips.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, np.ndarray]] = {}
    for op in settings.operators:
        pid = str(op.get("fcc_provider_id") or "")
        if not pid:
            continue
        flags = {k: np.zeros(len(samples), dtype=bool) for k in ("invehicle_5g", "invehicle_4g", "stationary_5g", "stationary_4g")}
        found = 0
        for st in states:
            rows = list_files(settings, raw_dir, as_of, pid, FIPS[st])
            best: dict[str, dict] = {}
            for r in rows:
                tech, env = classify(r)
                if tech is None or _file_type_rank(r) >= 9:
                    continue
                k = f"{env}_{tech.lower()}"
                if k not in best or _file_type_rank(r) < _file_type_rank(best[k]):
                    best[k] = r
            if not best and rows:
                kinds = sorted({(str(r.get("subcategory")), str(r.get("file_type")), str(r.get("technology_code_desc")), str(r.get("file_name"))[:60]) for r in rows})
                console.log(f"[yellow]FCC BDC: no usable file for {op['name']} in {st}; listed: {kinds[:8]}")
            for k, r in best.items():
                clip = clips / f"{as_of}_{r.get('file_id')}.parquet"
                if clip.exists():
                    c = pd.read_parquet(clip, columns=None)
                    if "h3" not in c.columns:
                        c = gpd.read_parquet(clip)
                else:
                    z = _download(settings, raw_dir.parent / "shared", r)
                    c = _clip(z, bbox)
                    c.to_parquet(clip)
                    z.unlink(missing_ok=True)          # keep only the corridor clip
                flags[k][_hits(c, lon, lat, pts)] = True
                found += 1
            if best:
                console.log(f"FCC BDC {st} {op['name']}: " + "; ".join(f"{k} <- {r.get('subcategory')} {r.get('file_type')} {str(r.get('file_name'))[:70]}" for k, r in sorted(best.items())))
        if found:
            out[op["id"]] = flags
            console.log(f"FCC BDC {as_of}: {op['name']} in-vehicle 4G on {flags['invehicle_4g'].mean():.0%} of the route, "
                        f"5G on {flags['invehicle_5g'].mean():.0%}; stationary 4G on {flags['stationary_4g'].mean():.0%}")
    if not out:
        raise SourceUnavailable("FCC BDC had no coverage files for these networks and states")
    return out


def fcc_prior(settings, samples: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Coverage prior rows (join_coverage's columns) from the FCC filings; networks without filings get no rows."""
    scores = {**DEFAULT_SCORES, **((settings.networks.get("fcc_bdc") or {}).get("scores") or {})}
    frames = []
    for pid, f in coverage_flags(settings, samples, raw_dir).items():
        s = np.full(len(samples), scores["none"])
        for k in ("stationary_4g", "stationary_5g", "invehicle_4g", "invehicle_5g"):     # ascending: the best one present wins
            s = np.where(f[k], np.maximum(s, scores[k]), s)
        five = f["invehicle_5g"] | f["stationary_5g"]
        frames.append(pd.DataFrame({
            "sample_id": samples["sample_id"].to_numpy(), "provider_id": pid, "prior_score": s.astype(np.float32),
            "level_4g": np.where(f["invehicle_4g"], 2.0, np.where(f["stationary_4g"], 1.0, 0.0)),
            "level_5g": np.where(f["invehicle_5g"], 2.0, np.where(f["stationary_5g"], 1.0, 0.0)),
            "radio_technology": np.where(five, "5G", "4G"), "source": SOURCE, "postcode": None, "postcode_distance_m": np.nan,
        }))
    return pd.concat(frames, ignore_index=True)


def probe(settings, raw_dir: Path, provider_id: str = "130077", state: str = "NY") -> dict:
    as_of = latest_as_of(settings, raw_dir)
    rows = list_files(settings, raw_dir, as_of, provider_id, FIPS[state.upper()])
    return {"as_of": as_of, "files": len(rows), "first": rows[:6], "classified": [classify(r) for r in rows[:20]]}
