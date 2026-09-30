"""Ofcom predicted coverage as the cellular prior (UK adapter).

Two routes to the same normalised table (sample_id, provider_id, prior_score, level_4g, level_5g, ...):

1. Ofcom API portal, product "Mobile Coverage UPRN (Basic)" -> "Ofcom Mobile Checker UPRN Coverage API"
   (subscription key). One call per corridor postcode returns every UPRN in it with a blended 4G+5G 0-4 level per
   MNO (Mc_EE / Mc_TH / Mc_O2 / Mc_VO); we aggregate the UPRNs (median by default) into the postcode prior and keep
   the spread as an uncertainty signal. `tcs probe-ofcom N1C4TB` prints the raw payload.
2. Connected Nations postcode open data (no key): % of premises with outdoor 4G/5G per operator.

Postcodes along the corridor come from OS Code-Point Open (OS OpenData, keyless download), one per route sample.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..geo import Projector, nearest_sample_index
from .base import Provenance, SourceUnavailable, console, http_get, now_iso

OFCOM_MOBILE_URL = "https://apim-cnapi-shared-prod.azure-api.net/mobilechecker/UPRN/{postcode}"
CODEPOINT_URL = "https://api.os.uk/downloads/v1/products/CodePointOpen/downloads?area=GB&format=CSV&redirect"


# ---------------------------------------------------------------- postcodes along the corridor
def corridor_postcodes(samples: pd.DataFrame, proj: Projector, raw_dir: Path, *, max_distance_m: float = 400, offline: bool = False) -> pd.DataFrame:
    """One nearest postcode per sample (deduplicated) from Code-Point Open. Columns: postcode, x, y, sample_id, dist_m."""
    zpath = http_get(CODEPOINT_URL, raw_dir=raw_dir.parent / "shared", name="codepoint_open", ext="zip", offline=offline, ttl_days=365)
    frames = []
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "data/csv/" in n.lower().replace("\\", "/")]
        if not names:
            raise SourceUnavailable("Code-Point Open zip has no Data/CSV files")
        minx, maxx = samples["x"].min() - 5000, samples["x"].max() + 5000
        miny, maxy = samples["y"].min() - 5000, samples["y"].max() + 5000
        for n in names:
            with zf.open(n) as fh:
                df = pd.read_csv(fh, header=None, usecols=[0, 2, 3], names=["postcode", "x", "y"], dtype={"postcode": str})
            df = df[(df.x >= minx) & (df.x <= maxx) & (df.y >= miny) & (df.y <= maxy)]
            if len(df):
                frames.append(df)
    if not frames:
        raise SourceUnavailable("no Code-Point Open postcodes inside the corridor bbox")
    pcs = pd.concat(frames, ignore_index=True)
    # Code-Point is BNG; if the route CRS is not BNG (non-GB route) this adapter does not apply.
    if proj.crs.to_epsg() != 27700:
        raise SourceUnavailable("Code-Point Open is GB-only")
    idx, dist = nearest_sample_index(samples["x"].values, samples["y"].values, pcs["x"].values, pcs["y"].values)
    pcs["sample_id"] = samples["sample_id"].values[idx]
    pcs["dist_m"] = dist
    pcs = pcs[pcs.dist_m <= max_distance_m]
    # keep the closest postcode per sample, then map every sample to its nearest kept postcode
    best = pcs.sort_values("dist_m").drop_duplicates("sample_id")
    s_idx, s_dist = nearest_sample_index(best["x"].values, best["y"].values, samples["x"].values, samples["y"].values)
    out = pd.DataFrame({
        "sample_id": samples["sample_id"].values,
        "postcode": best["postcode"].values[s_idx],
        "postcode_distance_m": s_dist.astype(np.float32),
    })
    return out


# ---------------------------------------------------------------- Ofcom API
# Contract (Ofcom API portal, "Ofcom Mobile Checker UPRN Coverage API", June 2025 methodology):
#   GET https://apim-cnapi-shared-prod.azure-api.net/mobilechecker/UPRN/{PostCode}
#   -> [{"PostCode": "...", "Availability": [{"uprn": 1234, "postcode": "...", "Mc_EE": 4, "Mc_TH": 3, "Mc_O2": 4, "Mc_VO": 2, ...}, ...]}]
#   One blended 4G+5G value per MNO: 0 poor/none · 1 variable outdoor · 2 good outdoor · 3 variable in-home, good outdoor
#   · 4 good in-home and outdoor. Field keys in the sample payload carry stray spaces, so keys are normalised.
def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _parse_payload(payload, operators: list[dict]) -> dict[str, dict]:
    """-> {provider_id: {"levels": [per-UPRN values], "n_uprn": int}} from one postcode response."""
    docs = payload if isinstance(payload, list) else [payload]
    rows: list[dict] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        av = doc.get("Availability") or doc.get("availability")
        if isinstance(av, list):
            rows.extend(r for r in av if isinstance(r, dict))
        elif any(_norm_key(k).startswith("mc") for k in doc):
            rows.append(doc)
    out: dict[str, dict] = {}
    for op in operators:
        keys = {("mc" + _norm_key(k)) for k in (op.get("ofcom_keys") or [op.get("ofcom_key", op["name"])])}
        levels: list[int] = []
        for r in rows:
            for k, v in r.items():
                if _norm_key(k) in keys:
                    try:
                        levels.append(int(v))
                    except (TypeError, ValueError):
                        pass
        out[op["id"]] = {"levels": levels, "n_uprn": len(rows)}
    return out


def _aggregate(levels: list[int], how: str) -> float | None:
    if not levels:
        return None
    arr = np.asarray(levels, dtype=float)
    return float({"median": np.median, "max": np.max, "min": np.min, "mean": np.mean}.get(how, np.median)(arr))


def fetch_ofcom_api(postcodes: pd.DataFrame, settings, raw_dir: Path, *, limit: int | None = None) -> pd.DataFrame:
    key = settings.key("OFCOM_API_KEY")
    if not key:
        raise SourceUnavailable("OFCOM_API_KEY not set")
    cfg = settings.networks["cellular"]
    ops = settings.operators
    level_map = {int(k): float(v) for k, v in cfg["ofcom_level_to_score"].items()}
    agg = cfg.get("ofcom_uprn_aggregate", "median")
    url_tpl = cfg.get("ofcom_api_url", OFCOM_MOBILE_URL)
    header = cfg.get("ofcom_api_key_header", "Ocp-Apim-Subscription-Key")
    uniq = postcodes["postcode"].dropna().unique().tolist()
    if limit:
        uniq = uniq[:limit]
    console.log(f"Ofcom API: {len(uniq)} postcodes to query (cached responses are reused; product limit 100/min)")
    rows = []
    missing = 0
    for n_done, pc in enumerate(uniq):
        norm = pc.replace(" ", "").upper()
        try:
            p = http_get(url_tpl.format(postcode=norm), raw_dir=raw_dir, name="ofcom_api", ext="json",
                         headers={header: key, "Accept": "application/json"}, offline=settings.offline, ttl_days=90)
        except SourceUnavailable as exc:
            if re.search(r"HTTP (401|403|429)", str(exc)):
                # A refusal (call quota used up, or a bad key) is not "no data for this postcode": carrying on would
                # fill the rest of the route with neutral stand-ins labelled as Ofcom. Stop; the answers fetched so
                # far are cached, so a run after the quota resets picks up where this one stopped.
                raise SourceUnavailable(f"Ofcom API refused the request after {n_done} of {len(uniq)} postcodes "
                                        f"(call quota used up, or the key is wrong): {exc}") from exc
            missing += 1
            if missing <= 5:
                console.log(f"[yellow]Ofcom API {pc}: {exc}")
            continue
        with open(p, "r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError:
                continue
        for pid, info in _parse_payload(payload, ops).items():
            lv = _aggregate(info["levels"], agg)
            spread = (max(info["levels"]) - min(info["levels"])) if info["levels"] else np.nan
            xs = sorted(level_map)
            score = float(np.interp(lv, xs, [level_map[k] for k in xs])) if lv is not None else np.nan
            rows.append({"postcode": pc, "provider_id": pid, "level_4g": lv, "level_5g": np.nan, "level_spread": spread, "n_uprn": info["n_uprn"],
                         "radio_technology": "blend", "prior_score": score})
        if n_done and n_done % 200 == 0:
            console.log(f"Ofcom API: {n_done}/{len(uniq)} postcodes")
    if not rows:
        raise SourceUnavailable("Ofcom API returned no parsable Mc_* fields (run `tcs probe-ofcom`)")
    df = pd.DataFrame(rows)
    df["source"] = "ofcom_predicted"
    Provenance("ofcom_api", url_tpl, now_iso(), notes="Ofcom Mobile Checker UPRN Coverage API: blended 4G/5G 0-4 level per MNO, "
               f"{agg} over the UPRNs of each corridor postcode; postcode found for {len(uniq) - missing}/{len(uniq)} queries.").write(raw_dir / "ofcom_api" / "provenance.json")
    return df


# ---------------------------------------------------------------- Connected Nations open data
def fetch_connected_nations(postcodes: pd.DataFrame, settings, raw_dir: Path) -> pd.DataFrame:
    cfg = settings.networks["cellular"]["connected_nations"]
    if not cfg.get("enabled", True):
        raise SourceUnavailable("connected_nations disabled")
    zpath = http_get(cfg["url"], raw_dir=raw_dir, name="connected_nations", ext="zip", offline=settings.offline, ttl_days=365)
    want = set(postcodes["postcode"].str.replace(" ", "").str.upper())
    ops = {op["id"]: [op.get("ofcom_key", op["name"])] + op.get("ofcom_keys", []) for op in settings.operators}
    pat = re.compile(r"^(?P<gen>4G|5G)_prem_out_(?P<op>[A-Za-z0-9]+)$", re.I)
    frames = []
    with zipfile.ZipFile(zpath) as zf:
        for n in zf.namelist():
            if not n.lower().endswith(".csv"):
                continue
            with zf.open(n) as fh:
                df = pd.read_csv(fh, dtype=str, low_memory=False)
            pc_col = next((c for c in df.columns if c.lower() in {"postcode", "pcd", "pcds", "postcode_sector"}), None)
            if pc_col is None:
                continue
            df["_pc"] = df[pc_col].str.replace(" ", "").str.upper()
            df = df[df["_pc"].isin(want)]
            if df.empty:
                continue
            for c in df.columns:
                m = pat.match(c)
                if not m:
                    continue
                pid = next((k for k, keys in ops.items() if m.group("op").lower() in {x.lower() for x in keys}), None)
                if not pid:
                    continue
                sub = df[["_pc", c]].rename(columns={"_pc": "postcode_norm", c: "pct"})
                sub["provider_id"] = pid
                sub["gen"] = m.group("gen").upper()
                frames.append(sub)
    if not frames:
        raise SourceUnavailable("Connected Nations: no matching postcode rows/columns (check column_pattern)")
    long = pd.concat(frames, ignore_index=True)
    long["pct"] = pd.to_numeric(long["pct"], errors="coerce")
    wide = long.pivot_table(index=["postcode_norm", "provider_id"], columns="gen", values="pct", aggfunc="max").reset_index()
    wide["level_4g"] = np.nan
    wide["level_5g"] = np.nan
    g5 = wide["5G"] if "5G" in wide else pd.Series(np.nan, index=wide.index)
    g4 = wide["4G"] if "4G" in wide else pd.Series(np.nan, index=wide.index)
    wide["radio_technology"] = np.where(g5.fillna(0) >= 50, "5G", "4G")
    wide["prior_score"] = (np.where(g5.fillna(0) >= 50, 0.85 * g5 / 100, 0.75 * g4 / 100)).astype(float)
    pcmap = postcodes.assign(postcode_norm=postcodes["postcode"].str.replace(" ", "").str.upper())[["postcode", "postcode_norm"]].drop_duplicates()
    out = wide.merge(pcmap, on="postcode_norm", how="left").drop(columns=["postcode_norm"])
    out["source"] = "ofcom_connected_nations"
    return out[["postcode", "provider_id", "level_4g", "level_5g", "radio_technology", "prior_score", "source"]]


def probe(postcode: str, settings, raw_dir: Path) -> dict:
    key = settings.key("OFCOM_API_KEY")
    if not key:
        raise SourceUnavailable("OFCOM_API_KEY not set")
    cfg = settings.networks["cellular"]
    p = http_get(cfg.get("ofcom_api_url", OFCOM_MOBILE_URL).format(postcode=postcode.replace(" ", "").upper()), raw_dir=raw_dir, name="ofcom_probe", ext="json",
                 headers={cfg.get("ofcom_api_key_header", "Ocp-Apim-Subscription-Key"): key, "Accept": "application/json"}, ttl_days=0)
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)
