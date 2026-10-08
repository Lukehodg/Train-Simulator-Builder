"""US route bundles kept between publishes, so a monthly publish rebuilds only the US routes that changed.

A US route takes no Ofcom data and its feeds (FCC coverage, USGS lidar, OSM track) change slowly, but building all of
them costs about 2,500 GitHub runner minutes, more than a private repository's monthly allowance. So each bundle built
on main is kept as an asset of the `us-bundles` release, named after the route and a fingerprint of what it was built
from (the route file, the US profile and the model settings), and a publish reuses it until that fingerprint changes or
the bundle is older than MAX_AGE_DAYS. A code change that should reach every US route is pushed through with the
workflow's `rebuild_us` input.

    python -m tcs.bundle_store name ROUTE_ID          # the asset name a build of the route is stored under
    python -m tcs.bundle_store plan IDS_JSON ASSETS_JSON [--force]
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from .config import CONFIG_DIR, list_routes

RELEASE_TAG = "us-bundles"
MAX_AGE_DAYS = 183           # FCC coverage files come out twice a year
INPUTS = ["countries/US", "simulation.yaml", "starlink.yaml", "report.yaml"]


def fingerprint(route_file: Path, config_dir: Path = CONFIG_DIR) -> str:
    """12 hex digits over the route file and the shared settings a US build reads."""
    h = hashlib.sha256()
    files = [Path(route_file)]
    for rel in INPUTS:
        p = config_dir / rel
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p] if p.exists() else []
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def asset_name(route_id: str, fp: str) -> str:
    return f"{route_id}--{fp}.tar.gz"


def plan(route_ids: list[str], files: dict[str, Path], assets: list[dict], now: dt.datetime, *, force: bool = False,
         config_dir: Path = CONFIG_DIR) -> tuple[list[str], list[str]]:
    """(route ids to build, asset names to reuse). assets: the release's assets as {name, createdAt}."""
    stored = {a["name"]: dt.datetime.fromisoformat(a["createdAt"].replace("Z", "+00:00")) for a in assets}
    build, reuse = [], []
    for rid in route_ids:
        name = asset_name(rid, fingerprint(files[rid], config_dir))
        made = stored.get(name)
        if force or made is None or now - made > dt.timedelta(days=MAX_AGE_DAYS):
            build.append(rid)
        else:
            reuse.append(name)
    return build, reuse


def _files() -> dict[str, Path]:
    return {r["id"]: Path(r["file"]) for r in list_routes()}


def main(argv: list[str]) -> int:
    if argv[:1] == ["name"] and len(argv) == 2:
        print(asset_name(argv[1], fingerprint(_files()[argv[1]])))
        return 0
    if argv[:1] == ["plan"] and len(argv) in (3, 4):
        build, reuse = plan(json.loads(argv[1]), _files(), json.loads(argv[2]), dt.datetime.now(dt.timezone.utc),
                            force="--force" in argv[3:])
        print(json.dumps({"build": build, "reuse": reuse}))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
