"""Build a self-contained distribution of the viewer + Train Builder.

Produces dist/train-link-simulator-<date>.zip containing:
  site/                 the built viewer, the Train Builder, every route bundle, example designs
  Start Simulator.bat   Windows launcher (PowerShell static server; nothing to install)
  start-simulator.sh    macOS / Linux launcher (python3 -m http.server)
  README.txt            what it is, how to run it, what the figures mean
  reports/              any evidence packs generated for the included routes (optional)

The recipient needs no Python, no Node and no network connection except for the basemap and terrain tiles.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

from .config import ROOT
from .sources.base import console

README = """Train Link Simulator
====================

A time-aware simulation of onboard train connectivity along real UK railway routes: each cellular operator and
the satellite link are estimated every 50 m, combined by the onboard link manager, and turned into a predicted
passenger Wi-Fi experience.

HOW TO RUN
----------
Windows:        double-click "Start Simulator.bat"
macOS / Linux:  run ./start-simulator.sh from a terminal

A browser window opens automatically. Keep the launcher window open while you use the simulator; closing it
stops the local server. Nothing is installed and nothing leaves your machine except requests for the background
map and terrain tiles (CARTO, AWS Open Data) - if you are offline the map tiles will be blank but every ribbon,
panel and figure still works.

WHAT YOU CAN DO
---------------
* Pick a route from the dropdown beside the title ({n_routes} UK routes included).
* Press play, scrub the timeline, jump to a station, switch the camera (chase / oblique / free / route).
* Colour the ribbons by signal quality, throughput, latency, packet loss, availability, Wi-Fi class or
  CONFIDENCE - the last one shows how much of the estimate rests on measured, predicted or synthetic data.
* Switch scenario presets: baseline, or the EDGE Rail 5G active antenna with Fleet Connect aggregating every
  cellular network and the satellite link at once. KPIs at the bottom compare against the baseline.
* Open the Train Builder (top right) to design a consist - carriages, EDGE Rail roof units, satcom terminals,
  switches, access points, Fleet Connect - save it as a .train.json file, then load it in the Train tab to see
  what that architecture delivers along the route.
* Click any ribbon to inspect a single 50 m sample: carrier values, estimated serving cell, obstruction state,
  provenance and the model reasoning behind each number.
* Export the on-screen scenario as CSV + JSON with the Export button.

HOW TO READ THE FIGURES
-----------------------
These are MODEL PREDICTIONS, not measurements of a deployed system. Every sample carries a confidence value:

  0.90-1.00  directly measured / well validated
  0.70-0.89  strong source coverage + calibrated model
  0.40-0.69  prediction with partial infrastructure support
  0.10-0.39  sparse data / synthetic estimate
  0.00       unknown

The Sources tab states exactly which inputs produced the bundle you are looking at. Ofcom coverage is
operator-predicted rather than measured; OpenCellID is community-contributed and a missing cell does not mean
there is no service; there is no public route-level satellite RF telemetry, so that link is modelled from sky
visibility. Throughput depends on network load and spectrum as well as signal strength.

Build: {built}  |  model version {model_version}  |  routes: {routes}
"""


def build_web(web_dir: Path) -> Path:
    """npm run build -> web/dist (includes public/: data bundles, Train Builder, examples)."""
    # Resolve the executable instead of using shell=True: with a list argv, POSIX shells run a bare `npm` and drop
    # the arguments. which() finds npm.cmd on Windows, so no shell is needed on any platform.
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm not found: install Node.js, or pass --skip-build to package an existing web/dist")
    if not (web_dir / "node_modules").exists():
        console.log("installing web dependencies (npm install)…")
        subprocess.run([npm, "install"], cwd=web_dir, check=True)
    console.log("building the viewer (npm run build)…")
    subprocess.run([npm, "run", "build"], cwd=web_dir, check=True)
    dist = web_dir / "dist"
    if not (dist / "index.html").exists():
        raise RuntimeError("web build produced no index.html")
    return dist


def package(out_dir: Path | None = None, include_reports: bool = True, zip_it: bool = True, skip_build: bool = False) -> Path:
    web = ROOT / "web"
    dist = (web / "dist") if skip_build else build_web(web)
    out_dir = out_dir or (ROOT / "dist")
    stamp = date.today().isoformat()
    stage = out_dir / f"train-link-simulator-{stamp}"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "site").mkdir(parents=True)
    console.log(f"staging -> {stage}")
    shutil.copytree(dist, stage / "site", dirs_exist_ok=True)

    index = stage / "site" / "data" / "index.json"
    routes = []
    if index.exists():
        routes = [r["id"] for r in json.loads(index.read_text(encoding="utf-8")).get("routes", [])]
    model_version = "unknown"
    metas = sorted((stage / "site" / "data").glob("*/meta.json"))
    if metas:
        model_version = json.loads(metas[0].read_text(encoding="utf-8")).get("model_version", "unknown")

    for name in ("Start Simulator.bat", "Start-Simulator.ps1", "start-simulator.sh"):
        src = ROOT / "packaging" / name
        if src.exists():
            shutil.copy(src, stage / name)
    (stage / "start-simulator.sh").chmod(0o755) if (stage / "start-simulator.sh").exists() else None

    if include_reports:
        for rep in (ROOT / "data" / "processed").glob("*/reports"):
            rid = rep.parent.name
            if routes and rid not in routes:
                continue
            for f in rep.glob("evidence_*"):
                dest = stage / "reports" / rid
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest / f.name)

    (stage / "README.txt").write_text(
        README.format(n_routes=len(routes) or "the", built=stamp, model_version=model_version, routes=", ".join(routes) or "(none built)"),
        encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 1e6
    console.log(f"staged {size_mb:.0f} MB, {len(routes)} routes")
    if not zip_it:
        return stage
    archive = shutil.make_archive(str(stage), "zip", root_dir=stage.parent, base_dir=stage.name)
    zip_mb = Path(archive).stat().st_size / 1e6
    console.log(f"[bold]{archive}[/bold] ({zip_mb:.0f} MB)")
    return Path(archive)
