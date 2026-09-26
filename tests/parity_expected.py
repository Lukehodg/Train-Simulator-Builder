"""Expected outputs for the browser-model parity check (web/tests/model-parity.mjs).

Re-simulates a built route with the Python model (tcs/model/*) for every link policy x vehicle profile x weather and
writes the per-sample results that web/src/sim/model.ts must reproduce from the exported bundle. Build the route with a
plain `tcs run` first (CI uses `tcs run --offline`).

    python tests/parity_expected.py [--route <route_id>] [--out <file.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from tcs.config import load_settings
from tcs.model import calibration
from tcs.model.bonding import POLICIES
from tcs.model.simulate import simulate
from tcs.model.wifi import CLASSES
from tcs.pipeline.sample_route import load_bundle


def main() -> int:
    ap = argparse.ArgumentParser(description="Python-model outputs for the browser parity check")
    ap.add_argument("--route", default=None, help="route id (default: config/route.yaml)")
    ap.add_argument("--out", type=Path, default=None, help="output JSON (default: data/processed/<route>/parity_expected.json)")
    args = ap.parse_args()

    s = load_settings(route_id=args.route)
    interim, web = s.paths()["interim"], s.paths()["web"]
    if not (web / "meta.json").exists():
        sys.exit(f"route '{s.route_id}' has not been built: run `tcs run --route {s.route_id}` first")
    meta = json.loads((web / "meta.json").read_text(encoding="utf-8"))
    # The browser model takes every parameter from meta.sim; a bundle built with a preset, a train design or an older
    # simulation.yaml would be compared against the wrong Python configuration.
    if meta["sim"] != s.sim:
        sys.exit("meta.json was built with different simulation settings (preset, train design or an edited "
                 "simulation.yaml): rebuild the route with a plain `tcs run` first")

    b = load_bundle(interim, s.route["country"])
    prior = pd.read_parquet(interim / "coverage_prior.parquet")
    serving = pd.read_parquet(interim / "serving.parquet")
    cal = calibration.load(interim / "calibration.json")
    order = [p["id"] for p in meta["providers"]]
    bit = {pid: 1 << k for k, pid in enumerate(order)}
    weathers = sorted({w for p in s.starlink["satcom"]["providers"] for w in p["availability"]["weather"]} | {"nominal"})

    scenarios = []
    for policy in POLICIES:
        for vehicle in s.sim["vehicle"]["profiles"]:
            for weather in weathers:
                s.sim["vehicle"]["profile"] = vehicle
                _, rc = simulate(s, b.samples, prior, serving, calibration=cal, weather=weather, policy=policy)
                scenarios.append({
                    "policy": policy, "vehicle": vehicle, "weather": weather,
                    # float32 -> float64 before rounding, or the JSON keeps float32 artefacts (765.2064819335938)
                    "bonded": rc["bonded_capacity_mbps"].astype(float).round(4).tolist(),
                    "wifi": rc["wifi_service_score"].astype(float).round(4).tolist(),
                    "conf": rc["confidence"].astype(float).round(4).tolist(),
                    "cls": [CLASSES.index(c) for c in rc["service_class"]],
                    "active": [sum(bit[p] for p in a.split("+")) if a else 0 for a in rc["active_links"]],
                })
    out = args.out or (s.paths()["processed"] / "parity_expected.json")
    out.write_text(json.dumps({"route_id": s.route_id, "n": len(b.samples), "providers": order, "scenarios": scenarios}), encoding="utf-8")
    print(f"{len(scenarios)} scenarios x {len(b.samples)} samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
