"""Generate isolated, offline Arrow fixtures for real-browser tests."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcs import config
from tcs.model.simulate import simulate
from tcs.pipeline.export import export_all
from tcs.pipeline.join_cells import candidate_cells, corridor_cells, serving_cells
from tcs.pipeline.join_coverage import coverage_prior
from tcs.pipeline.movement import movement
from tcs.pipeline.obstruction import enrich_terrain
from tcs.pipeline.sample_route import build_route


def main():
    with tempfile.TemporaryDirectory(prefix="tcs-browser-") as tmp:
        root = Path(tmp)
        config.RAW, config.INTERIM, config.PROCESSED = root / "raw", root / "interim", root / "processed"
        s = config.load_settings(offline=True)
        s.route["sample_spacing_m"] = 2000
        s.terrain["horizon_azimuths"] = 8
        s.terrain["horizon_reach_m"] = 1000
        b = build_route(s)
        b.samples = enrich_terrain(s, b)
        prior, b.samples = coverage_prior(s, b)
        cells = corridor_cells(s, b, b.samples)
        serving = serving_cells(s, b.samples, candidate_cells(s, b.samples, cells))
        b.samples, b.stations = movement(s, b.samples, b.stations)
        obs, rc = simulate(s, b.samples, prior, serving)
        export_all(s, b, b.samples, b.stations, cells, obs, rc, prior)
        dest = config.ROOT / "web/tests/.generated"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("meta.json", "route.arrow", "cells.arrow"):
            shutil.copyfile(s.paths()["web"] / name, dest / name)


if __name__ == "__main__":
    main()
