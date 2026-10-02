"""Antenna measurements out of vendor PDF reports: peak gain against frequency, horizontal and vertical pattern cuts,
and port-to-port isolation (S21), summarised per UK band as the gain a train's roof antenna has towards the masts.

Written for HUBER+SUHNER "HS Antenna Viewer" reports (one PDF per chart, printed as vector graphics), but nothing is
tied to their page layout: every chart is calibrated from its own axis labels, and the two pattern cuts are told apart
by their legend entries. Point it at the report folder or the zip it came in:

    python aa/antenna_patterns.py data/aa/vendor/PDF_File.zip --out data/aa/antenna

The inputs and everything this writes are vendor data: keep them out of git (data/aa/ is ignored); see aa/README.md.

What it writes (CSV):
  gain_<port>.csv        peak gain (dBi) against frequency, per port
  cuts.csv               per port and measured frequency: the elevation of the strongest direction, the vertical cut
                         near the horizon, the horizontal cut's average and deepest null
  horizon_by_band.csv    per UK band: peak gain; gain towards the horizon averaged around the train, per port and with
                         the strongest of the ports taken in each direction; the worst direction
  isolation.csv          per port pair and UK band: the weakest isolation (dB) in the band

Reading the cuts. Each cut is scaled to its own maximum (0 dB), so a cut is not gain until it is anchored. The
vertical cut's maximum is taken as the port's peak gain; the horizontal cut (the horizon plane) is then anchored to
the vertical cut where the two cross, straight ahead and straight behind. Expect about +-2 dB on the horizon figures,
and none of this accounts for the train's own roof.
"""
from __future__ import annotations

import argparse
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# UK bands (downlink, MHz): the range checked for isolation, and the measured frequencies that stand for the band
UK_BANDS = {
    "700": ((758, 788), (758,)),
    "800": ((791, 821), (790, 824)),
    "900": ((925, 960), (925, 960)),
    "1400": ((1452, 1492), (1450, 1500)),
    "1800": ((1805, 1880), (1785, 1850)),
    "2100": ((2110, 2170), (2100, 2155)),
    "2600": ((2620, 2690), (2600, 2700)),
    "3400-3800 (n78)": ((3410, 3800), (3400, 3500, 3600)),
}
NEAR_HORIZON_DEG = 10.0          # masts seen from a train sit within a few degrees of the horizon
GREY = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------- the maths (no PDFs involved)
def lin_mean_db(db) -> float:
    """Average in power, returned in dB."""
    return float(10 * np.log10(np.mean(10 ** (np.asarray(db, dtype=float) / 10))))


def linear_map(values, positions) -> tuple[float, float]:
    """value = a * position + b, least squares over every axis label."""
    a, b = np.polyfit(np.asarray(positions, dtype=float), np.asarray(values, dtype=float), 1)
    return float(a), float(b)


def wrap180(deg):
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


@dataclass
class Cut:
    """A pattern cut: angle (deg, as the chart labels it) and level (dB, 0 = the cut's own maximum)."""
    deg: np.ndarray
    db: np.ndarray

    def at(self, angle: float) -> float:
        o = np.argsort(self.deg)
        return float(np.interp(angle, self.deg[o], self.db[o], period=360))


def horizon_gain(peak_dbi: float, vertical: Cut, horizontal: Cut, az=np.arange(-180.0, 180.0, 1.0)) -> np.ndarray:
    """Gain (dBi) towards the horizon around the train, from one port's peak gain and its two cuts.

    Vertical cut: 0 deg = zenith, +-90 = the horizon straight ahead (+90) and behind (-90). Horizontal cut: the horizon
    plane, 0 deg = ahead, 180 = behind. The vertical cut's maximum is the peak gain; the horizontal cut is shifted so it
    meets the vertical cut where they cross."""
    hrel = np.array([horizontal.at(a) for a in az])
    offset = np.mean([vertical.at(90) - horizontal.at(0), vertical.at(-90) - horizontal.at(180)])
    return peak_dbi + hrel + offset


def band_summary(ports: dict[str, tuple[float, Cut, Cut]]) -> dict:
    """Per measured frequency: peak gain, and horizon gain per port and with the strongest port per direction."""
    az = np.arange(-180.0, 180.0, 1.0)
    cuts = {p: horizon_gain(peak, v, h, az) for p, (peak, v, h) in ports.items()}
    allp = np.vstack(list(cuts.values()))
    best = allp.max(axis=0)
    return {"peak_gain_dbi": float(np.mean([peak for peak, _, _ in ports.values()])),
            "horizon_mean_per_port_dbi": float(np.mean([lin_mean_db(c) for c in cuts.values()])),
            "horizon_worst_per_port_dbi": float(np.mean([c.min() for c in cuts.values()])),
            "horizon_best_port_mean_dbi": lin_mean_db(best),
            "horizon_best_port_worst_dbi": float(best.min())}


# ---------------------------------------------------------------- reading the PDFs
def _colour(obj) -> tuple[float, ...] | None:
    c = obj.get("stroking_color")
    if c is None or isinstance(c, str):
        return None
    c = tuple(float(v) for v in (c if isinstance(c, (tuple, list)) else (c,)))
    return c * 3 if len(c) == 1 else c[:3]


def _same(c, rgb, tol=0.15) -> bool:
    return c is not None and len(c) == 3 and all(abs(a - b) <= tol for a, b in zip(c, rgb))


def _numbers(page) -> list[tuple[float, float, float, float]]:
    """(value, x centre, y centre, right edge) of every number printed on the page, once each (some reports print
    every label twice); y grows downwards."""
    out = {}
    for w in page.dedupe_chars().extract_words(keep_blank_chars=False, use_text_flow=False):   # labels printed twice
        t = w["text"].replace("−", "-")
        if re.fullmatch(r"-?\d+(\.\d+)?", t):
            xc, yc = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
            out.setdefault((t, round(xc), round(yc)), (float(t), xc, yc, float(w["x1"])))
    return list(out.values())


def _paths(page, rgb) -> list[np.ndarray]:
    """Polylines in one colour: each as an (n, 2) array of x, y (y downwards)."""
    out = []
    for o in page.lines + page.curves:
        if _same(_colour(o), rgb) and o.get("pts"):
            out.append(np.array([(float(x), float(y)) for x, y in o["pts"]]))
    return out


def _groups(vals: list[tuple[float, float]], gap_factor: float = 2.5) -> list[list[tuple[float, float]]]:
    """Split axis labels (value, position along the axis) into runs: a gap well beyond the usual spacing starts a new
    chart's axis (two charts stacked on a page share a label column)."""
    vals = sorted(vals, key=lambda t: t[1])
    if len(vals) < 3:
        return [vals]
    gaps = np.diff([p for _, p in vals])
    usual = float(np.median(gaps)) or 1.0
    out, cur = [], [vals[0]]
    for g, v in zip(gaps, vals[1:]):
        if g > gap_factor * usual:
            out.append(cur)
            cur = []
        cur.append(v)
    out.append(cur)
    return out


def chart_axes(page, curve: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """(x -> value) and (y -> value) maps for the Cartesian chart a curve is drawn in, from the labels on its left and
    below it. A curve may run off the chart (drawn on, clipped by the page), so it is located by its median point."""
    nums = _numbers(page)
    xm, ym = np.median(curve, axis=0)
    x0 = float(np.percentile(curve[:, 0], 1))
    # y axis: the label column nearest the curve on its left (right-aligned labels, so grouped by their right edge),
    # the run of it level with the curve
    cols: dict[int, list] = {}
    for v, _, y, xr in nums:
        if xr < x0:
            cols.setdefault(round(xr / 3), []).append((v, y))
    ycol = None
    for key in sorted(cols, key=lambda k: -k):                      # nearest column first
        for run in _groups(cols[key]):
            if len(run) >= 3 and min(p for _, p in run) <= ym <= max(p for _, p in run):
                ycol = run
                break
        if ycol:
            break
    if not ycol:
        raise ValueError("could not find the chart's y-axis labels")
    # x axis: the first label row at or below the y axis's bottom label, the run of it under the curve
    bottom = max(p for _, p in ycol) - 2
    rows: dict[int, list] = {}
    for v, x, y, _ in nums:
        if y >= bottom:
            rows.setdefault(round(y / 3), []).append((v, x))
    xrow = None
    for key in sorted(rows):
        for run in _groups(rows[key]):
            if len(run) >= 3 and min(p for _, p in run) <= xm <= max(p for _, p in run):
                xrow = run
                break
        if xrow:
            break
    if not ycol or not xrow:
        raise ValueError("could not find the chart's x-axis labels")
    return linear_map(*zip(*xrow)), linear_map(*zip(*ycol))


def read_xy(pdf_path: Path, rgb: tuple[float, float, float]) -> pd.DataFrame:
    """The points of the curve drawn in `rgb`, in the chart's own units (x, y), sorted by x."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        paths = [p for p in _paths(page, rgb) if len(p) >= 2]
        curve = np.vstack([p for p in paths if not _legend_sample(p)]) if paths else None
        if curve is None or not len(curve):
            raise ValueError(f"{pdf_path.name}: no curve in that colour")
        (ax, bx), (ay, by) = chart_axes(page, curve)
    pts = np.unique(np.round(curve, 3), axis=0)
    return pd.DataFrame({"x": ax * pts[:, 0] + bx, "y": ay * pts[:, 1] + by}).sort_values("x").reset_index(drop=True)


def _legend_sample(p: np.ndarray) -> bool:
    """A legend's short sample line: two points, level."""
    return len(p) == 2 and abs(p[0, 1] - p[1, 1]) < 0.01


def read_cuts(pdf_path: Path) -> tuple[Cut, Cut]:
    """(vertical, horizontal) cuts of a pattern report: two polar charts, each with grey rings, a dB scale along a
    radius and angle labels round the outside, told apart by their legend ("Vertical" / "Horizontal")."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        words = page.dedupe_chars().extract_words()
        legend = {}
        for w in words:
            name = "vertical" if w["text"].lower().startswith("vertical") else "horizontal" if w["text"].lower().startswith("horizontal") else None
            if name:
                cy = (w["top"] + w["bottom"]) / 2
                samples = [o for o in page.lines + page.curves if o.get("pts") and len(o["pts"]) == 2
                           and abs(float(o["pts"][0][1]) - cy) < 4 and float(o["pts"][1][0]) <= w["x0"] + 1]
                if samples:
                    legend[name] = _colour(min(samples, key=lambda o: w["x0"] - float(o["pts"][1][0])))
        rings = [o for o in page.curves if _same(_colour(o), GREY, 0.05)]
        nums = _numbers(page)
        cuts = {}
        for name, rgb in legend.items():
            curve = max((p for p in _paths(page, rgb) if len(p) > 20), key=len)
            cx0, cy0 = curve.mean(axis=0)
            around = [r for r in rings if r["x0"] <= cx0 <= r["x1"] and r["top"] <= cy0 <= r["bottom"]]
            outer = max(around, key=lambda r: r["x1"] - r["x0"])
            cx, cy, R = (outer["x0"] + outer["x1"]) / 2, (outer["top"] + outer["bottom"]) / 2, (outer["x1"] - outer["x0"]) / 2
            ring = [r for r in rings if math.hypot((r["x0"] + r["x1"]) / 2 - cx, (r["top"] + r["bottom"]) / 2 - cy) < 0.02 * R]
            # dB scale: the labels along the radius right of the centre name the rings (and the centre) they sit beside;
            # the rings, not the labels, carry the scale (a label is printed a little inside its ring)
            radii = sorted({0.0} | {round((r["x1"] - r["x0"]) / 2, 1) for r in ring})
            scale = [(v, min(radii, key=lambda rr: abs(rr - (x - cx)))) for v, x, y, _ in nums
                     if abs(y - cy) < 0.08 * R and -0.15 * R <= x - cx <= R * 1.02 and v <= 0]
            a, b = linear_map(*zip(*scale))                           # dB = a * r + b
            # angle labels: just outside the outer ring
            ang = {v: math.degrees(math.atan2(y - cy, x - cx)) for v, x, y, _ in nums if 1.03 * R < math.hypot(x - cx, y - cy) < 1.35 * R}
            phi0, phi90 = ang[0.0], ang[90.0]
            sign = 1.0 if wrap180(phi90 - phi0) > 0 else -1.0
            dx, dy = curve[:, 0] - cx, curve[:, 1] - cy
            cuts[name] = Cut(deg=wrap180(sign * (np.degrees(np.arctan2(dy, dx)) - phi0)), db=a * np.hypot(dx, dy) + b)
    return cuts["vertical"], cuts["horizontal"]


# ---------------------------------------------------------------- a whole report
def _find(root: Path, *patterns: str) -> list[Path]:
    out = []
    for pat in patterns:
        out += sorted(root.rglob(pat))
    return out


def summarise(root: Path, out: Path) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    gains = {}
    for f in _find(root, "*Cellular*_gain.pdf", "*cellular*_gain.pdf"):
        port = re.search(r"(?i)cellular\s*(\d+)", f.name).group(1)
        g = read_xy(f, (1.0, 0.0, 0.0)).rename(columns={"x": "mhz", "y": "gain_dbi"})
        g.round(2).to_csv(out / f"gain_cellular{port}.csv", index=False)
        gains[port] = g
    if not gains:
        raise FileNotFoundError(f"no cellular gain charts (*Cellular*_gain.pdf) under {root}")

    rows, by_freq = [], {}
    for f in _find(root, "*_cellular*.pdf"):
        m = re.search(r"_(\d+)_cellular(\d+)\.pdf$", f.name, re.I)
        if not m or m.group(2) not in gains:
            continue
        mhz, port = int(m.group(1)), m.group(2)
        v, h = read_cuts(f)
        peak = float(np.interp(mhz, gains[port].mhz, gains[port].gain_dbi))
        near = np.abs(np.abs(v.deg) - 90) <= NEAR_HORIZON_DEG                 # within 10 deg of the horizon, ahead and behind
        rows.append({"port": port, "mhz": mhz, "peak_gain_dbi": peak,
                     "elevation_of_max_deg": 90 - abs(float(v.deg[np.argmax(v.db)])),
                     "vertical_near_horizon_db": lin_mean_db(v.db[near]),
                     "horizontal_mean_db": lin_mean_db(h.db), "horizontal_deepest_null_db": float(h.db.min())})
        by_freq.setdefault(mhz, {})[port] = (peak, v, h)
    cuts = pd.DataFrame(rows).sort_values(["mhz", "port"])
    cuts.round(2).to_csv(out / "cuts.csv", index=False)

    bands = []
    for band, (_, freqs) in UK_BANDS.items():
        for mhz in freqs:
            if mhz in by_freq:
                bands.append({"band": band, "mhz": mhz, **band_summary(by_freq[mhz])})
    horizon = pd.DataFrame(bands)
    if len(horizon):
        horizon = horizon.groupby("band", sort=False).mean(numeric_only=True).drop(columns="mhz").reset_index()
    horizon.round(1).to_csv(out / "horizon_by_band.csv", index=False)

    iso = []
    for f in _find(root, "*Cell * and Cell *.pdf"):
        pair = "-".join(re.search(r"Cell (\d+) and Cell (\d+)", f.name).groups())
        s = read_xy(f, (0.0, 0.5, 0.0))                              # S21 is drawn in green
        s = s.assign(mhz=s.x * (1000 if s.x.max() < 100 else 1))     # GHz axes
        for band, ((lo, hi), _) in UK_BANDS.items():
            w = s[(s.mhz >= lo) & (s.mhz <= hi)]
            if len(w):
                iso.append({"pair": pair, "band": band, "isolation_db": -float(w.y.max())})
    isolation = pd.DataFrame(iso)
    isolation.round(1).to_csv(out / "isolation.csv", index=False)
    return {"gains": pd.concat(gains, names=["port"]).reset_index(level=0), "cuts": cuts, "horizon": horizon, "isolation": isolation}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path, help="folder of the vendor's PDFs, or the zip they came in")
    ap.add_argument("--out", type=Path, default=Path("data/aa/antenna"), help="output folder (keep it out of git)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = a.report
        if root.suffix.lower() == ".zip":
            zipfile.ZipFile(root).extractall(tmp)
            root = Path(tmp)
        r = summarise(root, a.out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(r["horizon"].round(1).to_string(index=False))
        if len(r["isolation"]):
            print(r["isolation"].pivot(index="band", columns="pair", values="isolation_db").reindex(list(UK_BANDS)).round(1).to_string())
    print(f"-> {a.out}/")


if __name__ == "__main__":
    main()
