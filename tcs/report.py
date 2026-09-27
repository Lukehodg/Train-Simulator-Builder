"""Evidence pack for tender submissions: a Word report (charts, KPIs, station-to-station tables, method, provenance,
validation status) and an Excel data appendix, for one route and one scenario.

Everything in it is a model prediction unless the validation section says otherwise; the wording is deliberate.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings

WIFI_CLASSES = ["EXCELLENT", "GOOD", "USABLE", "POOR", "OUTAGE"]
CLASS_COLORS = {"EXCELLENT": "#1e7d3e", "GOOD": "#3f9a52", "USABLE": "#c9a100", "POOR": "#d6531a", "OUTAGE": "#9c1b2c"}

# Route heat maps: five bands per metric on one ordinal ramp, darker = worse, so weak stretches stand out. The ramp is
# a single hue with monotone lightness and its light end at 2.1:1 on white (checked with an OKLab validator), so it
# survives greyscale printing and colour-vision deficiency; the legends name every band in words and numbers.
BANDS = ["Excellent", "Good", "Fair", "Poor", "Very poor"]
BAND_COLORS = ["#e8a577", "#db773c", "#c24c0c", "#9d2c04", "#6f1a11"]
HEAT_METRICS = {   # band edges, best first; industry-conventional RSRP bands for LTE/5G
    "signal": {"title": "Signal strength", "sub": "strongest mobile network at the train (RSRP)",
               "ranges": ["≥ −80 dBm", "−90 to −80 dBm", "−100 to −90 dBm", "−110 to −100 dBm", "< −110 dBm or none"]},
    "throughput": {"title": "Throughput", "sub": "combined onboard WAN capacity",
                   "ranges": ["≥ 200 Mbps", "100 to 200 Mbps", "50 to 100 Mbps", "10 to 50 Mbps", "< 10 Mbps or no link"]},
    "latency": {"title": "Latency", "sub": "combined onboard WAN round trip",
                "ranges": ["< 40 ms", "40 to 60 ms", "60 to 100 ms", "100 to 150 ms", "≥ 150 ms or no link"]},
}
INK, MUTED, HAIRLINE = "#1f2933", "#5d6c7b", "#dde3e9"


# ---------------------------------------------------------------- analysis
def section_table(samples: pd.DataFrame, rc: pd.DataFrame, obs: pd.DataFrame, stations: pd.DataFrame, providers: list[str]) -> pd.DataFrame:
    """Station-to-station sections (between consecutive stations in the sequence)."""
    st = stations.sort_values("distance_m").reset_index(drop=True)
    rc = rc.set_index("sample_id")
    rows = []
    wide = obs.pivot_table(index="sample_id", columns="provider_id", values="capacity_mbps")
    avail = obs.pivot_table(index="sample_id", columns="provider_id", values="available", aggfunc="max")
    for k in range(len(st) - 1):
        a, b = int(st.loc[k, "sample_id"]), int(st.loc[k + 1, "sample_id"])
        if b <= a:
            continue
        seg = rc.loc[a:b]
        s = samples.iloc[a:b + 1]
        cls = seg["service_class"].value_counts(normalize=True)
        worst = avail.loc[a:b].mean().idxmin() if len(avail) else ""
        rows.append({
            "section": f"{st.loc[k, 'crs']} → {st.loc[k + 1, 'crs']}", "from": st.loc[k, "name"], "to": st.loc[k + 1, "name"],
            "length_km": round((s["distance_m"].iloc[-1] - s["distance_m"].iloc[0]) / 1000, 1),
            "minutes": round((s["sim_seconds"].iloc[-1] - s["sim_seconds"].iloc[0]) / 60, 1),
            "mean_speed_kph": round(float(s["speed_kph"].mean()), 0),
            "bonded_mean_mbps": round(float(seg["bonded_capacity_mbps"].mean()), 0),
            "bonded_p10_mbps": round(float(seg["bonded_capacity_mbps"].quantile(0.1)), 0),
            "bonded_min_mbps": round(float(seg["bonded_capacity_mbps"].min()), 0),
            "per_user_mean_mbps": round(float(seg["per_user_mbps"].mean()), 2),
            "latency_mean_ms": round(float(seg["effective_latency_ms"].mean()), 0),
            "streaming_share_pct": round(100 * (cls.get("EXCELLENT", 0) + cls.get("GOOD", 0)), 1),
            "usable_share_pct": round(100 * (1 - cls.get("OUTAGE", 0) - cls.get("POOR", 0)), 1),
            "outage_km": round(float(np.sum(np.diff(s["distance_m"]) * (seg["service_class"].iloc[:-1].to_numpy() == "OUTAGE"))) / 1000, 2),
            "tunnels": int(((s["in_tunnel"].astype(int).diff() == 1).sum()) + (1 if s["in_tunnel"].iloc[0] else 0)),
            "weakest_link": str(worst),
            **{f"{p}_avail_pct": round(100 * float(avail.loc[a:b, p].mean()), 1) for p in providers if p in avail},
            **{f"{p}_p50_mbps": round(float(wide.loc[a:b, p].median()), 0) for p in providers if p in wide},
            "confidence": round(float(seg["confidence"].mean()), 2),
        })
    return pd.DataFrame(rows)


def link_table(obs: pd.DataFrame, spacing_m: float) -> pd.DataFrame:
    rows = []
    for pid, g in obs.groupby("provider_id"):
        av = g[g["available"]]
        rows.append({"link": pid, "type": g["provider_type"].iloc[0], "availability_pct": round(100 * g["available"].mean(), 1),
                     "p50_capacity_mbps": round(float(av["capacity_mbps"].median()), 0) if len(av) else 0, "p10_capacity_mbps": round(float(av["capacity_mbps"].quantile(0.1)), 0) if len(av) else 0,
                     "mean_latency_ms": round(float(av["latency_ms"].mean()), 0) if len(av) else np.nan, "handover_events": int(g["handover"].sum()),
                     "unavailable_km": round(float((~g["available"]).sum() * spacing_m / 1000), 1), "mean_confidence": round(float(g["confidence"].mean()), 2),
                     "sources": " | ".join(sorted({f for x in g["source_flags"].astype(str).head(2000) for f in x.split("|")}))})
    return pd.DataFrame(rows)


def kpis(rc: pd.DataFrame, samples: pd.DataFrame, spacing_m: float) -> dict:
    cls = rc["service_class"].value_counts(normalize=True)
    return {
        "route_length_km": round(float(samples["distance_m"].max()) / 1000, 1),
        "journey_minutes": round(float(samples["sim_seconds"].max()) / 60, 0),
        "streaming_share_pct": round(100 * (cls.get("EXCELLENT", 0) + cls.get("GOOD", 0)), 1),
        "usable_share_pct": round(100 * (1 - cls.get("OUTAGE", 0) - cls.get("POOR", 0)), 1),
        "outage_share_pct": round(100 * cls.get("OUTAGE", 0), 2),
        "outage_km": round(float(np.sum(np.diff(samples["distance_m"]) * (rc["service_class"].iloc[:-1].to_numpy() == "OUTAGE"))) / 1000, 1),
        "bonded_mean_mbps": round(float(rc["bonded_capacity_mbps"].mean()), 0),
        "bonded_median_mbps": round(float(rc["bonded_capacity_mbps"].median()), 0),
        "bonded_p10_mbps": round(float(rc["bonded_capacity_mbps"].quantile(0.1)), 0),
        "per_user_mean_mbps": round(float(rc["per_user_mbps"].mean()), 2),
        "latency_mean_ms": round(float(rc["effective_latency_ms"].mean()), 0),
        "wifi_score_mean": round(float(rc["wifi_service_score"].mean()), 0),
        "mean_confidence": round(float(rc["confidence"].mean()), 2),
        "class_share": {c: round(100 * float(cls.get(c, 0)), 1) for c in WIFI_CLASSES},
    }


def rsrp_band(dbm: np.ndarray) -> np.ndarray:
    """0 = Excellent … 4 = Very poor; no signal (NaN) is Very poor."""
    return np.select([dbm >= -80, dbm >= -90, dbm >= -100, dbm >= -110], [0, 1, 2, 3], 4)


def heat_bands(samples: pd.DataFrame, rc: pd.DataFrame, obs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-sample band (0 = Excellent … 4 = Very poor) for each HEAT_METRICS entry, in route order."""
    sid = samples["sample_id"].to_numpy()
    cell = obs[obs["provider_type"] == "cellular"]
    best = cell.groupby("sample_id")["signal_primary"].max().reindex(sid).to_numpy(float)
    r = rc.set_index("sample_id").reindex(sid)
    cap, lat = r["bonded_capacity_mbps"].to_numpy(float), r["effective_latency_ms"].to_numpy(float)
    return {"signal": rsrp_band(best),
            "throughput": np.select([cap >= 200, cap >= 100, cap >= 50, cap >= 10], [0, 1, 2, 3], 4),
            "latency": np.select([lat < 40, lat < 60, lat < 100, lat < 150], [0, 1, 2, 3], 4)}   # NaN latency = no link


def typical_bands(bands: np.ndarray, n_runs: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the route into up to n_runs equal runs of samples; each run takes the band that at least half of it reaches
    (the upper median), so charts at page scale show sustained stretches without one 50 m sample colouring a kilometre.
    Returns (run start indices plus the end, band per run)."""
    edges = np.linspace(0, len(bands), min(n_runs, len(bands)) + 1).astype(int)
    typical = np.array([np.sort(bands[a:b])[(b - a) // 2] for a, b in zip(edges[:-1], edges[1:])], dtype=int)
    return edges, typical


def outage_stretches(samples: pd.DataFrame, rc: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Continuous stretches in the OUTAGE class, longest first: where they start, how long, between which stations, in which tunnel."""
    cols = ["from_km", "length_km", "from", "to", "tunnel"]
    cls = rc.set_index("sample_id").reindex(samples["sample_id"])["service_class"].to_numpy()
    out = (cls == "OUTAGE").astype(np.int8)
    d = samples["distance_m"].to_numpy(float)
    step = float(np.median(np.diff(d))) if len(d) > 1 else 0.0
    edges = np.flatnonzero(np.diff(np.r_[0, out, 0]))
    st = stations.sort_values("distance_m")
    names, sd = st["name"].tolist(), st["distance_m"].to_numpy(float)
    tun = samples["in_tunnel"].to_numpy(bool)
    tname = samples["tunnel_name"] if "tunnel_name" in samples else pd.Series([None] * len(samples))
    rows = []
    for a, b in zip(edges[::2], edges[1::2]):
        k = max(int(np.searchsorted(sd, d[a], side="right")) - 1, 0) if len(sd) else 0
        found = [str(t) for t in pd.unique(tname.iloc[a:b].dropna()) if str(t).strip()]
        where = ""
        if tun[a:b].any():
            where = (found[0] if "tunnel" in found[0].lower() else f"{found[0]} tunnel") if len(found) == 1 else "tunnels"
        rows.append({"from_km": round(d[a] / 1000, 2), "length_km": round((d[b - 1] - d[a] + step) / 1000, 2),
                     "from": names[k] if names else "", "to": names[min(k + 1, len(names) - 1)] if names else "", "tunnel": where})
    return pd.DataFrame(rows, columns=cols).sort_values("length_km", ascending=False, kind="stable").reset_index(drop=True)


def band_shares(bands: np.ndarray) -> list[float]:
    """Share of the route (%) in each band; samples are evenly spaced, so a share of samples is a share of length."""
    return list(np.bincount(bands, minlength=len(BANDS)) / max(len(bands), 1) * 100)


# ---------------------------------------------------------------- charts
ACCENT_HEX = "#00707c"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": HAIRLINE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 8.5, "axes.titlecolor": INK, "legend.fontsize": 7, "legend.labelcolor": INK})
    return plt


def _png(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06); plt.close(fig)
    return buf.getvalue()


def _station_axis(ax, stations: pd.DataFrame, km_max: float) -> None:
    """Calling points along the top edge, with hairlines down the plot."""
    stops = stations[stations["stop"]] if "stop" in stations else stations
    d = np.minimum(stops["distance_m"].to_numpy() / 1000, km_max)
    for x in d:
        ax.axvline(x, color=HAIRLINE, linewidth=0.7, zorder=0)
    top = ax.secondary_xaxis("top")
    top.set_xticks(d); top.set_xticklabels(stops["crs"], fontsize=6, rotation=90, color=MUTED)
    top.tick_params(length=0)
    top.spines["top"].set_visible(False)


def _chart_capacity(samples, rc, stations) -> bytes:
    """Combined throughput along the route: per-km median with the 10th-90th percentile band (11,000 raw points would be noise)."""
    plt = _plt()
    km = samples["distance_m"].to_numpy() / 1000
    cap = pd.Series(rc.set_index("sample_id").reindex(samples["sample_id"])["bonded_capacity_mbps"].to_numpy(float))
    step = float(np.median(np.diff(samples["distance_m"]))) if len(samples) > 1 else 50.0
    win_km = max(1, int(round(float(km.max()) / 150)))         # about 150 windows across the page, whatever the route length
    win = max(1, int(round(win_km * 1000 / max(step, 1))))
    roll = cap.rolling(win, center=True, min_periods=1)
    med, lo, hi = roll.median(), roll.quantile(0.1), roll.quantile(0.9)
    ymax = max(float(np.nanmax(hi)) * 1.1, 10.0)
    fig, ax = plt.subplots(figsize=(10, 3.0), dpi=160)
    tun = samples["in_tunnel"].to_numpy(bool)
    if tun.any():
        ax.fill_between(km, 0, ymax, where=tun, color="#9aa5b1", alpha=0.4, linewidth=0, label="Tunnel", step="mid")
    per = "each km" if win_km == 1 else f"each {win_km} km"
    ax.fill_between(km, lo, hi, color=ACCENT_HEX, alpha=0.18, linewidth=0, label=f"10th–90th percentile over {per}")
    ax.plot(km, med, color=ACCENT_HEX, linewidth=1.1, label=f"Median over {per}")
    _station_axis(ax, stations, float(km.max()))
    ax.set_xlim(0, float(km.max())); ax.set_ylim(0, ymax)
    ax.set_xlabel("Distance from origin (km)"); ax.set_ylabel("Combined throughput (Mbps)")
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", ncol=3, frameon=True, framealpha=0.92, edgecolor="none")
    return _png(fig)


def _chart_sections(sec: pd.DataFrame) -> bytes:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 2.9), dpi=160)
    x = np.arange(len(sec))
    s, u = sec["streaming_share_pct"].to_numpy(float), sec["usable_share_pct"].to_numpy(float)
    kw = {"width": 0.72, "edgecolor": "white", "linewidth": 0.8}
    ax.bar(x, s, color=CLASS_COLORS["GOOD"], label="Video calls and streaming (EXCELLENT / GOOD)", **kw)
    ax.bar(x, u - s, bottom=s, color=CLASS_COLORS["USABLE"], label="Browsing and email (USABLE)", **kw)
    ax.bar(x, 100 - u, bottom=u, color=CLASS_COLORS["OUTAGE"], label="Poor or no service (POOR / OUTAGE)", **kw)
    ax.set_xticks(x); ax.set_xticklabels([str(v).replace(" → ", "–") for v in sec["section"]], rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("Share of section length (%)"); ax.set_ylim(0, 100); ax.set_xlim(-0.6, len(sec) - 0.4)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    return _png(fig)


def _chart_links(links: pd.DataFrame) -> bytes:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(links) + 0.9), dpi=160, sharey=True)
    y = np.arange(len(links))[::-1]
    for ax, col, title, fmt, xmax in ((axes[0], "availability_pct", "Availability (% of route)", "{:.0f} %", 100),
                                      (axes[1], "p50_capacity_mbps", "Median capacity when available (Mbps)", "{:.0f}", None)):
        v = links[col].fillna(0).to_numpy(float)
        ax.barh(y, v, color=ACCENT_HEX, height=0.62)
        top = xmax or max(float(v.max()) * 1.18, 1.0)
        ax.set_xlim(0, top)
        for yy, vv in zip(y, v):
            ax.text(vv + top * 0.01, yy, fmt.format(vv), va="center", fontsize=7, color=INK)
        ax.set_title(title, loc="left"); ax.grid(axis="x", color=HAIRLINE, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels(links["name"], fontsize=8, color=INK)
    fig.tight_layout()
    return _png(fig)


def _band_legend(ax, metric: str, bands: np.ndarray, ncol: int = 1) -> None:
    from matplotlib.patches import Patch

    shares = band_shares(bands)
    fmt = lambda v: "0 %" if v == 0 else ("<1 %" if v < 0.5 else f"{v:.0f} %")
    labels = [f"{b}  {r}  ·  {fmt(v)}" for b, r, v in zip(BANDS, HEAT_METRICS[metric]["ranges"], shares)]
    handles = [Patch(facecolor=c, edgecolor="none") for c in BAND_COLORS]
    ax.legend(handles, labels, loc="upper left", ncol=ncol, frameon=False, fontsize=6.5, handlelength=1.4, handleheight=0.9,
              borderaxespad=0, labelcolor=INK, title="Share of route length", title_fontsize=6.5, alignment="left")


def _scale_bar(ax) -> None:
    """A km scale bar in the lower-left corner (axes are in km, equal aspect)."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    target = (x1 - x0) * 0.22
    length = max([v for v in (1, 2, 5, 10, 20, 25, 50, 100, 200) if v <= target] or [1])
    bx, by = x0 + (x1 - x0) * 0.04, y0 + (y1 - y0) * 0.05
    ax.plot([bx, bx + length], [by, by], color=MUTED, linewidth=1.2, solid_capstyle="butt", zorder=4)
    ax.text(bx + length / 2, by + (y1 - y0) * 0.02, f"{length} km", ha="center", va="bottom", fontsize=6, color=MUTED, zorder=4)


def _chart_heatmap(samples: pd.DataFrame, stations: pd.DataFrame, bands: dict[str, np.ndarray]) -> bytes:
    """Three small-multiple maps of the railway (signal, throughput, latency), each coloured by band along the line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patheffects import withStroke

    lat0 = np.radians(float(samples["latitude"].mean()))
    kx, ky = 111.32 * np.cos(lat0), 110.57                    # equirectangular km: accurate to <1 % over a UK route
    x, y = samples["longitude"].to_numpy() * kx, samples["latitude"].to_numpy() * ky
    w, h = max(float(np.ptp(x)), 1.0), max(float(np.ptp(y)), 1.0)
    xy = np.column_stack([x, y])
    tall = h > 1.1 * w                                         # north-south route: maps side by side; otherwise stacked
    if tall:
        map_h = float(np.clip(3.0 * h / w, 2.6, 6.0))
        fig = plt.figure(figsize=(10, map_h + 1.9), dpi=160)
        gs = fig.add_gridspec(2, 3, height_ratios=[map_h, 1.05], left=0.02, right=0.98, top=1 - 0.5 / (map_h + 1.9), bottom=0.01, wspace=0.08, hspace=0.06)
        cells = [(gs[0, i], gs[1, i]) for i in range(3)]
    else:
        map_h = float(np.clip(6.4 * h / w, 1.5, 2.6))
        fig = plt.figure(figsize=(10, 3 * (map_h + 0.5)), dpi=160)
        gs = fig.add_gridspec(3, 2, width_ratios=[6.4, 3.0], left=0.02, right=0.98, top=1 - 0.45 / (3 * (map_h + 0.5)), bottom=0.01, hspace=0.32, wspace=0.04)
        cells = [(gs[i, 0], gs[i, 1]) for i in range(3)]
    # Page scale: a run of line is about 2 pt long, and neighbouring runs overlap by about 0.7 pt so anti-aliasing leaves no seams.
    box_w, box_h = (3.0, map_h) if tall else (6.3, map_h)      # map box in inches (approximate; the 6 % padding is below)
    pt_per_km = 72 * min(box_w / (w * 1.12), box_h / (h * 1.12))
    path_km = float(np.hypot(np.diff(x), np.diff(y)).sum())
    n_runs = int(np.clip(path_km * pt_per_km / 2.0, 50, 600))
    spacing_km = max(float(np.median(np.diff(samples["distance_m"].to_numpy()))) / 1000, 1e-3)
    overlap = max(1, int(np.ceil(0.7 / pt_per_km / spacing_km)))
    stops = stations[stations["stop"]] if "stop" in stations else stations
    sx, sy = stops["longitude"].to_numpy() * kx, stops["latitude"].to_numpy() * ky
    for (map_cell, legend_cell), (metric, spec) in zip(cells, HEAT_METRICS.items()):
        ax = fig.add_subplot(map_cell)
        ax.set_aspect("equal", adjustable="datalim")
        pad = 0.06 * max(w, h)
        ax.update_datalim([(x.min() - pad, y.min() - pad), (x.max() + pad, y.max() + pad)])
        ax.plot(x, y, color="#8b939c", linewidth=4.4, solid_capstyle="round", solid_joinstyle="round", zorder=1)   # casing: the palest band still reads
        edges, typical = typical_bands(bands[metric], n_runs)
        change = np.r_[True, typical[1:] != typical[:-1]]      # merge neighbouring runs in the same band
        starts, kinds = edges[:-1][change], typical[change]
        ends = np.r_[starts[1:], len(xy)]
        runs = [xy[max(a - overlap, 0):min(b + overlap, len(xy))] for a, b in zip(starts, ends)]
        order = np.argsort(kinds, kind="stable")               # worse runs on top where runs overlap
        ax.add_collection(LineCollection([runs[i] for i in order], colors=np.array(BAND_COLORS)[kinds[order]], linewidths=3.0,
                                         capstyle="butt", joinstyle="round", zorder=2))
        ax.scatter(sx, sy, s=14, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3)
        placed: list[tuple[float, float]] = []
        gap = 0.07 * max(w, h)
        for k in [0, len(stops) - 1, *range(1, len(stops) - 1)]:   # ends first, then intermediate stops that have room
            if k < 0 or any(np.hypot(sx[k] - px, sy[k] - py) < gap for px, py in placed):
                continue
            placed.append((sx[k], sy[k]))
            ax.annotate(stops["crs"].iloc[k], (sx[k], sy[k]), xytext=(4, 3), textcoords="offset points", fontsize=6, color=INK, zorder=4,
                        path_effects=[withStroke(linewidth=2.2, foreground="white")])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(HAIRLINE); sp.set_linewidth(0.6)
        ax.annotate("N", xy=(0.96, 0.96), xytext=(0, -16), xycoords="axes fraction", textcoords="offset points", ha="center", va="center",
                    fontsize=6.5, color=MUTED, arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8})
        ax.annotate(spec["title"], (0, 1), xytext=(0, 12), xycoords="axes fraction", textcoords="offset points", fontsize=9, fontweight="bold", color=INK, va="bottom")
        ax.annotate(spec["sub"], (0, 1), xytext=(0, 3), xycoords="axes fraction", textcoords="offset points", fontsize=6.5, color=MUTED, va="bottom")
        fig.canvas.draw()                                      # settle the equal-aspect limits before placing the scale bar
        _scale_bar(ax)
        lax = fig.add_subplot(legend_cell); lax.axis("off")
        _band_legend(lax, metric, bands[metric])
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def _chart_networks(samples: pd.DataFrame, stations: pd.DataFrame, obs: pd.DataFrame, names: dict[str, str], best: np.ndarray) -> bytes:
    """Heat strip: signal band of each mobile network (rows) along the route (columns), plus the strongest of them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    sid = samples["sample_id"].to_numpy()
    cell = obs[obs["provider_type"] == "cellular"]
    ids = list(dict.fromkeys(cell["provider_id"]))
    rows = [rsrp_band(cell[cell["provider_id"] == p].set_index("sample_id")["signal_primary"].reindex(sid).to_numpy(float)) for p in ids]
    rows.append(best)
    labels = [names.get(p, p) for p in ids] + ["Strongest"]
    grid = np.vstack([typical_bands(r, 500)[1] for r in rows])  # same smoothing as the maps: about 0.3 mm of print per column
    km = float(samples["distance_m"].max()) / 1000
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(rows) + 1.35), dpi=160)
    ax.imshow(grid, cmap=ListedColormap(BAND_COLORS), vmin=-0.5, vmax=len(BANDS) - 0.5, aspect="auto", interpolation="nearest", extent=[0, km, len(rows), 0])
    for i in range(1, len(rows)):
        ax.axhline(i, color="white", linewidth=2.2 if i == len(rows) - 1 else 1.2)   # a wider gap sets off the "Strongest" row
    ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels(labels, fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Distance from origin (km)", fontsize=7, color=MUTED); ax.tick_params(axis="x", labelsize=6.5, colors=MUTED)
    stops = stations[stations["stop"]] if "stop" in stations else stations
    ax.set_xlim(0, km)
    top = ax.secondary_xaxis("top")
    top.set_xticks(np.minimum(stops["distance_m"].to_numpy() / 1000, km)); top.set_xticklabels(stops["crs"], fontsize=5.5, rotation=90, color=MUTED)
    top.tick_params(length=2, colors=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for sp in top.spines.values():
        sp.set_visible(False)
    ax.legend([Patch(facecolor=c) for c in BAND_COLORS], [f"{b}  {r}" for b, r in zip(BANDS, HEAT_METRICS["signal"]["ranges"])], loc="upper center",
              bbox_to_anchor=(0.5, -0.32 if len(rows) > 3 else -0.5), ncol=5, frameon=False, fontsize=6.5, handlelength=1.4, labelcolor=INK)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08); plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- documents
def _plain(v) -> str:
    """Config values for people: dicts as 'key value, ...', lists joined, underscores dropped."""
    if isinstance(v, dict):
        return ", ".join(f"{str(k).replace('_', ' ')} {_plain(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(_plain(x) for x in v) or "none"
    return str(v)


def _assumptions(settings: Settings) -> list[tuple[str, str]]:
    from .report_docx import POLICIES, VEHICLES

    sim = settings.sim
    vname = sim["vehicle"]["profile"]
    vp = sim["vehicle"]["profiles"][vname]
    pw, wan, cell = sim["passenger_wifi"], sim["wan"], sim["cellular"]
    lo, hi = pw["load_factor_range"]
    rows = [
        ("Antenna and modem", f"{VEHICLES.get(vname, vname)}: link budget {float(vp.get('db_offset', 0)):+g} dB (quality {float(vp.get('score_offset', 0)):+g}), "
                              f"throughput ×{vp.get('capacity_factor', 1)}"),
        ("Link management", POLICIES.get(wan["policy"], wan["policy"])),
        ("Bonding efficiency × congestion", f"{wan['bonding_efficiency']} × {wan['congestion_factor']}"),
        ("Minimum link score to carry traffic", str(wan["minimum_link_score"])),
        ("Link score weights", _plain(wan["score_weights"])),
        ("Mobile capacity at excellent signal (Mbps)", "; ".join(f"{op.get('name', op['id'])}: {_plain(op['capacity_prior_mbps'])}" for op in settings.operators)),
        ("Roof-unit aggregation factor", str(cell.get("units_capacity_factor", 1.0))),
        ("Cutting loss", f"up to −{cell['terrain']['cutting_penalty_max']} quality at {cell['terrain']['cutting_full_depth_m']} m depth"),
        ("Distance from serving cell", f"−{cell['cell_distance']['penalty_per_km']} quality per km beyond {cell['cell_distance']['free_km']} km"),
        ("Handover", f"{cell['handover']['duration_samples']} points (≈{cell['handover']['duration_samples'] * settings.spacing_m:.0f} m), "
                     f"+{cell['handover']['latency_spike_ms']} ms, capacity ×{cell['handover']['capacity_factor']}"),
        ("Tunnels", f"quality {cell['tunnels']['default_score']} without in-tunnel coverage; {cell['tunnels']['das_score']} where it is assumed "
                    f"({_plain(cell['tunnels']['das_tunnels'])})"),
        ("Satellite", "; ".join(f"{p.get('name', p['id'])}: {p['terminal'].replace('_', ' ')} terminal, {p['capacity_prior_mbps'][p['terminal']]} Mbps, "
                                f"minimum elevation {p['min_elevation_deg'][p['terminal']]}°, sky-visibility threshold {p['availability']['sky_threshold']}"
                                for p in settings.starlink["satcom"]["providers"]) + ("" if sim.get("satcom_enabled", True) else " (not fitted in this design)")),
        ("Passenger demand", f"{pw['passengers']} seats, load factor {lo}–{hi} along the route, {pw['active_share'] * 100:.0f} % online, "
                             f"{pw['per_user_demand_mbps']} Mbps each; access points {pw['ap_capacity_mbps']} Mbps in total"),
        ("Service-class thresholds (score out of 100)", ", ".join(f"{c} ≥ {v}" for c, v in pw["classes"].items())),
        ("Timetable", f"departure {settings.route.get('timetable', {}).get('departure')}, {settings.route.get('timetable', {}).get('dwell_s')} s dwell at each stop"),
        ("Model version", str(sim.get("model_version", ""))),
    ]
    return rows


def write_xlsx(path: Path, ev, samples: pd.DataFrame, rc: pd.DataFrame, obs: pd.DataFrame, reference: str) -> None:
    """The data appendix: a read-me sheet, the headline measures, every section, link and 50 m point, assumptions and sources."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1B2A41")
    rule = Border(bottom=Side(style="thin", color="D5DBE1"))

    def sheet(name: str, df: pd.DataFrame, widths: dict[str, int] | None = None):
        ws = wb.create_sheet(name)
        ws.append(list(df.columns))
        for c in ws[1]:
            c.font, c.fill, c.alignment = head_font, head_fill, Alignment(vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 30
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v) for v in row])
        for i, col in enumerate(df.columns, 1):
            sample = [len(str(v)) for v in df[col].head(300)] or [0]
            ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(col, min(46, max(10, len(str(col)) * 0.9, max(sample) + 2)))
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = ws.dimensions
        return ws

    k, sh = ev.k, ev.shares
    ws = wb.active; ws.title = "Read me"
    ws["A1"] = "Onboard connectivity performance: data appendix"; ws["A1"].font = Font(bold=True, size=14, color="1B2A41")
    info = [("Route", f"{ev.route_name}: {ev.origin} to {ev.destination}"), ("Scenario", f"{ev.scenario_title} ({ev.scenario_detail})"),
            ("Document reference", reference), ("Date of issue", ev.generated.strftime("%d %B %Y").lstrip("0")),
            ("Prepared by", ev.info.get("prepared_by", "")), ("Classification", ev.info.get("classification", "")),
            ("Status", "Model predictions, not measurements of a deployed system. See the Word report for method, sources and confidence.")]
    r = 3
    for a, b in info:
        if b:
            ws.cell(r, 1, a).font = Font(bold=True, color="5D6C7B"); ws.cell(r, 2, b); r += 1
    r += 1
    ws.cell(r, 1, "Sheet").font = head_font; ws.cell(r, 1).fill = head_fill
    ws.cell(r, 2, "Contents").font = head_font; ws.cell(r, 2).fill = head_fill
    for a, b in [("Summary", "Headline predictions for the whole journey"), ("Sections", "Every station-to-station section"),
                 ("Links", "Each mobile network and the satellite link on its own"),
                 ("Samples", "Every point along the railway (one row per sample): position, time, speed, terrain, each link's quality, capacity, "
                             "latency, availability and signal, and the combined connection and passenger service"),
                 ("Assumptions", "Model parameters used for this scenario"), ("Sources", "Data sources and whether each was live"),
                 *([("Validation", "Agreement between predictions and field measurements")] if ev.validation is not None and len(ev.validation) else [])]:
        r += 1
        ws.cell(r, 1, a).border = rule; ws.cell(r, 2, b).border = rule
        ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 22; ws.column_dimensions["B"].width = 110

    summary = pd.DataFrame([
        ("Route length", k["route_length_km"], "km"), ("Journey time", k["journey_minutes"], "min"),
        ("Journey supporting video calls and streaming (EXCELLENT or GOOD)", k["streaming_share_pct"], "%"),
        ("Journey usable or better", k["usable_share_pct"], "%"), ("Predicted loss of service", k["outage_km"], "km"),
        ("Predicted loss of service, share of route", k["outage_share_pct"], "%"),
        ("Combined throughput, mean", k["bonded_mean_mbps"], "Mbps"), ("Combined throughput, median", k["bonded_median_mbps"], "Mbps"),
        ("Combined throughput exceeded over 90 % of the route (P10)", k["bonded_p10_mbps"], "Mbps"),
        ("Throughput per active passenger, mean", k["per_user_mean_mbps"], "Mbps"), ("Latency, mean when connected", k["latency_mean_ms"], "ms"),
        ("Journey with latency below 60 ms", round(sh["latency"][0] + sh["latency"][1], 1), "%"),
        ("Journey where the strongest network is Good or better (≥ −90 dBm)", round(sh["signal"][0] + sh["signal"][1], 1), "%"),
        ("Passenger Wi-Fi service score, mean", k["wifi_score_mean"], "0–100"), ("Mean confidence of the estimates", k["mean_confidence"], "0–1"),
        *[(f"Journey in the {c} class", v, "%") for c, v in k["class_share"].items()],
    ], columns=["Measure", "Prediction", "Unit"])
    sheet("Summary", summary, {"Measure": 64, "Prediction": 14, "Unit": 10})
    sec = ev.sec.rename(columns={"section": "Section", "from": "From", "to": "To", "length_km": "Length (km)", "minutes": "Time (min)",
                                 "mean_speed_kph": "Mean speed (km/h)", "bonded_mean_mbps": "Mean throughput (Mbps)", "bonded_p10_mbps": "P10 throughput (Mbps)",
                                 "bonded_min_mbps": "Minimum throughput (Mbps)", "per_user_mean_mbps": "Per passenger (Mbps)", "latency_mean_ms": "Latency (ms)",
                                 "streaming_share_pct": "Streaming-capable (%)", "usable_share_pct": "Usable or better (%)", "outage_km": "Loss of service (km)",
                                 "tunnels": "Tunnels", "weakest_name": "Least available link", "confidence": "Confidence"}).drop(columns=["weakest_link"])
    sheet("Sections", sec)
    links = ev.links.rename(columns={"name": "Link", "type": "Type", "availability_pct": "Available (%)", "p50_capacity_mbps": "Median capacity (Mbps)",
                                     "p10_capacity_mbps": "P10 capacity (Mbps)", "mean_latency_ms": "Latency (ms)", "handover_events": "Handovers",
                                     "unavailable_km": "Unavailable (km)", "mean_confidence": "Confidence", "sources": "Sources"}).drop(columns=["link"])
    sheet("Links", links[["Link", *[c for c in links.columns if c != "Link"]]])
    wide = obs.pivot_table(index="sample_id", columns="provider_id", values=["capacity_mbps", "latency_ms", "available", "quality_score"])
    wide.columns = [f"{p}_{m}" for m, p in wide.columns]
    cell = obs[obs["provider_type"] == "cellular"]
    rsrp = cell.pivot_table(index="sample_id", columns="provider_id", values="signal_primary")
    rsrp.columns = [f"{p}_rsrp_dbm" for p in rsrp.columns]
    rsrp["strongest_rsrp_dbm"] = rsrp.max(axis=1)
    smp = samples[["sample_id", "distance_m", "timestamp_sim", "latitude", "longitude", "speed_kph", "elevation_m", "in_tunnel", "cutting_depth_m", "sky_visibility", "station_nearby"]].copy()
    smp["timestamp_sim"] = pd.to_datetime(smp["timestamp_sim"]).dt.strftime("%H:%M:%S")
    smp = smp.merge(wide.reset_index(), on="sample_id", how="left").merge(rsrp.round(1).reset_index(), on="sample_id", how="left").merge(
        rc[["sample_id", "active_links", "bonded_capacity_mbps", "effective_latency_ms", "packet_loss_pct", "per_user_mbps", "wifi_service_score", "service_class", "confidence", "source_flags"]], on="sample_id", how="left")
    sheet("Samples", smp)
    sheet("Assumptions", pd.DataFrame(ev.assumptions, columns=["Parameter", "Value"]), {"Parameter": 40, "Value": 120})
    sheet("Sources", pd.DataFrame(ev.sources, columns=["Input", "Source", "Status"]), {"Input": 52, "Source": 60, "Status": 20})
    if ev.validation is not None and len(ev.validation):
        sheet("Validation", ev.validation)
    wb.properties.creator = ev.info.get("prepared_by") or "Train Link Simulator"
    wb.properties.title = f"Onboard connectivity performance: {ev.route_name}"
    wb.save(path)


# ---------------------------------------------------------------- entry point
def build_report(settings: Settings, meta: dict, samples: pd.DataFrame, obs: pd.DataFrame, rc: pd.DataFrame, stations: pd.DataFrame, out_dir: Path, scenario_label: str,
                 validation: pd.DataFrame | None = None, weather: str = "nominal", baseline: dict | None = None) -> dict[str, Path]:
    """baseline: {"title", "obs", "rc"} of the baseline configuration, when this scenario is something else (for the comparison)."""
    from .report_docx import POLICIES, VEHICLES, Evidence, write_docx

    out_dir.mkdir(parents=True, exist_ok=True)
    providers = list(obs["provider_id"].unique())
    spacing = float(settings.spacing_m)
    sim = settings.sim
    names = {op["id"]: op.get("name", op["id"]) for op in settings.operators} | {p["id"]: p.get("name", p["id"]) for p in settings.starlink["satcom"]["providers"]}
    k = kpis(rc, samples, spacing)
    sec = section_table(samples, rc, obs, stations, providers)
    sec["weakest_name"] = sec["weakest_link"].map(lambda x: names.get(x, x)) if len(sec) else []
    links = link_table(obs, spacing)
    links["name"] = links["link"].map(lambda x: names.get(x, x))
    links = links.sort_values(["type", "name"], key=lambda c: c.map({"cellular": 0}).fillna(1) if c.name == "type" else c).reset_index(drop=True)
    bands = heat_bands(samples, rc, obs)
    shares = {m: band_shares(b) for m, b in bands.items()}
    live = lambda ok: "live" if ok else "synthetic stand-in"
    cov = meta.get("coverage_sources", [])
    geometry = {"osm": "OpenStreetMap (ODbL), routed station to station", "file": "Route file supplied", "synthetic_route": "Approximate line through the stations (stand-in)"}
    sources = [
        ("Route centreline, stations, tunnels, cuttings, line speed", geometry.get(meta["geometry_source"], meta["geometry_source"]), live(meta["geometry_source"] in ("osm", "file"))),
        ("Terrain and sky visibility", {"copernicus_glo30": "Copernicus DEM GLO-30 (30 m)", "synthetic_terrain": "Flat stand-in terrain"}.get(meta["terrain_source"], meta["terrain_source"]),
         live(meta["terrain_source"] != "synthetic_terrain")),
        ("Mobile coverage", "Ofcom operator coverage predictions" if any(c.startswith("ofcom") for c in cov) else (", ".join(cov) or "Stand-in coverage prior"),
         live(any(c.startswith("ofcom") for c in cov))),
        ("Cell sites and handovers", {"opencellid": "OpenCellID"}.get(meta.get("cell_source", ""), meta.get("cell_source", "") or "Stand-in cell sites"), live(meta.get("cell_source") == "opencellid")),
        ("Satellite", "Predictive sky-visibility model" + (" with terminal telemetry" if any("telemetry" in str(x) for x in obs["source_flags"].unique()) else ""), "predictive"),
        ("Timetable", "Route configuration" if settings.route.get("timetable", {}).get("source", "yaml") == "yaml" else str(settings.route["timetable"]["source"]), "configured"),
    ]
    charts = {"capacity": _chart_capacity(samples, rc, stations), "sections": _chart_sections(sec), "links": _chart_links(links),
              "heatmap": _chart_heatmap(samples, stations, bands), "networks": _chart_networks(samples, stations, obs, names, bands["signal"])}
    design = sim.get("train", {}).get("design")
    preset = sim.get("active_preset")
    presets = sim.get("presets", {})
    if design:
        title = f"Train design: {design.get('title')}"
    elif preset and preset != "baseline" and preset in presets:
        title = presets[preset].get("label", preset)
    else:
        title = "Baseline configuration"
    vname, policy = sim["vehicle"]["profile"], sim["wan"]["policy"]
    detail = f"{VEHICLES.get(vname, vname)} · {POLICIES.get(policy, policy).split(' (')[0].lower()}" + (f" · {weather} weather" if weather != "nominal" else "")
    sat = [f"{p.get('name', p['id'])} ({p['terminal'].replace('_', ' ')} terminal)" for p in settings.satcom_providers]   # "Starlink (performance terminal)"
    pw = sim["passenger_wifi"]
    meta = dict(meta)
    meta["scenario_id"] = "design" if design else (preset or "baseline")
    built = str(meta.get("provenance", {}).get("route", {}).get("fetched_at") or "")[:10]
    meta["built"] = datetime.strptime(built, "%Y-%m-%d").strftime("%d %B %Y").lstrip("0") if built else ""
    base = None
    if baseline is not None:
        base = {"title": baseline["title"], "k": kpis(baseline["rc"], samples, spacing),
                "shares": {m: band_shares(b) for m, b in heat_bands(samples, baseline["rc"], baseline["obs"]).items()}}
    ev = Evidence(
        meta=meta, route_name=meta["route"]["name"], origin=meta["stations"][0]["name"], destination=meta["stations"][-1]["name"],
        k=k, sec=sec, links=links, outages=outage_stretches(samples, rc, stations), shares=shares, charts=charts,
        assumptions=_assumptions(settings), sources=sources, validation=validation, scenario_title=title, scenario_detail=detail,
        vehicle=VEHICLES.get(vname, vname), policy=POLICIES.get(policy, policy),
        satcom=(" and ".join(sat) if sat and sim.get("satcom_enabled", True) else "no satellite link"),
        passengers=f"{pw['passengers']} seats; {pw['active_share'] * 100:.0f} % of passengers online, {pw['per_user_demand_mbps']} Mbps demand each",
        operators=[op.get("name", op["id"]) for op in settings.operators], design=design,
        claims=list(presets.get(preset, {}).get("claims", [])) if preset and not design else [], info=dict(settings.report), baseline=base)
    stem = f"evidence_{meta['route']['id']}_{(preset or 'baseline')}"
    docx_path, xlsx_path = out_dir / f"{stem}.docx", out_dir / f"{stem}.xlsx"
    reference = write_docx(docx_path, ev)
    write_xlsx(xlsx_path, ev, samples, rc, obs, reference)
    for name, png in charts.items():
        (out_dir / f"{stem}_{name}.png").write_bytes(png)
    return {"docx": docx_path, "xlsx": xlsx_path}
