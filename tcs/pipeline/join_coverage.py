"""Coverage prior per (sample, operator): Ofcom API -> Connected Nations -> synthetic, with provenance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..sources import synthetic
from ..sources.base import SourceUnavailable, console
from ..sources.ofcom_coverage import corridor_postcodes, fetch_connected_nations, fetch_ofcom_api
from .sample_route import RouteBundle, urban_from_point_density


def coverage_prior(settings: Settings, bundle: RouteBundle, *, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (prior table, samples with urban_density possibly refined)."""
    samples = bundle.samples
    raw = settings.paths()["raw"]
    postcodes = None
    if not settings.offline:
        try:
            postcodes = corridor_postcodes(samples, bundle.proj, raw, offline=settings.offline)
            console.log(f"corridor postcodes: {postcodes['postcode'].nunique()} unique within 400 m of the line")
        except SourceUnavailable as exc:
            console.log(f"[yellow]postcodes unavailable ({exc})")
    if postcodes is not None:
        # Postcode density is a good urban proxy (dense postcodes = dense premises).
        try:
            from ..sources.ofcom_coverage import CODEPOINT_URL  # noqa: F401  (documented dependency)
            samples = samples.copy()
            pcs_xy = postcodes.drop_duplicates("postcode")
            near = samples.set_index("sample_id").loc[pcs_xy["sample_id"]]
            samples["urban_density"] = np.maximum(samples["urban_density"].values,
                                                  urban_from_point_density(samples, near["x"].values, near["y"].values))
        except Exception as exc:  # pragma: no cover
            console.log(f"[yellow]urban density refinement skipped: {exc}")

    prior = None
    if postcodes is not None:
        for fn, label in ((lambda: fetch_ofcom_api(postcodes, settings, raw, limit=limit), "ofcom_api"),
                          (lambda: fetch_connected_nations(postcodes, settings, raw), "connected_nations")):
            try:
                by_pc = fn()
                prior = postcodes.merge(by_pc, on="postcode", how="left")
                prior = prior[prior["provider_id"].notna()]
                console.log(f"coverage prior from {label}: {prior['prior_score'].notna().mean():.0%} of sample-operator pairs covered")
                break
            except SourceUnavailable as exc:
                console.log(f"[yellow]{label}: {exc}")
    if prior is None or prior.empty:
        console.log("[yellow]coverage prior: synthetic stand-in (flagged, confidence capped)")
        prior = synthetic.synthetic_prior(samples, settings.operators)
    # Fill operators/samples missing from the live source with NaN rows so the model sees every pair.
    full = pd.MultiIndex.from_product([samples["sample_id"], [op["id"] for op in settings.operators]], names=["sample_id", "provider_id"]).to_frame(index=False)
    prior = full.merge(prior, on=["sample_id", "provider_id"], how="left")
    prior["source"] = prior["source"].fillna("no_coverage_record")
    return prior, samples
