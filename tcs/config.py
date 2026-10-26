"""Configuration loading: YAML files in config/ plus secrets from .env / environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
ROUTES_DIR = CONFIG_DIR / "routes"
COUNTRIES_DIR = CONFIG_DIR / "countries"
BASE_COUNTRY = "GB"           # config/*.yaml describe Great Britain; other countries override them (config/countries/<CC>/)
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
PROCESSED = DATA_DIR / "processed"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:
    route: dict[str, Any]
    terrain: dict[str, Any]
    networks: dict[str, Any]
    starlink: dict[str, Any]
    sim: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    offline: bool = False
    sim_defaults: dict[str, Any] = field(default_factory=dict)     # pristine copy of simulation.yaml (before presets / train designs)
    starlink_defaults: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)          # config/report.yaml: cover-page and header details
    country_profile: dict[str, Any] = field(default_factory=dict)  # config/countries/<CC>/profile.yaml; empty for GB

    # ---- convenience accessors -------------------------------------------------
    @property
    def country(self) -> str:
        return normalise_country(self.route.get("country"))

    @property
    def is_gb(self) -> bool:
        return self.country == BASE_COUNTRY

    @property
    def route_id(self) -> str:
        return self.route["id"]

    @property
    def spacing_m(self) -> float:
        return float(self.route.get("sample_spacing_m", 50))

    @property
    def operators(self) -> list[dict[str, Any]]:
        return self.networks["cellular"]["operators"]

    @property
    def satcom_providers(self) -> list[dict[str, Any]]:
        return [p for p in self.starlink["satcom"]["providers"] if p.get("enabled", True)]

    def key(self, name: str) -> str | None:
        v = self.env.get(name) or os.environ.get(name)
        return v.strip() if v and v.strip() else None

    def paths(self) -> dict[str, Path]:
        rid = self.route_id
        return {
            "raw": RAW / rid,
            "interim": INTERIM / rid,
            "processed": PROCESSED / rid,
            "web": PROCESSED / rid / "web",
        }


def normalise_country(code: str | None) -> str:
    c = (code or BASE_COUNTRY).strip().upper()
    return "GB" if c == "UK" else c


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` laid on top of `base`, dict by dict; any other value (lists included) replaces the base's."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def country_profile(country: str | None, config_dir: Path | None = None) -> dict[str, Any]:
    """config/countries/<CC>/profile.yaml for a route outside Great Britain, with `dir` set; {} for GB or a country
    without a profile (which then runs on the GB defaults, as before)."""
    cc = normalise_country(country)
    if cc == BASE_COUNTRY:
        return {}
    d = (config_dir or CONFIG_DIR) / "countries" / cc
    if not (d / "profile.yaml").exists():
        return {}
    return dict(_load_yaml(d / "profile.yaml"), dir=d, code=cc)


def list_routes(config_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every route in config/routes/*.yaml plus the default in config/route.yaml (id, name, origin, destination)."""
    cfg = config_dir or CONFIG_DIR
    out = []
    seen = set()
    for f in [cfg / "route.yaml"] + sorted((cfg / "routes").glob("*.yaml")):
        r = _load_yaml(f).get("route", {})
        if r.get("id") and r["id"] not in seen:
            seen.add(r["id"])
            out.append({"id": r["id"], "name": r.get("name"), "operator": r.get("operator"), "country": normalise_country(r.get("country")), "origin": r["stations"][0]["name"], "destination": r["stations"][-1]["name"], "file": str(f)})
    return out


def _route_config(cfg: Path, route_id: str | None) -> dict[str, Any]:
    base = _load_yaml(cfg / "route.yaml")["route"]
    if not route_id or route_id == base.get("id"):
        return base
    f = cfg / "routes" / f"{route_id}.yaml"
    if not f.exists():
        raise FileNotFoundError(f"no route '{route_id}' in {cfg / 'routes'} (tcs routes lists them)")
    r = _load_yaml(f)["route"]
    # Route files only need the geography + timetable; operational defaults come from route.yaml.
    for k in ("sample_spacing_m", "corridor_m", "geometry", "accel_mps2", "decel_mps2"):
        r.setdefault(k, base.get(k))
    r.setdefault("line_speed_kph", {"default": base.get("line_speed_kph", {}).get("default", 200), "restrictions": []})
    r.setdefault("timetable", {}).setdefault("dwell_s", base.get("timetable", {}).get("dwell_s", 120))
    r.setdefault("country", base.get("country", "GB"))
    return r


def load_settings(config_dir: Path | None = None, offline: bool = False, route_id: str | None = None) -> Settings:
    cfg = config_dir or CONFIG_DIR
    load_dotenv(ROOT / ".env")
    route_doc = _load_yaml(cfg / "route.yaml")
    route = _route_config(cfg, route_id)
    prof = country_profile(route.get("country"), cfg)
    terrain = route_doc.get("terrain", {})
    networks = _load_yaml(cfg / "networks.yaml")
    sim = _load_yaml(cfg / "simulation.yaml")
    if prof:
        # A country profile swaps in its own networks file and lays its overrides over the GB defaults.
        if prof.get("networks"):
            networks = _load_yaml(prof["dir"] / prof["networks"])
        terrain = _deep_merge(terrain, prof.get("terrain") or {})
        sim = _deep_merge(sim, prof.get("simulation") or {})
    s = Settings(
        route=route,
        terrain=terrain,
        networks=networks,
        starlink=_load_yaml(cfg / "starlink.yaml"),
        sim=sim,
        env={k: v for k, v in os.environ.items() if k in {"OFCOM_API_KEY", "OPENCELLID_TOKEN", "OS_DATAHUB_KEY", "NROD_USERNAME", "NROD_PASSWORD", "OVERPASS_URL", "FCC_BDC_USERNAME", "FCC_BDC_TOKEN"}},
        offline=offline,
        report=_load_yaml(cfg / "report.yaml").get("report", {}) if (cfg / "report.yaml").exists() else {},
        country_profile=prof,
    )
    import copy

    # In-tunnel coverage is geography: each route lists its own tunnels, and a list in simulation.yaml never leaks
    # one route's km ranges onto another.
    s.sim["cellular"]["tunnels"]["das_tunnels"] = list(s.route.get("das_tunnels") or [])
    s.sim_defaults = copy.deepcopy(s.sim)
    s.starlink_defaults = copy.deepcopy(s.starlink)
    for p in s.paths().values():
        p.mkdir(parents=True, exist_ok=True)
    return s
