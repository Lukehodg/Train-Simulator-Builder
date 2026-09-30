"""Train Studio (Train Diagram Builder) project files -> onboard architecture parameters.

The builder saves `<title>.train.json`:
  {"app": "motion-applied-train-studio", "version": 1, "project": {
      "title": str, "cars": [{"id", "type": "cabL"|"mid"|"cabR", "aps": int, "sw": bool, "edge": int, "fleet": bool,
                              "custom": {typeId: bool}, "mounts": {...}}],
      "types": [{"id", "slug", "name", "placement": "roof"|"ceiling"|"rack", "link": bool,
                 "hardwareKind": "satcom"|"router"|"product"|None, "presetId": "starlink"|"oneweb"|"satcom"|"edge-path"|"edge-mini"|"edge-core"|None}],
      ...}}

We read the consist and derive what the connectivity model needs (all mappings live in simulation.yaml `train:`):
  cellular units   EDGE Rail roof units (+ custom roof "product" items flagged as cellular) -> capacity aggregation factor
  satcom           EDGE Mini (integrated Starlink Mini), Starlink/OneWeb/SATCOM types -> satcom enabled + terminal class
  access points    per carriage, only counted when the carriage has a switch (the builder flags "needs switch")
  passengers       seats per carriage type
  vehicle profile  EDGE Rail active antenna when any roof unit exists, otherwise passenger handset
  link policy      Fleet Connect present -> bonding; absent -> failover
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

APP_ID = "motion-applied-train-studio"
MAX_PROJECT_BYTES = 5 * 1024 * 1024


def validate_project(project: Any) -> dict:
    """Validate imported designs before arithmetic or iteration; ignore unrelated editor fields."""
    if not isinstance(project, dict):
        raise ValueError("project must be an object")
    if not isinstance(project.get("title", ""), str) or len(project.get("title", "")) > 512:
        raise ValueError("project title must be text of at most 512 characters")
    cars, types = project.get("cars"), project.get("types", [])
    if not isinstance(cars, list) or not 1 <= len(cars) <= 100:
        raise ValueError("project must contain between 1 and 100 carriages")
    if not isinstance(types, list) or len(types) > 256:
        raise ValueError("project types must be an array of at most 256 entries")
    ids = set()
    for item in types:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"] or item["id"] in ids:
            raise ValueError("equipment types require unique non-empty text ids")
        ids.add(item["id"])
        for key in ("name", "presetId", "hardwareKind"):
            if item.get(key) is not None and not isinstance(item[key], str):
                raise ValueError(f"equipment {key} must be text")
        if not isinstance(item.get("name"), str):
            raise ValueError("equipment name must be text")
        if "link" in item and not isinstance(item["link"], bool):
            raise ValueError("equipment link must be boolean")
    for car in cars:
        if not isinstance(car, dict) or car.get("type", "mid") not in ("cabL", "mid", "cabR"):
            raise ValueError("carriage type must be cabL, mid or cabR")
        for key in ("aps", "edge"):
            value = car.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1000 or int(value) != value:
                raise ValueError(f"carriage {key} must be an integer from 0 to 1000")
        for key in ("sw", "fleet"):
            if key in car and not isinstance(car[key], bool):
                raise ValueError(f"carriage {key} must be boolean")
        custom = car.get("custom", {})
        if not isinstance(custom, dict) or any(not isinstance(v, bool) for v in custom.values()):
            raise ValueError("carriage custom equipment must be an object of booleans")
    return project


@dataclass
class Carriage:
    index: int
    type: str
    aps: int
    switch: bool
    cellular_units: int
    fleet_connect: bool
    satcom_units: int
    custom: list[str]
    aps_connected: int


@dataclass
class TrainDesign:
    title: str
    carriages: list[Carriage]
    cellular_units: int
    satcom_units: int
    satcom_terminal: str | None
    aps_total: int
    aps_connected: int
    fleet_connect: bool
    passengers: int
    vehicle_profile: str
    policy: str
    fitted_networks: list[str] | None                    # one network per EDGE Rail unit; None = no units (passenger handsets)
    ap_capacity_mbps: float
    warnings: list[str] = field(default_factory=list)
    source_file: str | None = None

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_carriages"] = len(self.carriages)
        return d


def load_project(path: Path) -> dict:
    if path.stat().st_size > MAX_PROJECT_BYTES:
        raise ValueError("Train Studio project exceeds the 5 MiB limit")
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get("app") != APP_ID or type(doc.get("version")) is not int or doc.get("version") != 1 or "project" not in doc:
        raise ValueError(f"{path.name} is not a Train Studio v1 project")
    return validate_project(doc["project"])


def derive(project: dict, tcfg: dict, satcom_cfg: dict | None = None) -> TrainDesign:
    """Map a project onto model parameters using the `train:` section of simulation.yaml."""
    validate_project(project)
    seats = tcfg.get("seats", {"cabL": 56, "mid": 76, "cabR": 56})
    types = {t["id"]: t for t in project.get("types", [])}
    sat_kind = tcfg.get("satcom_terminal_by_preset", {"edge-mini": "mini", "starlink": "performance", "oneweb": "performance", "satcom": "performance"})
    cars: list[Carriage] = []
    warnings: list[str] = []
    sat_units, sat_terminal = 0, None
    for i, c in enumerate(project.get("cars", [])):
        custom_on = [tid for tid, on in (c.get("custom") or {}).items() if on and tid in types]
        sat_here = 0
        for tid in custom_on:
            t = types[tid]
            is_sat = t.get("hardwareKind") == "satcom" or t.get("presetId") in ("starlink", "oneweb", "satcom", "edge-mini") \
                or str(t.get("name", "")).strip().lower() in ("starlink", "oneweb", "satcom")
            if is_sat:
                sat_here += 1
                term = sat_kind.get(t.get("presetId") or str(t.get("name", "")).lower(), tcfg.get("default_satcom_terminal", "performance"))
                sat_terminal = sat_terminal or term
        aps = int(c.get("aps", 0))
        sw = bool(c.get("sw", False))
        links_need_switch = aps > 0 or int(c.get("edge", 0)) > 0 or bool(c.get("fleet")) or any(types[t].get("link") for t in custom_on)
        if links_need_switch and not sw:
            warnings.append(f"carriage {i + 1}: equipment present but no switch — its {aps} access point(s) are not counted")
        cars.append(Carriage(index=i, type=c.get("type", "mid"), aps=aps, switch=sw, cellular_units=int(c.get("edge", 0)), fleet_connect=bool(c.get("fleet")),
                             satcom_units=sat_here, custom=[types[t]["name"] for t in custom_on], aps_connected=aps if sw else 0))
        sat_units += sat_here
    cell_units = sum(c.cellular_units for c in cars)
    # Each EDGE Rail unit carries one network's SIM: the 1st unit the first network in the list, and so on.
    order = [str(x) for x in tcfg.get("edge_rail_networks", ["ee", "vodafone", "three", "o2"])]
    if cell_units > len(order):
        warnings.append(f"{cell_units} EDGE Rail units but {len(order)} networks to put them on: the extra {cell_units - len(order)} are not modelled")
    fleet = any(c.fleet_connect for c in cars)
    aps_connected = sum(c.aps_connected for c in cars)
    passengers = sum(int(seats.get(c.type, seats.get("mid", 76))) for c in cars)
    design = TrainDesign(
        title=project.get("title", "Train Studio project"), carriages=cars, cellular_units=cell_units, satcom_units=sat_units,
        satcom_terminal=sat_terminal if sat_units else None, aps_total=sum(c.aps for c in cars), aps_connected=aps_connected, fleet_connect=fleet,
        passengers=passengers, vehicle_profile=tcfg.get("edge_rail_profile", "EDGE_RAIL_ACTIVE_ANTENNA") if cell_units > 0 else "PASSENGER_HANDSET_INSIDE_CARRIAGE",
        policy=tcfg.get("fleet_connect_policy", "PACKET_BONDING") if fleet else tcfg.get("no_fleet_connect_policy", "FAILOVER"),
        fitted_networks=order[:cell_units] if cell_units else None, ap_capacity_mbps=float(tcfg.get("ap_capacity_mbps_each", 120)) * aps_connected, warnings=warnings,
    )
    if cell_units == 0:
        design.warnings.append("no EDGE Rail units: cellular modelled as passenger handsets inside the carriage")
    if sat_units == 0:
        design.warnings.append("no SATCOM terminal: satellite link disabled")
    if aps_connected == 0:
        design.warnings.append("no connected access points: passenger Wi-Fi capacity is zero")
    return design


def apply_to_settings(settings, design: TrainDesign) -> None:
    """Override the simulation config in place so every model stage sees the designed train."""
    sim = settings.sim
    sim["vehicle"]["profile"] = design.vehicle_profile
    sim["wan"]["policy"] = design.policy
    sim["cellular"]["fitted_networks"] = design.fitted_networks
    sim["passenger_wifi"]["passengers"] = design.passengers
    sim["passenger_wifi"]["ap_capacity_mbps"] = max(0.0, design.ap_capacity_mbps)
    sim["satcom_enabled"] = design.satcom_units > 0
    for p in settings.starlink["satcom"]["providers"]:
        p["enabled"] = design.satcom_units > 0
        if design.satcom_terminal and design.satcom_terminal in p.get("capacity_prior_mbps", {}):
            p["terminal"] = design.satcom_terminal
    sim["train"] = {**sim.get("train", {}), "design": design.summary()}


def load_design(path: Path, settings) -> TrainDesign:
    design = derive(load_project(path), settings.sim.get("train", {}), settings.starlink)
    design.source_file = str(path)
    return design
