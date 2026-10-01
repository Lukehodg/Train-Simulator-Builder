# Model

All constants: `config/simulation.yaml` (versioned by `model_version`).

## Cellular (per operator, per 50 m sample)

```
q_base = prior_score                       # Ofcom level → score, or Connected Nations %, or synthetic
       − cutting_penalty_max · min(1, cutting_depth / 10 m)
       − penalty_per_km · max(0, serving_cell_distance_km − 3)
       + calibration_bias[operator]        # tcs calibrate (one route) or tcs calibrate-national (config/calibration.yaml)
q      = clip(q_base + vehicle_offset)     # rooftop 0 · handset −0.18 (≈ −18 dB penetration)
tunnel: q = floor + (q_portal − floor) · exp(−d / portal_decay_m), the better of the two portals, where q_portal is
        the open-air q just outside and d the distance in; floor = default_score (no in-tunnel infrastructure).
        0.52 where DAS is assumed (km ranges / names in the route's `das_tunnels`; none by default)

capacity = capacity_prior[tech] · ((q − 0.12)/0.88)^1 · (1 − 0.08·speed/200) · (1 − 0.4·handover_penalty)
latency  = 24 + 110·(1 − q) + 80·handover_penalty        (ms)
loss     = table(q) + 2·handover_penalty                  (%)
rsrp     = −120 + 46·q   (flagged synthetic until calibrated; then slope·q + intercept per operator, tunnels included)
```

Handover: serving cell = nearest candidate with hysteresis (switch when < 78 % of current distance or current
> 9 km); a switch opens a 6-sample (300 m) penalty window that decays linearly.

## Vehicle profiles and the EDGE Rail preset

| profile | score offset | dB | throughput factor | meaning |
|---|---|---|---|---|
| EXTERNAL_ROOFTOP_ANTENNA | 0 | 0 | 1.00 | passive roof antenna, coax to a rack router |
| EDGE_RAIL_ACTIVE_ANTENNA | +0.08 | +6 | 1.35 | modem inside the radome (Sierra EM9291, 4x4 MIMO, LTE Cat 20): no coax/splitter losses, MIMO diversity |
| PASSENGER_HANDSET_INSIDE_CARRIAGE | -0.18 | -18 | 1.00 | handset behind coated glass |

The `edge_rail_fleet_connect` preset = EDGE_RAIL_ACTIVE_ANTENNA + PACKET_BONDING (Fleet Connect aggregating every
cellular network and the satcom link simultaneously) with bonding efficiency 0.85. Motion Applied's marketing
figures (5x downloads, 10x uploads, 80 % fewer blackspots) are displayed as claims and never enter the model.

## Train Studio designs

`train_studio.derive()` maps a `*.train.json` consist onto: vehicle profile (EDGE Rail units -> active antenna,
none -> handset), fitted networks = one per EDGE Rail unit in `edge_rail_networks` order (EE, Vodafone, Three, O2;
units beyond that are not modelled), satcom enabled + terminal class
(EDGE Mini -> Starlink Mini: 100 Mbps prior, 35° minimum elevation), AP capacity = 120 Mbps per *connected* AP
(carriages without a switch are excluded), seats per carriage type (cab 56 / intermediate 76), and policy
(Fleet Connect -> bonding, otherwise failover).

## Satcom (Starlink)

```
dome     = solid-angle fraction of the dome above 20° left open by the horizon
  LiDAR (GB):  horizon = max(LiDAR skyline within 60 m: cutting walls, trees, buildings; terrain beyond)
               sky = dome · (1 − max(0.85·canopy, share of the track under a bridge))
  otherwise:   horizon = terrain model
               sky = (dome − min(0.55, cutting/22)) · (1 − 0.85·canopy)
sky      = sky − 0.18·urban·(1 − canopy) − weather.sky_penalty; 0 in tunnels
available = in service ∧ ¬tunnel ∧ sky ≥ 0.35
capacity  = 220 · sky^1.4 · weather.capacity_factor · (0.3 if temporary beam handover)
latency   = 28 + 35·(1 − sky) (+40 during handover)
reason    ∈ {OPEN_SKY, TUNNEL, DEEP_CUTTING, STATION_CANOPY, URBAN_OBSTRUCTION, TEMPORARY_HANDOVER, WEATHER_PENALTY, SERVICE_UNAVAILABLE}
```

## Link manager

Only links the train has a modem for take part: every satellite link, and the mobile networks in
`cellular.fitted_networks` (null = all of them, a multi-SIM router; the EDGE Rail preset fits EE, Vodafone and Three,
one antenna each). The other networks are still simulated and shown, marked "not fitted".

```
score = 0.35·cap/250 + 0.25·(1 − lat/200) + 0.20·(1 − loss/10) + 0.10·stability + 0.10·confidence
usable = available ∧ score ≥ 0.25
PACKET_BONDING:      bonded = 0.82 · Σ usable capacity · 0.9 ; latency = min + 5 ; loss = 0.7·min
WEIGHTED_LB:         bonded = 0.78 · Σ ; latency = capacity-weighted mean
FAILOVER / *_PRIMARY_*_BACKUP: single best (or primary/backup) link
```

## Passenger Wi-Fi

```
active_users = passengers · load_factor(km) · 0.35
per_user     = min(3.75, 0.85 · min(bonded, AP 300) / active_users)
score        = 100·(0.55·min(1, per_user/3) + 0.25·(1 − lat/250) + 0.20·(1 − loss/8))
class        = EXCELLENT ≥ 80 · GOOD ≥ 60 · USABLE ≥ 40 · POOR ≥ 15 · OUTAGE
```

## Confidence

synthetic prior 0.25 · Ofcom prior 0.55 · Ofcom + cells 0.65 · +0.15 when calibrated · measured within 250 m 0.92 ·
tunnels 0.90 (deterministic) · Starlink predictive 0.35 / telemetry 0.85. The WAN confidence is the mean over
active links. Switch the ribbons to **Confidence** in the viewer to see it.
