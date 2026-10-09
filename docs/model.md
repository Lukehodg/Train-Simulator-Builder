# Model

All constants: `config/simulation.yaml` (versioned by `model_version`).

## Cellular (per operator, per 50 m sample)

```
q_base = prior_score                       # Ofcom level → score, or Connected Nations %, or synthetic
       − cutting_penalty_max · min(1, cutting_depth / cutting_full_depth_m)   # 0.28 at 10 m (fitted; depth from LiDAR in GB)
       − penalty_per_km · max(0, serving_cell_distance_km − free_km)       # 0 (fitted: no penalty since model 0.4.2)
       + calibration_bias[operator]        # tcs calibrate (one route) or tcs calibrate-national (config/calibration.yaml)
       + measured_correction_db / slope    # where trains measured: earlier trips' errors here (tcs correct-routes)
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
> 9 km); a switch opens a 6-sample (300 m) penalty window that decays linearly. Candidates are OpenCellID's cells,
except that a 4G mast placed from the Global View logs (`config/masts.csv`) stands in for OpenCellID's cells of the
same mast at its fitted position (docs/data-sources.md, 4a).

The distance penalty was hand-set at 0.06 per km beyond 3 km until the national fit searched it (October 2026). With
OpenCellID's positions or with the fitted ones, the measurements choose almost none: 0.01 per km beyond 8 km. The
hand-set value made far-from-mast stretches too weak; removing it cut the 2026 check's error from 10.2 to 9.7 dB.
Model 0.4.2 fixed the distance beyond the candidate radius: 12 km from every cell of a network it had been left blank,
so no penalty applied exactly where the train was furthest from a mast. Refitted on the corrected distances, the
measurements choose no penalty at all (docs/validation.md). `serving_distance_m` is still the distance to the cell the
model hands over between, as the viewer shows it.

**Measured corrections** (`tcs/corrections.py`). Where scanner trains have measured a route, the calibrated model's
error at each measured route point (after one level per measurement set and network, taken over every route the set
covers, so a route the model over- or under-rates keeps that offset in its corrections) is smoothed along the track
with a 100 m Gaussian and shrunk where few measurements are near (correction = Σwg·e / (Σwg + 0.5)), with weight 1 for
the 2026 Global View logs and 0.1 for the 2018–19 Yellow Train logs. The correction (dB) enters the score through the
calibration's dB scale before the clip, so capacity, latency and the viewer follow. Inside a tunnel trains have
measured, in-tunnel errors alone (against the portal or DAS model) are smoothed and added to that model there.
Signal at a given spot repeats from trip to trip much more closely than coverage predictions place it, so this is the
largest single accuracy gain the model has (docs/validation.md). Confidence there is 0.85.

## Vehicle profiles and the EDGE Rail preset

| profile | score offset | dB | throughput factor | meaning |
|---|---|---|---|---|
| EXTERNAL_ROOFTOP_ANTENNA | 0 | 0 | 1.00 | passive roof antenna, coax to a rack router |
| EDGE_RAIL_ACTIVE_ANTENNA | +0.08 | +6 | 1.35 | HUBER+SUHNER SENCITY Rail Active Rooftop: modem inside the radome (Sierra EM9291, 4x4 MIMO, LTE Cat 20), no coax/splitter losses, MIMO diversity |
| PASSENGER_HANDSET_INSIDE_CARRIAGE | -0.18 | -18 | 1.00 | handset behind coated glass |

The `edge_rail_fleet_connect` preset = EDGE_RAIL_ACTIVE_ANTENNA + PACKET_BONDING (Fleet Connect, Motion Applied's
link-management software, aggregating every cellular network and the satcom link simultaneously) with bonding
efficiency 0.85. The marketing figures (5x downloads, 10x uploads, 80 % fewer blackspots) are displayed as claims and
never enter the model. The +6 dB and 1.35x are assumptions: the HUBER+SUHNER data sheet gives no antenna gain or
noise figure.

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
