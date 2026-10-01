# Active antenna vs passive install: how to work out the difference, route by route

## 0. The three profiles compared

A passive install is one of two kinds. The comparison only means something against the kind a fleet actually has.

| | **A: EDGE Rail active antenna** | **P2: rack router, 2×2 per modem** (legacy) | **P4: rack router, 4×4 per modem** (current) |
|---|---|---|---|
| Examples | HUBER+SUHNER SENCITY Rail Active Rooftop + Fleet Connect | Icomera X6 / X6i, X5 v1 (LTE Cat-12); earlier Nomad installs | Icomera X5 v2 (5G, LTE Cat-20 fallback) and X7; Nomad 5G (Cat-20, 4×4 roof antennas) |
| Where the modem is | in the radome, one network per unit | in the rack, one network per modem | in the rack, one network per modem |
| Between antenna and modem | nothing | one coax run per port: connectors, surge protection, maybe splitters | as P2, but **4 runs per modem** |
| Receive branches `M` | 4 | 2 | 4, if the roof antenna has 4 ports per modem and all 4 are cabled (otherwise it is P2) |
| Layers `L_modem` | 4 | 2 | 4 |
| Technology and bands | 5G sub-6 + LTE, every UK band incl. n78 | LTE only, as the modules support (X5 v1: every UK LTE band incl. 1400 MHz; older X6 modules may lack some); no 5G | 5G sub-6 incl. n78 + LTE |
| Feeder loss `L(c)` | 0 | `ℓ·(k1·√f + k2·f)` + fittings | same as P2, per run |
| Noise figure | the modem's | the modem's + `L(c)` | the modem's + `L(c)` |
| Networks at once | one per unit (typically 3 units: EE, Vodafone, Three) | one per modem: X6 4 (+1), X5 4, Nomad up to 6 | X5 v2 4, X7 up to 5, Nomad entry 2 / high end up to 6 |
| Combining networks | Fleet Connect (Motion Applied) | SureWAN (Icomera) / Nomad Connect | SureWAN / Nomad Connect, with LEO satellite on X7 / XS1 / Nomad 5G |

Placeholder defaults for P2 and P4, to be replaced by the real install:
- **Feeder loss:** 10 m of LMR-400-class cable gives 1.2 / 1.9 / 2.3 / 2.7 dB at 800 / 1800 / 2600 / 3500 MHz, plus 1 dB of connectors and surge protection. Thinner rail cable or longer runs mean more.
- **Splitters:** none (a 2-way split adds 3 dB).
- **Antenna gain:** the same elements as A, until the roof antenna's own pattern files are read with `antenna_patterns.py`.

What each comparison measures:
- **A vs P2:** feeder loss (small where interference dominates), plus **4 branches instead of 2** (about +5 dB at the worst 1 in 10, plus interference rejection), **4 layers instead of 2**, and **5G incl. n78**. This is where a large uplift is possible.
- **A vs P4:** mostly **feeder loss and noise figure**, which are worth under 1 dB of SINR where interference dominates (section 1.1) and more at coverage edges and in cuttings. Add whatever differs in how many networks run at once. Expect a modest uplift here, concentrated on noise-limited stretches.

Everything below is per route sample *s* (50 m), network *n*, carrier *c* and configuration *k* ∈ {A, P2, P4}; in
formulas, *P* stands for whichever passive profile is compared. The values a formula needs are listed in section 3.
The vendors publish modem counts and MIMO but not cable runs, connector losses or roof-antenna gain, so
**install-specific values for P2 and P4** come from the actual installation: number of modems, ports per modem,
cable type and length, splitters, and the roof antenna's data sheet.

## 1. Per sample: from signal to throughput

### 1.1 Signal, noise and interference at the modem

```
S_k = S0 + G_k(c) − L_k(c)                         wanted signal (dBm per resource element)
I_k = I0 + G_k(c) − L_k(c)                         interference arrives through the same antenna and cable
N_k = −174 + 10·log10(15 kHz) + NF_k + L_k(c)       noise: a loss ahead of the first amplifier adds to the noise figure
```

`S0` is the model's calibrated RSRP at the roof (Yellow Train level). `G` is antenna gain towards the horizon (per
band, from the vendor's pattern files via `antenna_patterns.py`). `L` is feeder loss: 0 for the active antenna,
whose modem sits at the antenna. `NF` is the receiver noise figure.

The interference-to-noise ratio, as the active antenna sees it, is `INR = I_A / N_A`. The SINR difference on one
branch is then:

```
ΔSINR = SINR_A − SINR_P = 10·log10( (INR + L_P·F_P/F_A) / (INR + 1) )        (L, F as linear ratios)
```

| INR | ΔSINR, 2 dB feeder loss | ΔSINR, 4 dB feeder loss |
|---|---|---|
| −10 dB (noise-limited) | +1.9 dB | +3.8 dB |
| 0 dB | +1.1 dB | +2.5 dB |
| 10 dB | +0.2 dB | +0.6 dB |
| 20 dB and up (interference-limited) | ≈ 0 | ≈ 0 |

Removing the cable only helps where the link is noise-limited. Antenna gain scales the wanted signal and the
interference alike, so a gain difference between the two antennas behaves the same way.

**What the measurements say.** In the Network Rail Global View 4G logs, on the strongest cell of each carrier:
- interference sits a median **~34 dB above the noise floor**;
- only **0.3 %** of carrier-seconds are noise-limited (INR < 0 dB), and 0–15 % depending on band have INR < 10 dB;
- this assumes a 7 dB scanner noise figure: 4–10 dB gives 0.1–0.8 % noise-limited.

Two caveats:
- **Survivorship:** the scanner logs nothing where it hears no cell, so true coverage edges are under-represented.
- **Scanner SINR vs modem SINR:** a scanner's reference-signal SINR can sit below what a modem reports.

So on UK railways the feeder loss is worth well under 1 dB of SINR most of the time. The active antenna's advantage
has to come from 1.2–1.4.

### 1.2 More receive branches: fading and interference

With *M* receive branches combined (MRC) on a Rayleigh-faded channel, the share of time below *x* is:

```
P(γ < x) = 1 − e^(−u) · Σ_{k=0}^{M−1} u^k / k!,     u = x / (mean SINR per branch)
```

Low percentiles of SINR, relative to one branch's mean:

| | M = 1 | M = 2 | M = 4 |
|---|---|---|---|
| 10th percentile | −9.8 dB | −2.7 dB | +2.4 dB |
| 1st percentile | −20.0 dB | −8.3 dB | −0.8 dB |

So 4 branches against 2 lift the worst-1-in-10 SINR by about **5 dB**, and the worst-1-in-100 by about **7.5 dB**.
Correlated branches give less. Use an effective branch count:

```
M_eff = M / (1 + (M − 1)·ρ)
```

The correlation ρ comes from port isolation and pattern diversity. The HUBER+SUHNER ports face different sides of
the train, which keeps ρ low. Their isolation is lower at 700–900 MHz than from 1800 MHz up, so the low bands get
the smaller gain.

In interference-limited conditions, an interference-rejecting receiver (IRC) with *M* branches can also suppress up to
*M − 1* dominant interferers:

```
SINR_IRC ≈ SINR_MRC + g_IRC(M, number of dominant interferers)
```

`g_IRC` is fitted from modem logs. This is where 4 antennas should matter most on a railway.

### 1.3 Layers (spatial multiplexing)

```
layers_eff = min( L_modem(k), L_network(c), rank(γ, ρ) )
P(rank ≥ r | γ) = 1 / (1 + e^(−(γ − γ_r)/w))
```

- `L_modem`: 4 for A and P4; 2 for P2.
- `L_network`: what the cell offers. UK macro LTE is mostly 2 layers; n78 offers up to 4.
- Rank thresholds `γ_r` (roughly 10 dB for 2 layers, 18–20 dB for 4) are fitted from the rank indicator in modem logs.

### 1.4 Throughput

```
T_c = B_c · layers_eff · η(γ − 10·log10(layers_eff)) · (1 − u_c) · (1 − BLER)
η(γ) = 0                    below γ_min
     = α · log2(1 + γ)      in between
     = η_max                above γ_max                (3GPP TR 36.942 attenuated Shannon)
T_n = Σ over the carriers the modem can aggregate  T_c
```

- `B_c`: carrier bandwidth.
- `u_c`: the share of the cell used by others (load). It's a prior until operator or test data sets it.
- `α`, `γ_min`, `η_max`: fitted from the Trainlab active-antenna logs (`aa/README.md`, step 3).
- Speed: Doppler `f_D = v·f_c/c` (650 Hz at 200 km/h on 3.5 GHz). A speed factor is fitted from logs, not assumed.

### 1.5 The whole train

```
T_train,k = η_bond · Σ_n T_n,k      bonding (Fleet Connect for A, the router's bonding for P)
          = max_n T_n,k             failover
```

## 2. Per route: what to report

Samples are weighted by the time the train spends on them: `Δt_s = 50 m / v_s`.

| Measure | Formula | Answers |
|---|---|---|
| Typical / worst 1 in 10 | time-weighted 50th / 10th percentile of `T_train` | what passengers usually get, and can count on |
| Availability at τ | `A(τ) = Σ Δt_s·1[T_s ≥ τ] / Σ Δt_s` (τ = 2, 10, 50 Mbit/s) | share of the journey a service works |
| Blackspots | runs of consecutive samples with `T < τ_min`: count, total minutes, longest | the passenger experience of "no signal" |
| Blackspot reduction | `R = 1 − minutes_A / minutes_P` | tests the "80 % fewer blackspots" claim |
| Uplift | `median_s(T_A / T_P)`, and `p10(T_A) / p10(T_P)` | tests "5× faster downloads" |
| Per passenger | `T_train / (seats × load × share online)` | sizing the onboard Wi-Fi |

Break every measure down by:
- **regime:** noise-limited (INR < 0 dB), mixed (0–15 dB), interference-limited (> 15 dB);
- **setting:** open, cutting, tunnel, urban;
- **n78:** present or not.

This shows where the uplift comes from. The model predicts most of it in cuttings, rural fringes and wherever 4
branches beat 2, and almost none from cable loss on busy urban stretches.

## 3. Where the inputs come from

| Input | Source | Status |
|---|---|---|
| `S0` per sample, network | the simulator's calibrated RSRP (Yellow Train, LiDAR cutting depths) | done |
| `INR` per sample, network, band | Global View: `INR = 10·log10((S/γ − N)/N)` per reading, median per sample, smoothed over ~1 km; where no reading, from the RSRP→SINR fit (`scripts/fit_rsrp_sinr.py`) | data in hand; per-route matching next |
| `G_A(c)`, isolation | vendor pattern files → `antenna_patterns.py` (kept out of git) | done for 1399.99.0153; confirm it is the active unit's antenna |
| `G_P(c)` | the passive antenna's data sheet / pattern files | needed |
| `L_P(c)` | `ℓ·(k1·√f + k2·f)` + connectors + surge protection + splitters (3 dB per 2-way split). E.g. 10 m of LMR-400-class cable: 1.2 / 1.9 / 2.3 / 2.7 dB at 800 / 1800 / 2600 / 3500 MHz, plus ~0.5–1.5 dB fittings | needs the real install |
| `M`, `L_modem`, modems, bands for P2 / P4 | defaults in section 0; replaced by the passive install's configuration | defaults set; real install needed |
| `α, γ_min, η_max`, rank thresholds, `g_IRC` | Trainlab active-antenna logs (SINR, rank, throughput) | waiting on data |
| `u_c` load | time-of-day prior; operator data or test runs later | prior |

## 4. Uncertainty and proof

- **Monte Carlo per route.** Draw the uncertain inputs and report the uplift's median and 10–90 % range, plus a
  tornado chart of what drives it (as the reports' sensitivity section does). The inputs are feeder loss, noise
  figures, ρ, rank thresholds, `g_IRC` and load.
- **Paired field test (the proof).** Run both installs on the same train at the same time, so load and position cancel.
  - Take per-second differences in throughput and in time below τ.
  - Get confidence intervals by **block bootstrap** over ~1 km blocks (successive seconds are not independent).
  - Test on block medians with a Wilcoxon signed-rank test.
  - Fit a mixed model: `log T = β·config + time of day + (1 | segment) + (1 | run)`.
- **Unpaired runs.** Match on route km, time of day and network. Compare distributions by quantile regression with
  configuration as a covariate.
- **Other routes.** Regress each trial route's uplift on route features: share noise-limited, median INR, share with
  n78, cutting and tunnel share, speed. Check by leaving one route out, as the national calibration does, then predict
  routes with no trial.

## 5. What this changes in the simulator

The `EDGE_RAIL_ACTIVE_ANTENNA` profile today adds a flat +6 dB to signal quality and ×1.35 to throughput everywhere.
Following this method, it becomes:
- feeder loss removed through the noise term only (1.1);
- 4 branches through diversity and interference rejection (1.2);
- 4 layers when SINR allows (1.3);
- every UK carrier incl. n78 (1.4).

The passive installs become two profiles, `RACK_ROUTER_2X2` (P2) and `RACK_ROUTER_4X4` (P4), each with its cable
losses, ports and bands from section 0. All three are evaluated per sample and summarised per route as in section 2,
so the reports and the viewer can show EDGE Rail against whichever install a fleet has today.
