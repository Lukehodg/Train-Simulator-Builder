/**
 * Client-side scenario model - a faithful mirror of tcs/model/{cellular,satcom,bonding,wifi}.py so the viewer can
 * switch policy / vehicle profile / weather instantly. Inputs are the per-sample base fields the pipeline exported;
 * every constant comes from meta.sim (simulation.yaml) so the two implementations cannot drift on parameters.
 */
import type { LinkResult, RouteData, Scenario, SimResult } from '../types'

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)
const satHandover = (km: number, sky: number, tunnel: boolean) => !tunnel && sky > 0.5 && Math.sin(km * 7.3) * Math.sin(km * 2.1 + 1) > 0.93

function lossTable(q: number, table: [number, number][]): number {
  let out = table[table.length - 1][1]
  for (let k = table.length - 1; k >= 0; k--) if (q < table[k][0]) out = table[k][1]
  return out
}

/** For each sample: the open-air sample just before and after its tunnel (-1 at a route end) and the distance to each;
 * open-air samples point at themselves. Mirrors portals() in cellular.py. */
function portals(inTunnel: Uint8Array, distance: Float64Array) {
  const n = inTunnel.length, before = new Int32Array(n), after = new Int32Array(n)
  for (let i = 0; i < n; i++) before[i] = !inTunnel[i] ? i : i > 0 ? (inTunnel[i - 1] ? before[i - 1] : i - 1) : -1
  for (let i = n - 1; i >= 0; i--) after[i] = !inTunnel[i] ? i : i < n - 1 ? (inTunnel[i + 1] ? after[i + 1] : i + 1) : -1
  return { before, after }
}

export function dasAt(entries: string[], name: string | null, km: number): boolean {
  for (const e of entries) {
    if (e.startsWith('km:')) { const [a, b] = e.slice(3).split('-').map(Number); if (km >= a && km <= b) return true }
    else if (name === e) return true
  }
  return false
}

export function simulate(data: RouteData, sc: Scenario): SimResult {
  const { n, meta } = data
  const sim = meta.sim
  const links: Record<string, LinkResult> = {}
  const cfg = sim.cellular
  const vprof = sim.vehicle.profiles[sc.vehicle]
  const das: string[] = cfg.tunnels.das_tunnels ?? []
  // networks with a modem on the train (one per EDGE Rail antenna); null = every network. Mirrors fitted_links() in simulate.py.
  const fittedIds: string[] | null = cfg.fitted_networks ?? null
  const fitted = (p: { id: string; type: string }) => p.type !== 'cellular' || fittedIds === null || fittedIds.includes(p.id)

  const decay: number | null = cfg.tunnels.portal_decay_m ?? null
  const port = decay ? portals(data.inTunnel, data.distance) : null

  for (const p of meta.providers) {
    const b = data.base[p.id]
    const r: LinkResult = { q: new Float32Array(n), cap: new Float32Array(n), lat: new Float32Array(n), loss: new Float32Array(n), avail: new Uint8Array(n), rsrp: new Float32Array(n), reason: new Array(n), score: new Float32Array(n) }
    if (p.type === 'cellular') {
      const floor = cfg.throughput.score_floor, expo = cfg.throughput.curve_exponent, h = cfg.handover
      const capPrior = p.capacity_prior_mbps as Record<string, number>
      const unitsFactor = Number(vprof.capacity_factor ?? 1) || 1
      const onTrain = fitted(p)
      const floorQ = cfg.tunnels.default_score
      const open = (k: number) => clamp(b.qb[k] + vprof.score_offset, 0, 1)
      // Signal in a tunnel: each portal's open-air quality fading with distance inside towards the deep-tunnel floor
      // (tunnel_quality() in cellular.py); the better portal wins.
      const inside = (i: number) => {
        if (!port || !decay) return floorQ
        let best = NaN
        for (const [k, d] of [[port.before[i], port.before[i] >= 0 ? data.distance[i] - data.distance[port.before[i]] : 0],
                              [port.after[i], port.after[i] >= 0 ? data.distance[port.after[i]] - data.distance[i] : 0]]) {
          if (k < 0) continue
          const qo = open(k), low = Math.min(qo, floorQ), v = low + (qo - low) * Math.exp(-d / decay)
          if (!(v <= best)) best = v
        }
        return Number.isNaN(best) ? floorQ : best
      }
      for (let i = 0; i < n; i++) {
        const tun = data.inTunnel[i] === 1
        const isDas = tun && dasAt(das, data.tunnelName[i], data.distance[i] / 1000)
        let q = open(i)
        if (isDas) q = cfg.tunnels.das_score; else if (tun) q = inside(i)
        q = clamp(q, 0, 1)
        const hp = b.hp[i] || 0
        const tech = b.tech[i] ?? '4G'
        const prior = capPrior[tech] ?? capPrior['4G']
        const usable = Math.pow(clamp((q - floor) / (1 - floor), 0, 1), expo)
        const speedF = 1 - 0.08 * clamp((data.speed[i] || 0) / 200, 0, 1)
        r.q[i] = q
        r.cap[i] = prior * usable * speedF * (1 - (1 - h.capacity_factor) * hp) * unitsFactor
        r.lat[i] = cfg.latency_ms.base + cfg.latency_ms.at_zero_extra * (1 - q) + h.latency_spike_ms * hp
        r.loss[i] = lossTable(q, cfg.packet_loss_pct) + h.packet_loss_pct * hp
        r.avail[i] = q > floor ? 1 : 0
        r.rsrp[i] = (b.rsrpIntercept?.[i] ?? cfg.rsrp_dbm.at_zero) + (b.rsrpSlope?.[i] ?? (cfg.rsrp_dbm.at_one - cfg.rsrp_dbm.at_zero)) * q
        r.reason[i] = tun && !isDas ? 'TUNNEL' : r.avail[i] ? 'OK' : 'NO_COVERAGE'
        if (!onTrain) { r.avail[i] = 0; r.reason[i] = 'NOT_FITTED' }    // the network's own signal still shows; the train cannot use it
      }
    } else {
      const av = p.availability!, w = av.weather[sc.weather] ?? av.weather.nominal, prior = p.capacity_prior_mbps as number, lp = p.latency_prior_ms!
      const satFitted = sim.satcom_enabled !== false && p.enabled !== false   // switched off, or no terminal in the design
      const satEnabled = satFitted && (!p.service_area || p.service_area.countries.some(c => c.toUpperCase() === meta.route.country.toUpperCase()))
      for (let i = 0; i < n; i++) {
        const tun = data.inTunnel[i] === 1
        let sky = clamp(b.qb[i] - w.sky_penalty - 0.18 * data.urban[i] * (1 - data.canopy[i]), 0, 1)
        if (tun) sky = 0
        const temp = satHandover(data.distance[i] / 1000, sky, tun)
        const avail = satEnabled && !tun && sky >= av.sky_threshold
        r.q[i] = sky
        r.avail[i] = avail ? 1 : 0
        r.cap[i] = avail ? prior * Math.pow(sky, 1.4) * w.capacity_factor * (temp ? 0.3 : 1) : 0
        r.lat[i] = lp.base + lp.obstruction_penalty * (1 - sky) + (temp ? 40 : 0)
        r.loss[i] = avail ? (sky < 0.6 ? 1.2 : 0.3) + (temp ? 3 : 0) : 100
        r.rsrp[i] = NaN
        let reason = sc.weather !== 'nominal' ? 'WEATHER_PENALTY' : 'OPEN_SKY'
        if (temp) reason = 'TEMPORARY_HANDOVER'
        if (data.urban[i] > 0.6) reason = 'URBAN_OBSTRUCTION'
        if (data.cutting[i] > 10) reason = 'DEEP_CUTTING'
        if (data.canopy[i] > 0.5) reason = 'STATION_CANOPY'
        if (tun) reason = 'TUNNEL'
        if (!satEnabled) reason = satFitted ? 'SERVICE_UNAVAILABLE' : 'NOT_FITTED'
        r.reason[i] = reason
      }
    }
    links[p.id] = r
  }

  // ---- link manager ----------------------------------------------------------------------------
  const wcfg = sim.wan, wts = wcfg.score_weights, nrm = wcfg.normalisation
  const ids = meta.providers.map(p => p.id)
  const isCell = meta.providers.map(p => p.type === 'cellular')
  const onTrain = meta.providers.map(fitted)
  const P = ids.length
  const out: SimResult = {
    links, active: new Uint32Array(n), bonded: new Float32Array(n), lat: new Float32Array(n), loss: new Float32Array(n), conf: new Float32Array(n),
    perUser: new Float32Array(n), activeUsers: new Float32Array(n), wifi: new Float32Array(n), wifiClass: new Uint8Array(n),
  }
  const wf = sim.passenger_wifi, ww = wf.score_weights, th = wf.classes
  const score = new Float32Array(P)
  for (let i = 0; i < n; i++) {
    let best = -1, bs = -1, bestCell = -1, bcs = -1, bestSat = -1, bss = -1
    for (let k = 0; k < P; k++) {
      const r = links[ids[k]], conf = data.base[ids[k]].conf[i]
      if (!onTrain[k]) { r.score[i] = score[k] = 0; continue }        // not in the router's link set at all, as in Python
      const nc = clamp(r.cap[i] / nrm.capacity_mbps, 0, 1), nl = clamp(1 - r.lat[i] / nrm.latency_ms, 0, 1), np_ = clamp(1 - r.loss[i] / nrm.packet_loss_pct, 0, 1)
      const handover = isCell[k] ? data.base[ids[k]].hp[i] > 0 : satHandover(data.distance[i] / 1000, r.q[i], data.inTunnel[i] === 1)
      const stab = handover ? 0.3 : 1
      const s = r.avail[i] ? wts.capacity * nc + wts.latency * nl + wts.packet_loss * np_ + wts.stability * stab + wts.confidence * conf : 0
      r.score[i] = score[k] = s
      if (s > bs) { bs = s; best = k }
      if (isCell[k] && s > bcs) { bcs = s; bestCell = k }
      if (!isCell[k] && s > bss) { bss = s; bestSat = k }
    }
    const usable = (k: number) => k >= 0 && links[ids[k]].avail[i] === 1 && score[k] >= wcfg.minimum_link_score
    let mask = 0
    switch (sc.policy) {
      case 'FAILOVER': if (usable(best)) mask = 1 << best; break
      case 'CELLULAR_PRIMARY_STARLINK_BACKUP': if (usable(bestCell)) mask = 1 << bestCell; else if (usable(bestSat)) mask = 1 << bestSat; break
      case 'STARLINK_PRIMARY_CELLULAR_BACKUP': if (usable(bestSat)) mask = 1 << bestSat; else if (usable(bestCell)) mask = 1 << bestCell; break
      default: for (let k = 0; k < P; k++) if (usable(k)) mask |= 1 << k
    }
    if (mask === 0 && best >= 0 && links[ids[best]].avail[i]) mask = 1 << best
    let csum = 0, minLat = Infinity, minLoss = Infinity, wl = 0, confAcc = 0, cnt = 0
    for (let k = 0; k < P; k++) if (mask & (1 << k)) {
      const r = links[ids[k]]; csum += r.cap[i]; minLat = Math.min(minLat, r.lat[i]); minLoss = Math.min(minLoss, r.loss[i]); wl += r.lat[i] * r.cap[i]; confAcc += data.base[ids[k]].conf[i]; cnt++
    }
    let bonded = 0, lat = NaN, loss = 100
    if (cnt) {
      if (sc.policy === 'PACKET_BONDING') { bonded = wcfg.bonding_efficiency * csum * wcfg.congestion_factor; lat = minLat + 5; loss = minLoss * 0.7 }
      else if (sc.policy === 'WEIGHTED_LOAD_BALANCING' || sc.policy === 'POLICY_BASED') { bonded = csum * wcfg.load_balancing_efficiency; lat = csum > 0 ? wl / csum : minLat; loss = minLoss }
      else { bonded = csum; lat = minLat; loss = minLoss }
    }
    out.active[i] = mask; out.bonded[i] = bonded; out.lat[i] = lat; out.loss[i] = loss; out.conf[i] = cnt ? confAcc / cnt : 0
    // ---- passenger Wi-Fi ------------------------------------------------------------------------
    const km = data.distance[i] / 1000
    const load = wf.load_factor_range[0] + (wf.load_factor_range[1] - wf.load_factor_range[0]) * (0.5 + 0.5 * Math.sin(km / 90))
    const active = wf.passengers * load * wf.active_share
    const wanCap = Math.min(bonded, wf.ap_capacity_mbps)
    const perUser = active > 0 ? Math.min(wf.per_user_cap_mbps, wanCap * 0.85 / Math.max(active, 1)) : 0
    const l = Number.isNaN(lat) ? 999 : lat
    let s = 100 * (ww.per_user * clamp(perUser / wf.per_user_target_mbps, 0, 1) + ww.latency * clamp(1 - l / 250, 0, 1) + ww.loss * clamp(1 - loss / 8, 0, 1))
    if (wanCap < 1) s = 0
    out.perUser[i] = perUser; out.activeUsers[i] = active; out.wifi[i] = s
    out.wifiClass[i] = s >= th.EXCELLENT ? 0 : s >= th.GOOD ? 1 : s >= th.USABLE ? 2 : s >= th.POOR ? 3 : 4
  }
  return out
}

/** Parity check against the Python outputs shipped in the bundle (default scenario only). */
export function parityReport(data: RouteData, res: SimResult): { maxBondedDiff: number; classAgreement: number } {
  let maxDiff = 0, agree = 0
  const names = ['EXCELLENT', 'GOOD', 'USABLE', 'POOR', 'OUTAGE']
  for (let i = 0; i < data.n; i++) {
    const d = Math.abs(res.bonded[i] - data.py.bonded[i]); if (d > maxDiff) maxDiff = d
    if (names[res.wifiClass[i]] === data.py.cls[i]) agree++
  }
  return { maxBondedDiff: maxDiff, classAgreement: agree / data.n }
}
