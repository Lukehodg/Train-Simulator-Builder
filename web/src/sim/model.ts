/**
 * Client-side scenario model - a faithful mirror of tcs/model/{cellular,satcom,bonding,wifi}.py so the viewer can
 * switch policy / vehicle profile / weather instantly. Inputs are the per-sample base fields the pipeline exported;
 * every constant comes from meta.sim (simulation.yaml) so the two implementations cannot drift on parameters.
 */
import type { LinkResult, RouteData, Scenario, SimResult } from '../types'

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)

function lossTable(q: number, table: [number, number][]): number {
  let out = table[table.length - 1][1]
  for (let k = table.length - 1; k >= 0; k--) if (q < table[k][0]) out = table[k][1]
  return out
}

function dasAt(entries: string[], name: string | null, km: number): boolean {
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

  for (const p of meta.providers) {
    const b = data.base[p.id]
    const r: LinkResult = { q: new Float32Array(n), cap: new Float32Array(n), lat: new Float32Array(n), loss: new Float32Array(n), avail: new Uint8Array(n), rsrp: new Float32Array(n), reason: new Array(n), score: new Float32Array(n) }
    if (p.type === 'cellular') {
      const floor = cfg.throughput.score_floor, expo = cfg.throughput.curve_exponent, h = cfg.handover
      const capPrior = p.capacity_prior_mbps as Record<string, number>
      const unitsFactor = (Number(cfg.units_capacity_factor ?? 1) || 1) * (Number(vprof.capacity_factor ?? 1) || 1)
      for (let i = 0; i < n; i++) {
        const tun = data.inTunnel[i] === 1
        const isDas = tun && dasAt(das, data.tunnelName[i], data.distance[i] / 1000)
        let q = b.qb[i] + vprof.score_offset
        if (isDas) q = cfg.tunnels.das_score; else if (tun) q = cfg.tunnels.default_score
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
        r.rsrp[i] = cfg.rsrp_dbm.at_zero + (cfg.rsrp_dbm.at_one - cfg.rsrp_dbm.at_zero) * q
        r.reason[i] = tun && !isDas ? 'TUNNEL' : r.avail[i] ? 'OK' : 'NO_COVERAGE'
      }
    } else {
      const av = p.availability!, w = av.weather[sc.weather] ?? av.weather.nominal, prior = p.capacity_prior_mbps as number, lp = p.latency_prior_ms!
      const satEnabled = sim.satcom_enabled !== false
      for (let i = 0; i < n; i++) {
        const tun = data.inTunnel[i] === 1
        let sky = clamp(b.qb[i] - w.sky_penalty - 0.18 * data.urban[i] * (1 - data.canopy[i]), 0, 1)
        if (tun) sky = 0
        const temp = b.hp[i] > 0.5
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
        if (!satEnabled) reason = 'SERVICE_UNAVAILABLE'
        r.reason[i] = reason
      }
    }
    links[p.id] = r
  }

  // ---- link manager ----------------------------------------------------------------------------
  const wcfg = sim.wan, wts = wcfg.score_weights, nrm = wcfg.normalisation
  const ids = meta.providers.map(p => p.id)
  const isCell = meta.providers.map(p => p.type === 'cellular')
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
      const nc = clamp(r.cap[i] / nrm.capacity_mbps, 0, 1), nl = clamp(1 - r.lat[i] / nrm.latency_ms, 0, 1), np_ = clamp(1 - r.loss[i] / nrm.packet_loss_pct, 0, 1)
      const stab = data.base[ids[k]].hp[i] > 0 ? 0.3 : 1
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
    if (bonded < 1) s = 0
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
