import { tableFromIPC, Table } from 'apache-arrow'
import { CONFIG } from './config'
import type { CellRecord, Meta, ProviderBase, RouteData } from './types'

function f32(t: Table, name: string, n: number): Float32Array {
  const v = t.getChild(name)
  const out = new Float32Array(n)
  if (!v) return out
  for (let i = 0; i < n; i++) { const x = v.get(i); out[i] = x == null ? NaN : Number(x) }
  return out
}
function f64(t: Table, name: string, n: number): Float64Array {
  const v = t.getChild(name)
  const out = new Float64Array(n)
  if (!v) return out
  for (let i = 0; i < n; i++) { const x = v.get(i); out[i] = x == null ? NaN : Number(x) }
  return out
}
function u8(t: Table, name: string, n: number): Uint8Array {
  const v = t.getChild(name)
  const out = new Uint8Array(n)
  if (!v) return out
  for (let i = 0; i < n; i++) out[i] = v.get(i) ? 1 : 0
  return out
}
function str(t: Table, name: string, n: number): (string | null)[] {
  const v = t.getChild(name)
  const out: (string | null)[] = new Array(n).fill(null)
  if (!v) return out
  for (let i = 0; i < n; i++) { const x = v.get(i); out[i] = x == null ? null : String(x) }
  return out
}

export async function loadRoute(routeId = CONFIG.defaultRoute): Promise<RouteData> {
  const base = `${CONFIG.dataRoot}/${routeId}`
  const [metaRes, arrowRes, cellsRes] = await Promise.all([fetch(`${base}/meta.json`), fetch(`${base}/route.arrow`), fetch(`${base}/cells.arrow`)])
  if (!metaRes.ok || !arrowRes.ok) throw new Error(`Route bundle not found under ${base}. Run \`tcs run --offline\` first.`)
  const meta: Meta = await metaRes.json()
  const t = tableFromIPC(await arrowRes.arrayBuffer())
  const n = t.numRows
  if (n < 2) throw new Error('Route bundle must contain at least two samples.')
  const providers: Record<string, ProviderBase> = {}
  for (const p of meta.providers) {
    providers[p.id] = {
      rsrpSlope: t.getChild(`${p.id}_rsrp_slope`) ? f32(t, `${p.id}_rsrp_slope`, n) : undefined,
      rsrpIntercept: t.getChild(`${p.id}_rsrp_intercept`) ? f32(t, `${p.id}_rsrp_intercept`, n) : undefined,
      qb: f32(t, `${p.id}_qb`, n), hp: f32(t, `${p.id}_hp`, n), conf: f32(t, `${p.id}_conf`, n), tech: str(t, `${p.id}_tech`, n),
      cell: str(t, `${p.id}_cell`, n), celld: f32(t, `${p.id}_celld`, n), src: str(t, `${p.id}_src`, n), rsrp: f32(t, `${p.id}_sig`, n),
    }
  }
  const cells: CellRecord[] = []
  const cellIndex = new Map<string, CellRecord>()
  if (cellsRes.ok) {
    const ct = tableFromIPC(await cellsRes.arrayBuffer())
    const m = ct.numRows
    const key = str(ct, 'cell_key', m), prov = str(ct, 'provider_id', m), radio = str(ct, 'radio', m), src = str(ct, 'source', m)
    const lat = f64(ct, 'latitude', m), lon = f64(ct, 'longitude', m), samples = f32(ct, 'samples', m), range = f32(ct, 'range_m', m)
    for (let i = 0; i < m; i++) {
      const c: CellRecord = { key: key[i]!, provider: prov[i]!, radio: radio[i] ?? '', lat: lat[i], lon: lon[i], samples: samples[i], range: range[i], source: src[i] ?? '' }
      cells.push(c); cellIndex.set(c.key, c)
    }
  }
  return {
    n, meta,
    distance: f64(t, 'distance_m', n), lat: f64(t, 'latitude', n), lon: f64(t, 'longitude', n), elev: f32(t, 'elevation_m', n), terrain: f32(t, 'terrain_m', n),
    bearing: f32(t, 'bearing_deg', n), inTunnel: u8(t, 'in_tunnel', n), tunnelName: str(t, 'tunnel_name', n), cutting: f32(t, 'cutting_depth_m', n),
    canopy: f32(t, 'canopy_probability', n), urban: f32(t, 'urban_density', n), sky: f32(t, 'sky_visibility', n), speed: f32(t, 'speed_kph', n),
    t: f64(t, 'sim_seconds', n), nextStation: str(t, 'next_station', n), ttn: f32(t, 'time_to_next_station_s', n), stationNear: str(t, 'station_nearby', n),
    base: providers,
    py: { bonded: f32(t, 'wan_bonded_capacity_mbps', n), wifi: f32(t, 'wan_wifi_service_score', n), cls: str(t, 'wan_service_class', n), active: str(t, 'wan_active_links', n) },
    cells, cellIndex,
  }
}

/** Binary search: last index with t[i] <= time. */
export function indexAtTime(t: Float64Array, time: number): number {
  let lo = 0, hi = t.length - 1
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (t[m] <= time) lo = m; else hi = m - 1 }
  return lo
}
export function indexAtDistance(d: Float64Array, dist: number): number {
  let lo = 0, hi = d.length - 1
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (d[m] <= dist) lo = m; else hi = m - 1 }
  return lo
}

/** A route's coverage counts as live when at least this share of it comes from Ofcom (same rule as tcs/report.py). */
export const LIVE_COVERAGE_MIN = 0.9

/** Share of the route's coverage prior that comes from Ofcom (API or Connected Nations). Bundles built before the
 *  share was recorded count as fully live when any Ofcom source is listed, as they used to. */
export function liveCoverageShare(m: Meta): number {
  if (m.coverage_share) return Object.entries(m.coverage_share).filter(([k]) => k.startsWith('ofcom')).reduce((a, [, v]) => a + v, 0)
  return m.coverage_sources.some(s => s.startsWith('ofcom')) ? 1 : 0
}

/** Whether the train has this link: a fitted mobile network (sim.cellular.fitted_networks, null = all of them) or a
 *  satellite terminal that is switched on. The same rule as fitted_links() in simulate.py and the satcom model. */
export function linkFitted(m: Meta, p: { id: string; type: string; enabled?: boolean }): boolean {
  if (p.type === 'cellular') { const f = m.sim.cellular.fitted_networks as string[] | null | undefined; return f == null || f.includes(p.id) }
  return m.sim.satcom_enabled !== false && p.enabled !== false
}
