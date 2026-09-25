import type { Metric, RouteData, SimResult } from '../types'

/** Class index: 0 strong · 1 usable · 2 poor · 3 outage · 4 unknown. Confidence uses its own 5-step ramp. */
export const CLASS_NAMES = ['STRONG', 'USABLE', 'POOR', 'OUTAGE', 'UNKNOWN']
export const WIFI_CLASSES = ['EXCELLENT', 'GOOD', 'USABLE', 'POOR', 'OUTAGE']
export const CLASS_VARS = ['--c-strong', '--c-usable', '--c-poor', '--c-outage', '--c-unknown']
export const CONF_VARS = ['--conf-1', '--conf-2', '--conf-3', '--conf-4', '--conf-5']
export const CONF_LABELS = ['unknown', '0.10–0.39 sparse / synthetic', '0.40–0.69 prediction', '0.70–0.89 calibrated', '0.90–1.00 measured']

export const capClass = (c: number) => (c >= 50 ? 0 : c >= 20 ? 1 : c >= 5 ? 2 : 3)
export const latClass = (l: number) => (l <= 45 ? 0 : l <= 80 ? 1 : l <= 150 ? 2 : 3)
export const lossClass = (p: number) => (p <= 0.5 ? 0 : p <= 2 ? 1 : p <= 5 ? 2 : 3)
export const qClass = (q: number) => (q >= 0.6 ? 0 : q >= 0.4 ? 1 : q >= 0.2 ? 2 : 3)
export const confStep = (c: number) => (c >= 0.9 ? 4 : c >= 0.7 ? 3 : c >= 0.4 ? 2 : c >= 0.1 ? 1 : 0)
export const wifiToClass = (wc: number) => (wc <= 1 ? 0 : wc === 2 ? 1 : wc === 3 ? 2 : 3)

export function classOf(metric: Metric, link: string, i: number, data: RouteData, res: SimResult): number {
  if (metric === 'conf') return confStep(link === 'wan' ? res.conf[i] : data.base[link].conf[i])
  if (link === 'wan') {
    const any = res.active[i] !== 0
    switch (metric) {
      case 'capacity': return any ? capClass(res.bonded[i]) : 3
      case 'latency': return any ? latClass(res.lat[i]) : 3
      case 'loss': return any ? lossClass(res.loss[i]) : 3
      case 'avail': return any ? 0 : 3
      default: return wifiToClass(res.wifiClass[i])
    }
  }
  const r = res.links[link]
  if (!r.avail[i]) return 3
  switch (metric) {
    case 'capacity': case 'wifi': return capClass(r.cap[i])
    case 'latency': return latClass(r.lat[i])
    case 'loss': return lossClass(r.loss[i])
    case 'avail': return 0
    default: return qClass(r.q[i])
  }
}

export const LEGENDS: Record<Metric, string[]> = {
  quality: ['Strong · q ≥ 0.6', 'Usable · 0.4–0.6', 'Poor · 0.2–0.4', 'Outage · < 0.2'],
  capacity: ['≥ 50 Mbps', '20–50 Mbps', '5–20 Mbps', '< 5 Mbps'],
  latency: ['≤ 45 ms', '45–80 ms', '80–150 ms', '> 150 ms'],
  loss: ['≤ 0.5 %', '0.5–2 %', '2–5 %', '> 5 %'],
  avail: ['Available', '', '', 'Unavailable'],
  wifi: ['Excellent / good', 'Usable', 'Poor', 'Outage'],
  conf: CONF_LABELS,
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const v = h.length === 3 ? h.split('').map(c => c + c).join('') : h
  const n = parseInt(v, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}
export function palette(): { cls: [number, number, number][]; conf: [number, number, number][]; accent: [number, number, number]; text: [number, number, number]; muted: [number, number, number] } {
  return {
    cls: CLASS_VARS.map(v => hexToRgb(cssVar(v))),
    conf: CONF_VARS.map(v => hexToRgb(cssVar(v))),
    accent: hexToRgb(cssVar('--accent')), text: hexToRgb(cssVar('--text')), muted: hexToRgb(cssVar('--muted')),
  }
}
