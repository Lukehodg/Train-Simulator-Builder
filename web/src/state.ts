import type { CameraMode, Metric, RouteData, Scenario, SimResult } from './types'

export interface State {
  data: RouteData
  sim: SimResult
  scenario: Scenario
  metric: Metric
  linkVisible: Record<string, boolean>   // provider ids + 'wan'
  layers: { cells: boolean; sky: boolean; stations: boolean; terrain: boolean; labels: boolean; buildings: boolean; trees: boolean; track: boolean }
  camera: CameraMode
  t: number
  playing: boolean
  speed: number
  selected: number | null
  hover: number | null
  theme: 'light' | 'dark'
}

type Listener = (s: State, changed: Set<keyof State>) => void

export class Store {
  state: State
  private listeners: Listener[] = []
  constructor(initial: State) { this.state = initial }
  set(patch: Partial<State>) {
    const changed = new Set<keyof State>()
    for (const k of Object.keys(patch) as (keyof State)[]) { if (this.state[k] !== patch[k]) { (this.state as any)[k] = patch[k]; changed.add(k) } }
    if (changed.size) for (const l of this.listeners) l(this.state, changed)
  }
  on(l: Listener) { this.listeners.push(l); return () => { this.listeners = this.listeners.filter(x => x !== l) } }
}

export const fmtClock = (dep: string, t: number) => {
  const [h0, m0] = dep.split(':').map(Number)
  const s = Math.floor(h0 * 3600 + m0 * 60 + t)
  return [Math.floor(s / 3600) % 24, Math.floor(s / 60) % 60, s % 60].map(v => String(v).padStart(2, '0')).join(':')
}
export const fmtHM = (dep: string, t: number) => fmtClock(dep, t).slice(0, 5)
