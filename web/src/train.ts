/**
 * Train Studio integration: read a `*.train.json` project (the Train Diagram Builder hosted at /train-builder/),
 * derive the onboard architecture and apply it to the simulation config in the browser. Mirrors
 * tcs/sources/train_studio.py; the mapping constants come from meta.sim.train (simulation.yaml).
 */
import { esc } from './html'
import type { Meta, Policy, Vehicle } from './types'

export interface Carriage { index: number; type: string; aps: number; switch: boolean; cellular_units: number; fleet_connect: boolean; satcom_units: number; custom: string[]; aps_connected: number }
export interface TrainDesign {
  title: string; carriages: Carriage[]; cellular_units: number; satcom_units: number; satcom_terminal: string | null; aps_total: number; aps_connected: number
  fleet_connect: boolean; passengers: number; vehicle_profile: Vehicle; policy: Policy; units_capacity_factor: number; ap_capacity_mbps: number; warnings: string[]; source_file?: string | null
}

const SAT_PRESETS = new Set(['starlink', 'oneweb', 'satcom', 'edge-mini'])
export const MAX_PROJECT_BYTES = 5 * 1024 * 1024
const object = (value: any) => value !== null && typeof value === 'object' && !Array.isArray(value)

export function validateProject(project: any): void {
  if (!object(project)) throw new Error('project must be an object')
  if ('title' in project && (typeof project.title !== 'string' || project.title.length > 512)) throw new Error('project title must be text of at most 512 characters')
  if (!Array.isArray(project.cars) || project.cars.length < 1 || project.cars.length > 100) throw new Error('project must contain between 1 and 100 carriages')
  const types = 'types' in project ? project.types : []
  if (!Array.isArray(types) || types.length > 256) throw new Error('project types must be an array of at most 256 entries')
  const ids = new Set<string>()
  for (const item of types) {
    if (!object(item) || typeof item.id !== 'string' || !item.id || ids.has(item.id)) throw new Error('equipment types require unique non-empty text ids')
    ids.add(item.id)
    for (const key of ['name', 'presetId', 'hardwareKind']) if (item[key] != null && typeof item[key] !== 'string') throw new Error(`equipment ${key} must be text`)
    if (typeof item.name !== 'string') throw new Error('equipment name must be text')
    if ('link' in item && typeof item.link !== 'boolean') throw new Error('equipment link must be boolean')
  }
  for (const car of project.cars) {
    if (!object(car) || !['cabL', 'mid', 'cabR'].includes('type' in car ? car.type : 'mid')) throw new Error('carriage type must be cabL, mid or cabR')
    for (const key of ['aps', 'edge']) {
      const value = key in car ? car[key] : 0
      if (!Number.isInteger(value) || value < 0 || value > 1000) throw new Error(`carriage ${key} must be an integer from 0 to 1000`)
    }
    for (const key of ['sw', 'fleet']) if (key in car && typeof car[key] !== 'boolean') throw new Error(`carriage ${key} must be boolean`)
    const custom = 'custom' in car ? car.custom : {}
    if (!object(custom) || Object.values(custom).some(v => typeof v !== 'boolean')) throw new Error('carriage custom equipment must be an object of booleans')
  }
}

export function parseProject(text: string): any {
  if (new TextEncoder().encode(text).byteLength > MAX_PROJECT_BYTES) throw new Error('Train Studio project exceeds the 5 MiB limit')
  const doc = JSON.parse(text)
  if (doc?.app !== 'motion-applied-train-studio' || doc?.version !== 1 || !doc.project) throw new Error('Not a Train Studio v1 project file (*.train.json).')
  validateProject(doc.project)
  return doc.project
}

export function derive(project: any, tcfg: any): TrainDesign {
  validateProject(project)
  const seats: Record<string, number> = tcfg?.seats ?? { cabL: 56, mid: 76, cabR: 56 }
  const satKind: Record<string, string> = tcfg?.satcom_terminal_by_preset ?? { 'edge-mini': 'mini', starlink: 'performance', oneweb: 'performance', satcom: 'performance' }
  const types: Record<string, any> = Object.fromEntries((project.types ?? []).map((t: any) => [t.id, t]))
  const cars: Carriage[] = []
  const warnings: string[] = []
  let satUnits = 0, satTerminal: string | null = null
  ;(project.cars ?? []).forEach((c: any, i: number) => {
    const customOn = Object.entries(c.custom ?? {}).filter(([id, on]) => on && types[id]).map(([id]) => id)
    let satHere = 0
    for (const id of customOn) {
      const t = types[id]
      const isSat = t.hardwareKind === 'satcom' || SAT_PRESETS.has(t.presetId) || ['starlink', 'oneweb', 'satcom'].includes(String(t.name ?? '').trim().toLowerCase())
      if (isSat) { satHere++; satTerminal = satTerminal ?? (satKind[t.presetId ?? String(t.name ?? '').toLowerCase()] ?? tcfg?.default_satcom_terminal ?? 'performance') }
    }
    const aps = Number(c.aps ?? 0), sw = !!c.sw
    const needs = aps > 0 || Number(c.edge ?? 0) > 0 || !!c.fleet || customOn.some(id => types[id].link)
    if (needs && !sw) warnings.push(`Carriage ${i + 1}: equipment present but no switch — its ${aps} access point(s) are not counted`)
    cars.push({ index: i, type: c.type ?? 'mid', aps, switch: sw, cellular_units: Number(c.edge ?? 0), fleet_connect: !!c.fleet, satcom_units: satHere, custom: customOn.map(id => types[id].name), aps_connected: sw ? aps : 0 })
    satUnits += satHere
  })
  const cellUnits = cars.reduce((a, c) => a + c.cellular_units, 0)
  const decay = Number(tcfg?.unit_capacity_decay ?? 0.85)
  if (!Number.isFinite(decay) || decay < 0 || decay > 1) throw new Error('unit_capacity_decay must be between 0 and 1')
  const unitsFactor = decay === 1 ? cellUnits : (1 - Math.pow(decay, cellUnits)) / (1 - decay)
  const fleet = cars.some(c => c.fleet_connect)
  const apsConnected = cars.reduce((a, c) => a + c.aps_connected, 0)
  const d: TrainDesign = {
    title: project.title ?? 'Train Studio project', carriages: cars, cellular_units: cellUnits, satcom_units: satUnits, satcom_terminal: satUnits ? satTerminal : null,
    aps_total: cars.reduce((a, c) => a + c.aps, 0), aps_connected: apsConnected, fleet_connect: fleet,
    passengers: cars.reduce((a, c) => a + Number(seats[c.type] ?? seats.mid ?? 76), 0),
    vehicle_profile: cellUnits > 0 ? (tcfg?.edge_rail_profile ?? 'EDGE_RAIL_ACTIVE_ANTENNA') : 'PASSENGER_HANDSET_INSIDE_CARRIAGE',
    policy: (fleet ? tcfg?.fleet_connect_policy ?? 'PACKET_BONDING' : tcfg?.no_fleet_connect_policy ?? 'FAILOVER') as Policy,
    units_capacity_factor: Math.round(unitsFactor * 1e4) / 1e4, ap_capacity_mbps: Number(tcfg?.ap_capacity_mbps_each ?? 120) * apsConnected, warnings,
  }
  if (cellUnits === 0) d.warnings.push('No EDGE Rail units: cellular modelled as passenger handsets inside the carriage')
  if (satUnits === 0) d.warnings.push('No SATCOM terminal: satellite link disabled')
  if (apsConnected === 0) d.warnings.push('No connected access points: passenger Wi-Fi capacity is zero')
  return d
}

/** Apply a design to a *copy* of meta (sim config + satcom provider priors). Returns the new meta. */
export function applyDesign(meta: Meta, d: TrainDesign | null, baseline: Meta): Meta {
  const m: Meta = JSON.parse(JSON.stringify(baseline))
  if (!d) return m
  m.sim.vehicle.profile = d.vehicle_profile
  m.sim.wan.policy = d.policy
  m.sim.cellular.units_capacity_factor = d.cellular_units ? d.units_capacity_factor : 1
  m.sim.passenger_wifi.passengers = d.passengers
  m.sim.passenger_wifi.ap_capacity_mbps = Math.max(0, d.ap_capacity_mbps)
  m.sim.satcom_enabled = d.satcom_units > 0
  for (const p of m.providers) {
    if (p.type !== 'satcom') continue
    p.enabled = d.satcom_units > 0
    if (d.satcom_terminal && p.capacity_priors && p.capacity_priors[d.satcom_terminal] != null) { p.terminal = d.satcom_terminal; p.capacity_prior_mbps = p.capacity_priors[d.satcom_terminal] }
  }
  m.sim.train = { ...(m.sim.train ?? {}), design: { ...d, n_carriages: d.carriages.length } }
  void meta
  return m
}

export function designFromMeta(meta: Meta): TrainDesign | null {
  const d = meta.sim?.train?.design
  return d && Array.isArray(d.carriages) ? (d as TrainDesign) : null
}

/** Consist strip + equipment table + derived parameters, as HTML. */
export function renderDesign(d: TrainDesign | null, sourceLabel: string): string {
  if (!d) {
    return `<div class="empty">
      <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="16" height="12" rx="2"/><path d="M7 20l-2 2M17 20l2 2M4 10h16"/></svg>
      <p><b>No train design loaded.</b><br>The simulation is using the defaults in <code>simulation.yaml</code>.</p>
      <p>Design a consist in the Train Builder — carriages, EDGE Rail roof units, satcom terminals, switches,
      access points, Fleet Connect — save it as <code>.train.json</code>, then load it here to see what that
      architecture delivers along the route.</p>
    </div>`
  }
  const w = 100 / Math.max(1, d.carriages.length)
  const cars = d.carriages.map(c => {
    const cab = c.type === 'cabL' ? 'cab-l' : c.type === 'cabR' ? 'cab-r' : ''
    const badges = [c.cellular_units ? `<b title="EDGE Rail roof units">${c.cellular_units}×E</b>` : '', c.satcom_units ? `<b title="SATCOM terminal">SAT</b>` : '',
      c.aps ? `<b title="Access points${c.switch ? '' : ' (not connected: no switch)'}" class="${c.switch ? '' : 'warn'}">${c.aps}×AP</b>` : '', c.fleet_connect ? `<b title="Fleet Connect">FC</b>` : '',
      c.switch ? '' : `<b class="warn" title="No switch">no sw</b>`].filter(Boolean).join('')
    return `<div class="car ${cab}" style="width:${w}%"><span class="car-body"></span><span class="car-no">${c.index + 1}</span><div class="car-badges">${badges}</div></div>`
  }).join('')
  const kv = (k: string, v: string) => `<div class="kv"><span>${k}</span><b>${v}</b></div>`
  let h = `<div class="train-head"><div><div class="train-title">${esc(d.title)}</div><div class="hint" style="margin:0">${esc(sourceLabel)}</div></div></div>`
  h += `<div class="consist">${cars}</div>`
  h += `<h4>Equipment</h4><div class="wan">${kv('Carriages', String(d.carriages.length))}${kv('EDGE Rail (cellular roof units)', String(d.cellular_units))}${kv('SATCOM terminals', d.satcom_units ? `${d.satcom_units} · ${esc(d.satcom_terminal)}` : 'none')}${kv('Access points (connected)', `${d.aps_connected} / ${d.aps_total}`)}${kv('Fleet Connect', d.fleet_connect ? 'yes' : 'no')}</div>`
  h += `<h4>What the model uses</h4><div class="wan">${kv('Vehicle profile', d.vehicle_profile === 'EDGE_RAIL_ACTIVE_ANTENNA' ? 'EDGE Rail active antenna' : d.vehicle_profile === 'EXTERNAL_ROOFTOP_ANTENNA' ? 'passive rooftop antenna' : 'handset in carriage')}${kv('Link policy', d.policy.toLowerCase().replace(/_/g, ' '))}${kv('Cellular capacity factor', `×${d.units_capacity_factor.toFixed(2)}`)}${kv('Wi-Fi AP capacity', `${d.ap_capacity_mbps.toFixed(0)} Mbps`)}${kv('Seats (demand model)', String(d.passengers))}${kv('Satcom', d.satcom_units ? `enabled · ${esc(d.satcom_terminal)} terminal` : 'disabled')}</div>`
  if (d.warnings.length) h += `<h4>Notes</h4><ul class="train-warn">${d.warnings.map(x => `<li>${esc(x)}</li>`).join('')}</ul>`
  h += `<p class="hint">Terminal field of view (Mini 35°, Performance 20°) changes the sky-visibility mask, which is computed by the pipeline: run <code>tcs run --train &lt;file&gt;</code> for that part; everything else applies here instantly.</p>`
  return h
}
