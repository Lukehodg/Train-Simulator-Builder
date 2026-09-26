import { PathLayer, ScatterplotLayer, TextLayer, LineLayer, PolygonLayer, ColumnLayer } from '@deck.gl/layers'
import { SimpleMeshLayer } from '@deck.gl/mesh-layers'
import type { Layer } from '@deck.gl/core'
import { CONFIG } from '../config'
import type { State } from '../state'
import { classOf, palette } from '../sim/classify'
import { offsetLonLat, offsetRight } from './map'
import { carriageMesh, edgeRailMesh, satcomMesh, type MeshData } from './train-mesh'
import { poseAt } from './environment'

// deck.gl builds one glyph atlas per font and caches it, so labels drawn before the bundled face has loaded would keep
// the fallback for good. Use the system face until IBM Plex Sans is in, then switch (a new font name means a new atlas).
let labelFont = 'system-ui, sans-serif'
export const labelFontReady: Promise<void> = (document.fonts?.load('500 12px "IBM Plex Sans"') ?? Promise.resolve([]))
  .then(() => { labelFont = '"IBM Plex Sans", system-ui, sans-serif' }, () => {})

export interface Run { lane: string; cls: number; i0: number; i1: number; path: [number, number, number][] }


/** Lane definitions: centre = combined WAN; cellular lanes offset to the right; satcom floats above the centre. */
export function lanes(s: State) {
  const out: { key: string; link: string; offset: number; width: number; lift: number; sky: boolean }[] = []
  const r = CONFIG.ribbons
  out.push({ key: 'wan', link: 'wan', offset: -r.laneOffsetM, width: r.centreWidthM, lift: r.groundLiftM, sky: false })
  let k = 1
  for (const p of s.data.meta.providers) {
    if (p.type === 'cellular') { out.push({ key: p.id, link: p.id, offset: r.laneOffsetM * k, width: r.laneWidthM, lift: r.groundLiftM + 2 * k, sky: false }); k++ }
    else out.push({ key: p.id, link: p.id, offset: 0, width: r.laneWidthM, lift: r.skyLiftM, sky: true })
  }
  return out
}

export function elevationOf(s: State, i: number, useTerrain: boolean): number {
  return useTerrain ? s.data.elev[i] * CONFIG.terrain.exaggeration : 0
}

/** Build colour runs for every visible lane (called when metric / scenario / theme / terrain change). */
export function buildRuns(s: State, useTerrain: boolean): Run[] {
  const runs: Run[] = []
  const d = s.data
  for (const L of lanes(s)) {
    if (!s.linkVisible[L.key]) continue
    let cur: Run | null = null
    for (let i = 0; i < d.n; i++) {
      const c = classOf(s.metric, L.link, i, d, s.sim)
      const [lon, lat] = L.offset !== 0 ? offsetRight(d.lon[i], d.lat[i], d.bearing[i], L.offset) : [d.lon[i], d.lat[i]]
      const z = elevationOf(s, i, useTerrain) + L.lift
      const pt: [number, number, number] = [lon, lat, z]
      if (!cur || cur.cls !== c) {
        if (cur) { cur.path.push(pt); cur.i1 = i; runs.push(cur) }   // overlap one point so runs join without gaps
        cur = { lane: L.key, cls: c, i0: i, i1: i, path: [pt] }
      } else cur.path.push(pt)
    }
    if (cur) { cur.i1 = d.n - 1; runs.push(cur) }
  }
  return runs
}

export function ribbonLayers(s: State, runs: Run[], onPick: (i: number) => void): Layer[] {
  const pal = palette()
  const byLane = new Map<string, Run[]>()
  for (const r of runs) { const a = byLane.get(r.lane) ?? []; a.push(r); byLane.set(r.lane, a) }
  const out: Layer[] = []
  for (const L of lanes(s)) {
    const data = byLane.get(L.key); if (!data) continue
    out.push(new PathLayer<Run>({
      id: `ribbon-${L.key}`, data, getPath: r => r.path, widthUnits: 'meters', getWidth: L.width, widthMinPixels: L.key === 'wan' ? 3.5 : 2, widthMaxPixels: L.key === 'wan' ? 9 : 6,
      capRounded: true, jointRounded: true, billboard: false, pickable: true, autoHighlight: true, highlightColor: [255, 255, 255, 90],
      getColor: r => { const c = s.metric === 'conf' ? pal.conf[r.cls] : pal.cls[r.cls]; return [c[0], c[1], c[2], L.sky ? 215 : 255] },
      onClick: info => { if (info.object && info.coordinate) { onPick(nearestInRun(s, info.object, info.coordinate as [number, number])); return true } return false },
      updateTriggers: { getColor: [s.metric, s.theme] }, parameters: { depthTest: !L.sky } as any,
    }))
  }
  return out
}

function nearestInRun(s: State, r: Run, c: [number, number]): number {
  let best = r.i0, bd = Infinity
  for (let i = r.i0; i <= r.i1; i++) { const dx = (s.data.lon[i] - c[0]) * Math.cos(c[1] * Math.PI / 180), dy = s.data.lat[i] - c[1]; const d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i } }
  return best
}

export function stationLayers(s: State, useTerrain: boolean): Layer[] {
  if (!s.layers.stations) return []
  const pal = palette()
  const st = s.data.meta.stations.map(x => ({ ...x, z: elevationOf(s, x.sample_id, useTerrain) + 4 }))
  return [
    new ScatterplotLayer({ id: 'stations', data: st, getPosition: (d: any) => [d.lon, d.lat, d.z], getRadius: (d: any) => (d.stop ? 90 : 60), radiusMinPixels: 3, radiusMaxPixels: 9,
      getFillColor: (d: any) => (d.stop ? [...pal.accent, 255] : [...pal.muted, 255]) as any, getLineColor: [255, 255, 255, 220], lineWidthMinPixels: 1.5, stroked: true, pickable: true }),
    new TextLayer({ id: 'station-labels', data: st, getPosition: (d: any) => [d.lon, d.lat, d.z], getText: (d: any) => `${d.name}${d.scheduled ? '  ' + d.scheduled : ''}`, getSize: 12,
      getColor: [...pal.text, 255] as any, background: true, getBackgroundColor: s.theme === 'dark' ? [15, 18, 21, 215] : [255, 255, 255, 220], backgroundPadding: [6, 3, 6, 3],
      getPixelOffset: [0, -16], fontFamily: labelFont, fontWeight: 500, sizeUnits: 'pixels', getTextAnchor: 'middle', getAlignmentBaseline: 'bottom',
      characterSet: 'auto', billboard: true, updateTriggers: { getColor: [s.theme], getBackgroundColor: [s.theme] } }),
  ]
}

export function cellLayers(s: State, i: number, useTerrain: boolean): Layer[] {
  if (!s.layers.cells) return []
  const pal = palette()
  const d = s.data
  const z0 = elevationOf(s, i, useTerrain)
  const lines: { from: [number, number, number]; to: [number, number, number]; provider: string; key: string }[] = []
  for (const p of d.meta.providers) {
    if (p.type !== 'cellular' || !s.linkVisible[p.id]) continue
    const key = d.base[p.id].cell[i]; if (!key) continue
    const c = d.cellIndex.get(key); if (!c) continue
    lines.push({ from: [d.lon[i], d.lat[i], z0 + 14], to: [c.lon, c.lat, z0 + 40], provider: p.id, key })
  }
  const visible = new Set(d.meta.providers.filter(p => s.linkVisible[p.id]).map(p => p.id))
  return [
    new ColumnLayer({ id: 'cells', data: d.cells.filter(c => visible.has(c.provider)), diskResolution: 6, radius: 28, extruded: true, getPosition: (c: any) => [c.lon, c.lat, z0 - 5],
      getElevation: 60, getFillColor: [...pal.muted, 190] as any, pickable: true, elevationScale: 1 }),
    new LineLayer({ id: 'serving-lines', data: lines, getSourcePosition: (l: any) => l.from, getTargetPosition: (l: any) => l.to, getColor: [...pal.text, 170] as any, getWidth: 1.5, widthUnits: 'pixels' }),
  ]
}

export interface ConsistCar { kind: 'cabF' | 'cabR' | 'mid'; edge: number; sat: number }

/** Consist to draw: the loaded Train Studio design, else a 5-car default reflecting the vehicle profile. */
export function consistOf(s: State): ConsistCar[] {
  const d = s.data.meta.sim?.train?.design
  const usingDesign = s.data.meta.sim?.active_preset === 'design' && d && Array.isArray(d.carriages)
  if (usingDesign) {
    return d.carriages.map((c: any, k: number) => ({ kind: k === 0 ? 'cabF' : k === d.carriages.length - 1 ? 'cabR' : 'mid', edge: c.cellular_units ?? 0, sat: c.satcom_units ?? 0 }))
  }
  const edge = s.scenario.vehicle === 'EDGE_RAIL_ACTIVE_ANTENNA' ? 1 : 0
  const sat = s.data.meta.sim?.satcom_enabled === false ? 0 : 1
  return [{ kind: 'cabF', edge, sat }, { kind: 'mid', edge: 0, sat: 0 }, { kind: 'mid', edge, sat: 0 }, { kind: 'mid', edge: 0, sat: 0 }, { kind: 'cabR', edge, sat: 0 }]
}

const MESHES: Record<string, MeshData> = {}
const meshFor = (kind: ConsistCar['kind']) => (MESHES[kind] ??= carriageMesh(kind, CAR_LEN))
const EDGE_MESH = edgeRailMesh(), SAT_MESH = satcomMesh()
const CAR_LEN = 26, CAR_GAP = 0.6

export function trainLayers(s: State, i: number, frac: number, useTerrain: boolean, zoom: number): Layer[] {
  const d = s.data
  const j = Math.min(d.n - 1, i + 1)
  const headDist = d.distance[i] + (d.distance[j] - d.distance[i]) * frac
  const head = poseAt(s, headDist, useTerrain)
  const lon = head.lon, lat = head.lat, z = head.z + CONFIG.ribbons.groundLiftM * 0 + 0.3
  const pal = palette()
  // keep the train legible when zoomed out: scale it up (schematic beyond ~z15.6)
  const scale = Math.max(1, Math.min(6, Math.pow(2, 15.6 - zoom)))
  const cars = consistOf(s)
  const carData: { position: [number, number, number]; yaw: number; kind: ConsistCar['kind'] }[] = []
  const units: { position: [number, number, number]; yaw: number }[] = []
  const sats: { position: [number, number, number]; yaw: number }[] = []
  let cursor = headDist - 2 * scale
  cars.forEach(c => {
    const centre = cursor - (CAR_LEN * scale) / 2
    const pose = poseAt(s, centre, useTerrain)
    carData.push({ position: [pose.lon, pose.lat, pose.z + 0.3], yaw: 90 - pose.bearing, kind: c.kind })
    for (let k = 0; k < Math.min(2, c.edge); k++) {
      const p = poseAt(s, centre + (c.edge === 1 ? 5 : k === 0 ? 8 : -4) * scale, useTerrain)
      units.push({ position: [p.lon, p.lat, p.z + 0.3 + 3.95 * scale], yaw: 90 - p.bearing })
    }
    if (c.sat > 0) { const p = poseAt(s, centre - 8 * scale, useTerrain); sats.push({ position: [p.lon, p.lat, p.z + 0.3 + 3.95 * scale], yaw: 90 - p.bearing }) }
    cursor = centre - (CAR_LEN * scale) / 2 - CAR_GAP * scale
  })
  const material = { ambient: 0.55, diffuse: 0.6, shininess: 32, specularColor: [60, 60, 60] as [number, number, number] }
  const layers: Layer[] = []
  for (const kind of ['cabF', 'mid', 'cabR'] as const) {
    const data = carData.filter(c => c.kind === kind); if (!data.length) continue
    layers.push(new SimpleMeshLayer({ id: `train-${kind}`, data, mesh: meshFor(kind) as any, getPosition: (c: any) => c.position, getOrientation: (c: any) => [0, c.yaw, 0], getColor: [255, 255, 255, 255], sizeScale: scale, material, pickable: false }))
  }
  if (units.length) layers.push(new SimpleMeshLayer({ id: 'train-edge-rail', data: units, mesh: EDGE_MESH as any, getPosition: (u: any) => u.position, getOrientation: (u: any) => [0, u.yaw, 0], getColor: [255, 255, 255, 255], sizeScale: scale, material }))
  if (sats.length) layers.push(new SimpleMeshLayer({ id: 'train-satcom', data: sats, mesh: SAT_MESH as any, getPosition: (u: any) => u.position, getOrientation: (u: any) => [0, u.yaw, 0], getColor: [255, 255, 255, 255], sizeScale: scale, material }))
  // sky window above the head car
  const sat = d.meta.providers.find(p => p.type === 'satcom')
  const skyLink = sat ? s.sim.links[sat.id] : null
  const skyOn = s.layers.sky && !!sat && s.linkVisible[sat!.id] && d.inTunnel[i] === 0
  if (skyOn) {
    const skyOk = skyLink ? skyLink.avail[i] === 1 : false
    const skyCol = skyOk ? pal.cls[0] : pal.cls[3]
    const skyR = (14 + 40 * (skyLink ? skyLink.q[i] : 0)) * scale
    const top = z + 55 * scale
    layers.push(new PolygonLayer({ id: 'sky-window', data: [{ polygon: circle(lon, lat, skyR, top) }], getPolygon: (x: any) => x.polygon, filled: true, stroked: true, getFillColor: [...skyCol, 26] as any, getLineColor: [...skyCol, 200] as any, lineWidthMinPixels: 1.5, getLineWidth: 2, extruded: false, parameters: { depthTest: false } as any }))
    layers.push(new LineLayer({ id: 'sky-beam', data: [{ a: [lon, lat, z + 4 * scale], b: [lon, lat, top] }], getSourcePosition: (x: any) => x.a, getTargetPosition: (x: any) => x.b, getColor: [...skyCol, 200] as any, getWidth: 2, widthUnits: 'pixels' }))
  }
  return layers
}

function circle(lon: number, lat: number, r: number, z: number): [number, number, number][] {
  const pts: [number, number, number][] = []
  for (let k = 0; k <= 36; k++) { const [x, y] = offsetLonLat(lon, lat, k * 10, r); pts.push([x, y, z]) }
  return pts
}
