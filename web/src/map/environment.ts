/**
 * Environment around the train: 3D buildings from the basemap's vector tiles, rails + ballast, catenary masts and
 * wire, and woodland trees sampled from the basemap's landcover polygons. Everything is built from data already
 * on screen (no extra downloads) and only within a window around the train.
 */
import type { Map as MLMap } from 'maplibre-gl'
import { PathLayer } from '@deck.gl/layers'
import { SimpleMeshLayer } from '@deck.gl/mesh-layers'
import type { Layer } from '@deck.gl/core'
import type { State } from '../state'
import { indexAtDistance } from '../data'
import { CONFIG } from '../config'
import { mastMesh, treeMesh } from './train-mesh'
import { offsetRight, whenStyleReady } from './map'

const MAST = mastMesh(), TREE = treeMesh()

/** Pose (lon, lat, z, bearing) at an arbitrary along-track distance, interpolated between 50 m samples. */
export function poseAt(s: State, d: number, useTerrain: boolean): { lon: number; lat: number; z: number; bearing: number } {
  const R = s.data
  const dd = Math.max(0, Math.min(R.meta.length_m, d))
  const i = indexAtDistance(R.distance, dd), j = Math.min(R.n - 1, i + 1)
  const span = R.distance[j] - R.distance[i] || 1
  const f = Math.max(0, Math.min(1, (dd - R.distance[i]) / span))
  let b0 = R.bearing[i], b1 = R.bearing[j]; let db = b1 - b0; if (db > 180) db -= 360; if (db < -180) db += 360
  const z = useTerrain ? (R.elev[i] + (R.elev[j] - R.elev[i]) * f) * CONFIG.terrain.exaggeration : 0
  return { lon: R.lon[i] + (R.lon[j] - R.lon[i]) * f, lat: R.lat[i] + (R.lat[j] - R.lat[i]) * f, z, bearing: (b0 + db * f + 360) % 360 }
}

// ---------------------------------------------------------------- 3D buildings (MapLibre fill-extrusion)
export function setBuildings(map: MLMap, on: boolean, theme: 'light' | 'dark') {
  const id = 'tls-buildings-3d'
  if (!(map as any).style?._loaded) { whenStyleReady(map, () => setBuildings(map, on, theme)); return }
  if (map.getLayer(id)) map.removeLayer(id)
  if (!on) return
  const style = map.getStyle()
  const ref = style.layers.find(l => (l as any)['source-layer'] === 'building' && l.type === 'fill') as any
  if (!ref) return
  const firstSymbol = style.layers.find(l => l.type === 'symbol')?.id
  map.addLayer({
    id, type: 'fill-extrusion', source: ref.source, 'source-layer': 'building', minzoom: 13,
    paint: {
      // OpenMapTiles derives render_height from OSM height / building:levels and writes 5 when neither is tagged;
      // untagged buildings get a stable 5-11 m spread from the feature id so streets do not look extruded from a slab.
      'fill-extrusion-color': ['interpolate', ['linear'], ['coalesce', ['get', 'render_height'], 5], 5, theme === 'dark' ? '#39424b' : '#dfe3e7', 30, theme === 'dark' ? '#4a545e' : '#c9cfd6', 80, theme === 'dark' ? '#5b6672' : '#b4bcc5'],
      'fill-extrusion-height': ['case', ['==', ['coalesce', ['get', 'render_height'], 5], 5], ['+', 5, ['%', ['coalesce', ['id'], 0], 7]], ['coalesce', ['get', 'render_height'], ['get', 'height'], 6]],
      'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], ['get', 'min_height'], 0],
      'fill-extrusion-opacity': 0.9, 'fill-extrusion-vertical-gradient': true,
    },
  } as any, firstSymbol)
  for (const l of ['building', 'building-top']) if (map.getLayer(l)) map.setLayoutProperty(l, 'visibility', 'none')
}

// ---------------------------------------------------------------- track: rails, ballast, catenary
const GAUGE_HALF = 0.72
let trackCache: { data: State['data']; theme: State['theme']; i0: number; i1: number; useTerrain: boolean; layers: Layer[] } | null = null

export function trackLayers(s: State, head: number, useTerrain: boolean): Layer[] {
  const R = s.data
  const win = Math.round(2500 / R.meta.route.sample_spacing_m)
  const i0 = Math.max(0, head - win), i1 = Math.min(R.n - 1, head + win)
  if (trackCache && trackCache.data === R && trackCache.theme === s.theme && trackCache.useTerrain === useTerrain
    && (trackCache.i0 === 0 || head - trackCache.i0 > win * 0.4)
    && (trackCache.i1 === R.n - 1 || trackCache.i1 - head > win * 0.4)) return trackCache.layers
  const zOf = (i: number) => (useTerrain ? R.elev[i] * CONFIG.terrain.exaggeration : 0)
  const ballast: [number, number, number][] = [], railL: [number, number, number][] = [], railR: [number, number, number][] = [], wire: [number, number, number][] = []
  const masts: { position: [number, number, number]; yaw: number }[] = []
  for (let i = i0; i <= i1; i++) {
    const z = zOf(i) + 0.3
    ballast.push([R.lon[i], R.lat[i], z])
    const l = offsetRight(R.lon[i], R.lat[i], R.bearing[i], -GAUGE_HALF), r = offsetRight(R.lon[i], R.lat[i], R.bearing[i], GAUGE_HALF)
    railL.push([l[0], l[1], z + 0.2]); railR.push([r[0], r[1], z + 0.2])
    wire.push([R.lon[i], R.lat[i], z + 5.6])
    if (i % 1 === 0 && !R.inTunnel[i]) { const m = offsetRight(R.lon[i], R.lat[i], R.bearing[i], -2.7); masts.push({ position: [m[0], m[1], z], yaw: 90 - R.bearing[i] }) }
  }
  const layers: Layer[] = [
    new PathLayer({ id: 'track-ballast', data: [{ path: ballast }], getPath: (d: any) => d.path, widthUnits: 'meters', getWidth: 5.2, widthMinPixels: 0, getColor: s.theme === 'dark' ? [58, 60, 62, 255] : [150, 145, 135, 255], capRounded: true, jointRounded: true }),
    new PathLayer({ id: 'track-rail-l', data: [{ path: railL }], getPath: (d: any) => d.path, widthUnits: 'meters', getWidth: 0.16, widthMinPixels: 1, getColor: [138, 142, 146, 255] }),
    new PathLayer({ id: 'track-rail-r', data: [{ path: railR }], getPath: (d: any) => d.path, widthUnits: 'meters', getWidth: 0.16, widthMinPixels: 1, getColor: [138, 142, 146, 255] }),
    new SimpleMeshLayer({ id: 'catenary-masts', data: masts, mesh: MAST as any, getPosition: (m: any) => m.position, getOrientation: (m: any) => [0, m.yaw, 0], getColor: [255, 255, 255, 255], sizeScale: 1 }),
    new PathLayer({ id: 'catenary-wire', data: [{ path: wire }], getPath: (d: any) => d.path, widthUnits: 'meters', getWidth: 0.08, widthMinPixels: 1, getColor: [120, 124, 128, 200] }),
  ]
  trackCache = { data: R, theme: s.theme, i0, i1, useTerrain, layers }
  return layers
}

// ---------------------------------------------------------------- trees from basemap woodland polygons
interface Tree { position: [number, number, number]; scale: number; yaw: number; tint: [number, number, number] }
let treeCache: { key: string; layer: Layer | null; at: number } = { key: '', layer: null, at: 0 }

function pointInRing(x: number, y: number, ring: number[][]): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}
function hash(n: number) { const s = Math.sin(n * 12.9898 + 78.233) * 43758.5453; return s - Math.floor(s) }

export function treeLayer(s: State, map: MLMap, head: number, useTerrain: boolean, now: number): Layer[] {
  if (!s.layers.trees) return []
  const R = s.data
  const key = `${Math.round(head / 8)}:${useTerrain}:${map.getZoom() < 13 ? 'far' : 'near'}`
  if (now - treeCache.at < 1500) return treeCache.layer ? [treeCache.layer] : []
  treeCache.at = now
  if (map.getZoom() < 13) { treeCache = { key, layer: null, at: now }; return [] }
  if (!map.getLayer('landcover')) { treeCache = { key, layer: null, at: now }; return [] }
  const feats = map.queryRenderedFeatures(undefined, { layers: ['landcover'] } as any).filter(f => f.properties?.class === 'wood')
  const trees: Tree[] = []
  const cLon = R.lon[head], cLat = R.lat[head], mPerDegLat = 110540, mPerDegLon = 111320 * Math.cos(cLat * Math.PI / 180)
  const maxR = 1600, CAP = 2600
  const trainZ = useTerrain ? R.elev[head] * CONFIG.terrain.exaggeration : 0
  const seen = new Set<string>()
  for (const f of feats) {
    const g = f.geometry as any
    const polys: number[][][][] = g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : []
    for (const poly of polys) {
      const ring = poly[0]; if (!ring || ring.length < 4) continue
      const holes = poly.slice(1)
      let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity
      for (const [x, y] of ring) { if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y }
      const pkey = `${minx.toFixed(5)}:${miny.toFixed(5)}:${ring.length}`; if (seen.has(pkey)) continue; seen.add(pkey)
      const dx = ((minx + maxx) / 2 - cLon) * mPerDegLon, dy = ((miny + maxy) / 2 - cLat) * mPerDegLat
      const wM = (maxx - minx) * mPerDegLon, hM = (maxy - miny) * mPerDegLat
      if (Math.hypot(dx, dy) > maxR + Math.max(wM, hM) / 2) continue
      // Jittered grid: canopy spacing ~14 m near the line, thinning with distance (LOD) so the cap goes further.
      const seed = Math.round(minx * 1e5) ^ Math.round(miny * 1e5) ^ ring.length
      const stepM = 14
      const gx = stepM / mPerDegLon, gy = stepM / mPerDegLat
      const nx = Math.ceil((maxx - minx) / gx), ny = Math.ceil((maxy - miny) / gy)
      if (nx * ny > 40000) continue   // huge forest: only its near edge would matter, skip rather than stall
      for (let iy = 0; iy < ny && trees.length < CAP; iy++) for (let ix = 0; ix < nx && trees.length < CAP; ix++) {
        const h1 = hash(seed + ix * 73 + iy * 151), h2 = hash(seed + ix * 37 + iy * 97 + 11)
        const x = minx + (ix + 0.15 + 0.7 * h1) * gx, y = miny + (iy + 0.15 + 0.7 * h2) * gy
        if (!pointInRing(x, y, ring)) continue
        if (holes.some(hr => pointInRing(x, y, hr))) continue
        const dist = Math.hypot((x - cLon) * mPerDegLon, (y - cLat) * mPerDegLat)
        if (dist > maxR) continue
        const keep = dist < 500 ? 1 : dist < 1000 ? 0.5 : 0.25          // thin out with distance
        if (hash(seed + ix * 5 + iy * 13 + 3) > keep) continue
        const te = useTerrain ? map.queryTerrainElevation({ lng: x, lat: y }) : null
        const z = te != null && Math.abs(te - trainZ) < 400 ? te : trainZ
        const species = hash(seed + ix * 17 + iy * 29 + 7)
        trees.push({ position: [x, y, z], scale: 0.7 + 0.8 * hash(seed + ix * 7 + iy * 3), yaw: h1 * 360, tint: species < 0.35 ? [190, 210, 175] : species < 0.7 ? [255, 255, 255] : [150, 175, 130] })
        if (trees.length >= CAP) break
      }
      if (trees.length >= CAP) break
    }
    if (trees.length >= CAP) break
  }
  const layer = trees.length ? new SimpleMeshLayer({ id: 'trees', data: trees, mesh: TREE as any, getPosition: (t: any) => t.position, getOrientation: (t: any) => [0, t.yaw, 0], getScale: (t: any) => [t.scale, t.scale, t.scale], getColor: ((t: any) => [t.tint[0], t.tint[1], t.tint[2], 255]) as any, sizeScale: 1 }) : null
  treeCache = { key, layer, at: now }
  return layer ? [layer] : []
}
