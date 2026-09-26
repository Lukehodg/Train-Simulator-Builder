import * as maplibregl from 'maplibre-gl'
import { Map as MLMap, type StyleSpecification } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { AmbientLight, DirectionalLight, LightingEffect, type Layer, type PickingInfo } from '@deck.gl/core'
import { CONFIG } from '../config'
import { cssVar } from '../sim/classify'
import type { CameraMode, RouteData } from '../types'

maplibregl.setWorkerUrl(workerUrl)

export interface MapCtx {
  map: MLMap
  overlay: MapboxOverlay
  terrainOn: boolean
  setTerrain(on: boolean): void
  setLabels(on: boolean): void
  setTheme(theme: 'light' | 'dark'): void
  setLayers(layers: Layer[]): void
  fitRoute(): void
  onUserInteract(cb: () => void): void
  drive(mode: CameraMode, lon: number, lat: number, bearingDeg: number, dt: number): void
  ready: Promise<void>
}

/** Stand-in basemap: just the theme's surface colour. Used when the real style can't be fetched. */
const plainStyle = (): StyleSpecification => ({ version: 8, sources: {}, layers: [{ id: 'background', type: 'background', paint: { 'background-color': cssVar('--surface-3') } }] })

export function createMap(container: HTMLElement, data: RouteData, theme: 'light' | 'dark', getTooltip: (info: PickingInfo) => any, onBasemapDown?: () => void): MapCtx {
  const bounds = routeBounds(data)
  const styleUrl = (t: 'light' | 'dark') => (t === 'dark' ? CONFIG.basemap.dark : CONFIG.basemap.light)
  let pendingStyle = styleUrl(theme)
  const map = new maplibregl.Map({
    container, style: pendingStyle, bounds, fitBoundsOptions: { padding: 40 },
    maxPitch: 75, attributionControl: { compact: true }, canvasContextAttributes: { antialias: true },
  })
  map.on('error', e => console.error('[maplibre]', e.error?.message ?? e))

  // Without its style the map never fires 'style.load', so `ready` (and the whole app) would wait forever.
  // If the style request fails, or hasn't parsed within the timeout, switch to the plain style and carry on.
  let basemapDown = false
  const fallBack = () => {
    if (basemapDown) return
    basemapDown = true
    console.warn(`[tls] basemap style unavailable (${pendingStyle}); using a plain map`)
    map.setStyle(plainStyle(), { diff: false })
    onBasemapDown?.()
  }
  map.on('error', e => {
    const url = (e.error as { url?: string } | undefined)?.url
    if (!styleParsed(map) && (url == null || url === pendingStyle)) fallBack()   // a tile or sprite failing is not the style failing
  })
  setTimeout(() => { if (!styleParsed(map)) fallBack() }, CONFIG.basemap.timeoutMs)
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right')
  map.addControl(new maplibregl.ScaleControl({ maxWidth: 120 }), 'bottom-left')
  const lighting = new LightingEffect({ ambient: new AmbientLight({ color: [255, 255, 255], intensity: 1.5 }), sun: new DirectionalLight({ color: [255, 250, 240], intensity: 1.3, direction: [-1, -2, -2.5] }) })
  const overlay = new MapboxOverlay({ interleaved: false, layers: [], getTooltip, effects: [lighting], onError: (e: Error) => console.warn('[deck-error]', e.stack),
    // deck.gl makes its canvas a tab stop for its own keyboard controller, but MapLibre owns the camera here: drop the empty, unnamed stop
    onLoad: () => map.getContainer().querySelector<HTMLElement>('#deckgl-overlay')?.setAttribute('tabindex', '-1') })
  map.addControl(overlay as any)

  const ctx: MapCtx = {
    map, overlay, terrainOn: false,
    setTerrain(on) { ctx.terrainOn = on; applyTerrain(map, on) },
    setLabels(on) { whenStyleReady(map, () => { for (const l of map.getStyle()?.layers ?? []) if (l.type === 'symbol') map.setLayoutProperty(l.id, 'visibility', on ? 'visible' : 'none') }) },
    setTheme(t) {
      map.once('style.load', () => whenStyleReady(map, () => applyTerrain(map, ctx.terrainOn)))
      if (basemapDown) { map.setStyle(plainStyle(), { diff: false }); return }   // don't retry a basemap that already failed
      pendingStyle = styleUrl(t); map.setStyle(pendingStyle)
    },
    setLayers(layers) { overlay.setProps({ layers }) },
    fitRoute() { map.fitBounds(bounds, { padding: 60, pitch: 0, bearing: 0, duration: 800 }) },
    onUserInteract(cb) { for (const ev of ['dragstart', 'wheel', 'rotatestart', 'pitchstart'] as const) map.on(ev, (e: any) => { if (e?.originalEvent) cb() }) },
    drive(mode, lon, lat, bearing, _dt) {
      if (mode === 'chase') {
        const ahead = offsetLonLat(lon, lat, bearing, CONFIG.camera.chaseAheadM)
        map.jumpTo({ center: ahead, bearing: bearing + CONFIG.camera.chaseYaw, pitch: CONFIG.camera.chasePitch, zoom: CONFIG.camera.chaseZoom })
      } else if (mode === 'oblique') {
        map.jumpTo({ center: [lon, lat], bearing: bearing + CONFIG.camera.obliqueYaw, pitch: CONFIG.camera.obliquePitch, zoom: CONFIG.camera.obliqueZoom })
      }
    },
    ready: new Promise<void>(resolve => whenStyleReady(map, resolve)),
  }
  map.on('style.load', () => { /* labels visibility is re-applied by the caller after theme switches */ })
  return ctx
}

/** Run fn once the style JSON is parsed and its layers exist. (isStyleLoaded() also waits for every tile, which
 * never settles while the chase camera keeps moving, so we look at the style object itself.) */
const styleParsed = (map: MLMap) => Boolean((map as any).style && (map as any).style._loaded)
export function whenStyleReady(map: MLMap, fn: () => void) {
  if (styleParsed(map)) fn(); else map.once('style.load', () => fn())
}

function applyTerrain(map: MLMap, on: boolean) {
  if (!styleParsed(map)) { whenStyleReady(map, () => applyTerrain(map, on)); return }
  if (on) {
    if (!map.getSource('terrain-dem')) {
      map.addSource('terrain-dem', { type: 'raster-dem', tiles: CONFIG.terrain.tiles, encoding: CONFIG.terrain.encoding, tileSize: CONFIG.terrain.tileSize, maxzoom: CONFIG.terrain.maxzoom, attribution: 'Terrain: Mapzen / AWS Open Data' })
    }
    if (!map.getLayer('hillshade')) {
      const firstSymbol = map.getStyle().layers.find(l => l.type === 'symbol')?.id
      map.addLayer({ id: 'hillshade', type: 'hillshade', source: 'terrain-dem', paint: { 'hillshade-exaggeration': 0.35, 'hillshade-shadow-color': '#4a5560', 'hillshade-highlight-color': '#ffffff', 'hillshade-accent-color': '#8a9199' } }, firstSymbol)
    }
    map.setTerrain({ source: 'terrain-dem', exaggeration: CONFIG.terrain.exaggeration })
  } else {
    map.setTerrain(null)
    if (map.getLayer('hillshade')) map.removeLayer('hillshade')
  }
}

export function routeBounds(data: RouteData): [[number, number], [number, number]] {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity
  for (let i = 0; i < data.n; i += 5) { const x = data.lon[i], y = data.lat[i]; if (x < w) w = x; if (x > e) e = x; if (y < s) s = y; if (y > n) n = y }
  return [[w, s], [e, n]]
}

/** Move a lon/lat by `metres` along a compass bearing (clockwise from north). */
export function offsetLonLat(lon: number, lat: number, bearingDeg: number, metres: number): [number, number] {
  const b = bearingDeg * Math.PI / 180
  const dx = Math.sin(b) * metres, dy = Math.cos(b) * metres
  return [lon + dx / (111320 * Math.cos(lat * Math.PI / 180)), lat + dy / 110540]
}
/** Perpendicular offset: positive = right-hand side of travel. */
export function offsetRight(lon: number, lat: number, bearingDeg: number, metres: number): [number, number] {
  return offsetLonLat(lon, lat, bearingDeg + 90, metres)
}
