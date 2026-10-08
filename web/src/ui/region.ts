/** Region switch (UK / USA) in the app bar. A region is a country code on the route catalogue (index.json `country`,
 * GB when absent). Switching opens the route last viewed in that region, else its first route. */
import { CONFIG } from '../config'

export type Region = 'GB' | 'US'
export const REGIONS: { id: Region; label: string; name: string }[] = [
  { id: 'GB', label: 'UK', name: 'United Kingdom' },
  { id: 'US', label: 'USA', name: 'United States' },
]

export interface CatalogueRoute { id: string; country?: string; [k: string]: unknown }

export const regionOf = (r: { country?: string } | undefined): Region => (r?.country === 'US' ? 'US' : 'GB')

export function normaliseRegion(s: string | null | undefined): Region | null {
  const u = (s ?? '').trim().toUpperCase()
  return u === 'US' || u === 'USA' ? 'US' : u === 'GB' || u === 'UK' ? 'GB' : null
}

const KEY = (r: Region) => `tls-route-${r}`
export function rememberRoute(region: Region, routeId: string) { try { localStorage.setItem(KEY(region), routeId) } catch {} }

/** The route to open for a region: the one last viewed there if it is still built, else the default route if it is in
 * the region, else the region's first route; null when the region has none. */
export function routeForRegion(routes: CatalogueRoute[], region: Region): string | null {
  const mine = routes.filter(r => regionOf(r) === region)
  let saved: string | null = null
  try { saved = localStorage.getItem(KEY(region)) } catch {}
  for (const id of [saved, CONFIG.defaultRoute]) if (id && mine.some(r => r.id === id)) return id
  return mine[0]?.id ?? null
}

export function routeUrl(routeId: string): string {
  const u = new URL(location.href)
  u.searchParams.set('route', routeId); u.searchParams.delete('region')
  return u.toString()
}

/** Draw the switch: the current region pressed, a region with no built route disabled. */
export function renderRegionSwitch(el: HTMLElement, routes: CatalogueRoute[], current: Region, onPick: (r: Region) => void) {
  el.innerHTML = ''
  for (const r of REGIONS) {
    const b = document.createElement('button')
    const has = routes.some(x => regionOf(x) === r.id)
    b.type = 'button'; b.textContent = r.label; b.dataset.region = r.id
    b.classList.toggle('on', r.id === current)
    b.setAttribute('aria-pressed', String(r.id === current))
    b.disabled = !has
    b.title = has ? (r.id === current ? `Showing ${r.name} routes` : `Switch to ${r.name} routes`) : `No ${r.name} route has been built yet`
    if (r.id !== current && has) b.addEventListener('click', () => onPick(r.id))
    el.append(b)
  }
  el.hidden = false
}
