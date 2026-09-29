import type { Store } from '../state'
import type { CameraMode, Metric } from '../types'
import { CLASS_VARS, CONF_VARS, cssVar } from '../sim/classify'
import { esc } from '../html'
import { arrowNav } from './a11y'
import { showTab } from './inspector'

const $ = (id: string) => document.getElementById(id)!

/** Legend copy per metric: either graded classes or a sequential ramp with end labels. */
const LEGENDS: Record<Metric, { kind: 'classes' | 'ramp'; rows?: string[]; ends?: [string, string]; note?: string }> = {
  quality: { kind: 'classes', rows: ['Strong · score ≥ 0.6', 'Usable · 0.4 – 0.6', 'Poor · 0.2 – 0.4', 'Outage · below 0.2'] },
  capacity: { kind: 'classes', rows: ['50 Mbps and above', '20 – 50 Mbps', '5 – 20 Mbps', 'Under 5 Mbps'] },
  latency: { kind: 'classes', rows: ['45 ms or less', '45 – 80 ms', '80 – 150 ms', 'Over 150 ms'] },
  loss: { kind: 'classes', rows: ['0.5 % or less', '0.5 – 2 %', '2 – 5 %', 'Over 5 %'] },
  avail: { kind: 'classes', rows: ['Available', '', '', 'Unavailable'] },
  wifi: { kind: 'classes', rows: ['Excellent / good', 'Usable', 'Poor', 'Outage'] },
  conf: { kind: 'ramp', ends: ['0.0 unknown', '1.0 measured'], note: 'How much of the estimate rests on measured rather than predicted or synthetic inputs.' },
}

export function initRail(store: Store) {
  const s = store.state

  // ---- icon rail: one panel open at a time, clicking the active icon collapses it ------------
  const app = $('app')
  $('iconRail').addEventListener('click', e => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>('.rail-btn')
    if (!b) return
    const next = app.dataset.panel === b.dataset.panel ? 'none' : b.dataset.panel!
    app.dataset.panel = next
    for (const btn of document.querySelectorAll<HTMLElement>('.rail-btn')) btn.setAttribute('aria-selected', String(btn.dataset.panel === next))
    for (const sec of document.querySelectorAll<HTMLElement>('.panel-section')) sec.classList.toggle('on', sec.dataset.panel === next)
    rovingTabs()
    window.dispatchEvent(new Event('tls-layout'))
  })

  // one tab stop for the whole rail (the selected icon, or the first when every panel is closed); arrows move within it
  const railBtns = [...document.querySelectorAll<HTMLElement>('.rail-btn')]
  const rovingTabs = () => { const sel = railBtns.find(b => b.getAttribute('aria-selected') === 'true') ?? railBtns[0]; railBtns.forEach(b => { b.tabIndex = b === sel ? 0 : -1 }) }
  arrowNav(railBtns, 'vertical', b => { b.tabIndex = 0; railBtns.forEach(x => { if (x !== b) x.tabIndex = -1 }) })
  rovingTabs()

  // ---- links ---------------------------------------------------------------------------------
  const list = $('linkList')
  list.innerHTML = ''
  let lane = 1
  // grouped by what the link is, so the subtitle no longer repeats "cellular" on every operator
  const rows = [{ group: 'Onboard', key: 'wan', label: 'Combined Wi-Fi', sub: 'after the link policy', tag: 'centre' }]
  for (const p of s.data.meta.providers) {
    if (p.type === 'cellular') rows.push({ group: 'Cellular', key: p.id, label: p.name, sub: '', tag: `lane ${lane++}` })
    else rows.push({ group: 'Satellite', key: p.id, label: p.name, sub: p.terminal ? `${p.terminal} terminal` : '', tag: 'sky' })
  }
  let group = ''
  for (const r of rows) {
    if (r.group !== group) { group = r.group; const h = document.createElement('div'); h.className = 'grp-h'; h.textContent = group; list.appendChild(h) }
    const el = document.createElement('label')
    el.className = 'link-row'
    el.innerHTML = `<input type="checkbox" ${s.linkVisible[r.key] ? 'checked' : ''}><span class="name">${esc(r.label)}${r.sub ? `<small>${esc(r.sub)}</small>` : ''}</span><span class="lane">${esc(r.tag)}</span>`
    el.querySelector('input')!.addEventListener('change', e => store.set({ linkVisible: { ...store.state.linkVisible, [r.key]: (e.target as HTMLInputElement).checked } }))
    list.appendChild(el)
  }

  // ---- metric + legend ------------------------------------------------------------------------
  const metricSel = $('metric') as HTMLSelectElement
  metricSel.value = s.metric
  metricSel.addEventListener('change', () => store.set({ metric: metricSel.value as Metric }))

  // ---- layers ---------------------------------------------------------------------------------
  const layerIds: Record<string, keyof typeof s.layers> = {
    ly_cells: 'cells', ly_sky: 'sky', ly_stations: 'stations', ly_terrain: 'terrain',
    ly_labels: 'labels', ly_buildings: 'buildings', ly_trees: 'trees', ly_track: 'track',
  }
  for (const [id, key] of Object.entries(layerIds)) {
    const el = $(id) as HTMLInputElement
    el.checked = s.layers[key]
    el.addEventListener('change', () => store.set({ layers: { ...store.state.layers, [key]: el.checked } }))
  }
  // Synthetic-elevation bundles can't show 3D terrain (the toggle had no effect); say so instead of offering it.
  if (s.data.meta.terrain_source === 'synthetic_terrain') {
    const t = $('ly_terrain') as HTMLInputElement
    t.disabled = true
    $('ly_terrain_note').textContent = 'needs real elevation'
    t.closest('label')!.title = 'This route was built with synthetic elevation. Rebuild it with tcs run (without --offline; the Copernicus DEM needs no key) to enable 3D terrain.'
  }

  // ---- camera ---------------------------------------------------------------------------------
  $('camSeg').addEventListener('click', e => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button')
    if (b) store.set({ camera: b.dataset.v as CameraMode })
  })

  // ---- provenance chip opens Sources -----------------------------------------------------------
  $('provBadge').addEventListener('click', () => {
    const a = $('app')
    if (a.dataset.inspector !== 'open') { a.dataset.inspector = 'open'; $('btnInspector').setAttribute('aria-pressed', 'true'); window.dispatchEvent(new Event('tls-layout')) }
    showTab('sources')
  })

  store.on((st, changed) => {
    if (changed.has('metric') || changed.has('theme')) { metricSel.value = st.metric; renderLegend(st.metric) }
    if (changed.has('camera')) for (const b of document.querySelectorAll<HTMLElement>('#camSeg button')) { b.classList.toggle('on', b.dataset.v === st.camera); b.setAttribute('aria-pressed', String(b.dataset.v === st.camera)) }
    if (changed.has('layers')) for (const [id, key] of Object.entries(layerIds)) ($(id) as HTMLInputElement).checked = st.layers[key]
  })
  renderLegend(s.metric)
}

function renderLegend(metric: Metric) {
  const el = $('legend')
  const spec = LEGENDS[metric]
  if (spec.kind === 'ramp') {
    el.innerHTML = `<div class="ramp">${CONF_VARS.map(v => `<i style="background:${cssVar(v)}"></i>`).join('')}</div>
      <div class="keys"><span>${spec.ends![0]}</span><span>${spec.ends![1]}</span></div>
      <p class="hint">${spec.note ?? ''}</p>`
    return
  }
  // "Name · threshold" rows show the class name and its threshold separately; plain rows are thresholds on their own
  el.innerHTML = `<div class="rows">${spec.rows!.map((label, k) => {
    if (!label) return ''
    const [name, rule] = label.split(' · ')
    return `<span><i style="background:${cssVar(CLASS_VARS[k])}"></i>${rule ? `<b>${name}</b><em>${rule}</em>` : `<b>${name}</b>`}</span>`
  }).join('')}</div>`
}
