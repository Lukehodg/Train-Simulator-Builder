import './styles.css'
import type { PickingInfo } from '@deck.gl/core'
import { CONFIG } from './config'
import { indexAtTime, loadRoute } from './data'
import { createMap } from './map/map'
import { buildRuns, cellLayers, ribbonLayers, stationLayers, trainLayers, type Run } from './map/layers'
import { setBuildings, trackLayers, treeLayer } from './map/environment'
import { parityReport, simulate } from './sim/model'
import { Store, fmtClock, type State } from './state'
import { initInspector } from './ui/inspector'
import { initRail } from './ui/rail'
import { trapTab } from './ui/a11y'
import { setSimState } from './ui/status'
import { initTimeline } from './ui/timeline'
import { applyDesign, derive, designFromMeta, MAX_PROJECT_BYTES, parseProject, renderDesign, type TrainDesign } from './train'
import type { CameraMode, Meta, Policy, RouteData, SimResult, Vehicle } from './types'

const $ = (id: string) => document.getElementById(id)!

async function main() {
  const routeId = new URLSearchParams(location.search).get('route') ?? CONFIG.defaultRoute
  let theme: 'light' | 'dark' = 'dark'   // pit-wall default; an explicit choice (theme button) is remembered
  try { const saved = localStorage.getItem('tls-theme'); if (saved === 'light' || saved === 'dark') theme = saved } catch {}
  document.documentElement.dataset.theme = theme
  setSimState('loading')
  let data: RouteData
  try { data = await loadRoute(routeId) } catch (e) {
    setSimState('error')
    for (const id of ['play', 'btnExport']) ($(id) as HTMLButtonElement).disabled = true
    const pick = $('routePick') as HTMLSelectElement
    pick.innerHTML = ''; pick.append(new Option(routeId)); pick.disabled = true
    const other = routeId !== CONFIG.defaultRoute
    showNotice(`Could not load the route "${routeId}": its data bundle is missing or unreadable. Build it with tcs run --route ${routeId}${other ? ', or open the default route.' : '.'}`, true,
      other ? { label: 'Open the default route', href: `?route=${encodeURIComponent(CONFIG.defaultRoute)}` } : undefined, String((e as Error).message))
    return
  }
  const meta = data.meta
  const scenario = { policy: (meta.sim.wan.policy as Policy) ?? 'PACKET_BONDING', vehicle: (meta.sim.vehicle.profile as Vehicle) ?? 'EXTERNAL_ROOFTOP_ANTENNA', weather: 'nominal' }
  const sim = simulate(data, scenario)
  const realTerrain = meta.terrain_source !== 'synthetic_terrain'
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
  const linkVisible: Record<string, boolean> = { wan: true }
  for (const p of meta.providers) linkVisible[p.id] = true
  const store = new Store({
    data, sim, scenario, metric: 'quality', linkVisible, layers: { cells: false, sky: true, stations: true, terrain: realTerrain, labels: true, buildings: true, trees: true, track: true },
    camera: 'chase', t: meta.duration_s * CONFIG.playback.startFraction, playing: !reduced, speed: CONFIG.playback.defaultSpeed, selected: null, hover: null, theme,
  } as State)

  // ---- header ----------------------------------------------------------------------------------
  $('routeTitle').textContent = `${(meta.length_m / 1000).toFixed(0)} km · ${Math.round(meta.duration_s / 60)} min · ${meta.route.sample_spacing_m} m samples`
  fetch(`${CONFIG.dataRoot}/index.json`).then(r => (r.ok ? r.json() : null)).then((idx: any) => {
    const sel = $('routePick') as HTMLSelectElement
    const routes: any[] = idx?.routes?.length ? idx.routes : [{ id: routeId, name: meta.route.name, origin: meta.stations[0]?.name, destination: meta.stations[meta.stations.length - 1]?.name, length_km: meta.length_m / 1000 }]
    sel.innerHTML = routes.map(r => `<option value="${r.id}">${r.name ?? r.id} · ${r.origin} → ${r.destination} (${Math.round(r.length_km)} km)</option>`).join('')
    sel.value = routeId
    sel.addEventListener('change', () => { const u = new URL(location.href); u.searchParams.set('route', sel.value); location.href = u.toString() })
  }).catch(() => {})
  $('btnExport').addEventListener('click', () => exportScenario(store))
  const liveFlags = [meta.geometry_source === 'osm' || meta.geometry_source === 'file', realTerrain, meta.coverage_sources.some(s => s.startsWith('ofcom')), meta.cell_source === 'opencellid']
  const liveCount = liveFlags.filter(Boolean).length
  $('provDots').innerHTML = liveFlags.map(ok => `<i class="${ok ? 'live' : ''}"></i>`).join('')
  $('provText').textContent = liveCount === 4 ? 'Live data' : liveCount === 0 ? 'Synthetic' : `${liveCount}/4 live`
  $('provBadge').title = `Route ${meta.geometry_source} · terrain ${meta.terrain_source} · coverage ${meta.coverage_sources.join(', ') || 'synthetic'} · cells ${meta.cell_source} — click for detail`
  const parity = parityReport(data, sim)
  console.info(`[tls] parity vs Python default scenario: max bonded diff ${parity.maxBondedDiff.toFixed(2)} Mbps, class agreement ${(parity.classAgreement * 100).toFixed(1)} %`)
  if (!realTerrain) showNotice('3D terrain is off: this bundle was built with synthetic elevation. Run `tcs run` online (Copernicus DEM is keyless) to enable it.')
  // ---- scenario: presets, train design, manual overrides -------------------------------------
  // Baseline = pristine simulation.yaml (the bundle may have been produced with a preset or a train design applied).
  const baselineMeta: Meta = JSON.parse(JSON.stringify({ ...meta, sim: meta.sim_defaults ?? meta.sim }))
  for (const p of baselineMeta.providers) if (p.type === 'satcom' && p.terminal_default && p.capacity_priors) { p.terminal = p.terminal_default; p.capacity_prior_mbps = p.capacity_priors[p.terminal_default] }
  for (const p of baselineMeta.providers) if (p.type === 'satcom') p.enabled = p.enabled_default ?? true
  const baselineScenario = { policy: (baselineMeta.sim.wan.policy as Policy), vehicle: (baselineMeta.sim.vehicle.profile as Vehicle), weather: "nominal" as string }
  const baselineSim = simulate({ ...data, meta: baselineMeta }, baselineScenario)
  let design: TrainDesign | null = designFromMeta(meta)
  let designLabel = design ? `from pipeline run${design.source_file ? ' · ' + design.source_file.split(/[\\/]/).pop() : ''}` : ''
  const presetSel = $('preset') as HTMLSelectElement
  const syncSelects = () => { for (const k of ['policy', 'vehicle', 'weather'] as const) ($(k) as HTMLSelectElement).value = (store.state.scenario as any)[k] }
  const rerun = (m: Meta, sc: typeof scenario) => { const d = { ...store.state.data, meta: m }; store.set({ data: d, scenario: sc, sim: simulate(d, sc) }); syncSelects() }
  const applyPreset = (name: string) => {
    const pr = baselineMeta.sim.presets?.[name]
    if (name === 'design') {
      if (!design) { showNotice('No train design loaded yet: use the Train tab to load a .train.json saved by the Train Builder.'); presetSel.value = store.state.data.meta.sim.active_preset ?? 'baseline'; return }
      const m = applyDesign(store.state.data.meta, design, baselineMeta); m.sim.active_preset = 'design'
      rerun(m, { ...store.state.scenario, policy: design.policy, vehicle: design.vehicle_profile })
    } else if (name === 'edge_rail_fleet_connect' && pr) {
      const m: Meta = JSON.parse(JSON.stringify(baselineMeta)); m.sim.active_preset = name
      if (pr.bonding_efficiency) m.sim.wan.bonding_efficiency = pr.bonding_efficiency
      m.sim.satcom_enabled = pr.satcom_enabled !== false
      rerun(m, { ...store.state.scenario, policy: pr.policy ?? 'PACKET_BONDING', vehicle: pr.vehicle_profile ?? 'EDGE_RAIL_ACTIVE_ANTENNA' })
    } else {
      const m: Meta = JSON.parse(JSON.stringify(baselineMeta)); m.sim.active_preset = 'baseline'
      rerun(m, { ...store.state.scenario, policy: baselineScenario.policy, vehicle: baselineScenario.vehicle })
    }
    renderTrainTab()
  }
  presetSel.addEventListener('change', () => { applyPreset(presetSel.value); renderScenario() })
  ;(['policy', 'vehicle', 'weather'] as const).forEach(k => {
    const el = $(k) as HTMLSelectElement; el.value = (scenario as any)[k]
    el.addEventListener('change', () => { const sc = { ...store.state.scenario, [k]: el.value }; store.set({ scenario: sc, sim: simulate(store.state.data, sc) }); renderTrainTab(); renderScenario() })
  })
  // The four selects live in a popover; the pill states the current scenario in one line.
  const pill = $('scenarioPill'), pop = $('scenarioPopover') as HTMLElement
  const VEHICLE_SHORT: Record<string, string> = { EDGE_RAIL_ACTIVE_ANTENNA: 'EDGE Rail antenna', EXTERNAL_ROOFTOP_ANTENNA: 'rooftop antenna', PASSENGER_HANDSET_INSIDE_CARRIAGE: 'handset' }
  const POLICY_SHORT: Record<string, string> = { PACKET_BONDING: 'bonding', WEIGHTED_LOAD_BALANCING: 'load balancing', FAILOVER: 'failover', CELLULAR_PRIMARY_STARLINK_BACKUP: 'cellular first', STARLINK_PRIMARY_CELLULAR_BACKUP: 'satcom first' }
  const renderScenario = () => {
    const sc = store.state.scenario
    const presetLabel = presetSel.options[presetSel.selectedIndex]?.text.split(' · ')[0] ?? 'Baseline'
    const bits = [presetLabel, VEHICLE_SHORT[sc.vehicle] ?? sc.vehicle, POLICY_SHORT[sc.policy] ?? sc.policy]
    if (sc.weather !== 'nominal') bits.push(sc.weather)
    $('scenarioSummary').textContent = bits.join(' · ')
  }
  const setPopover = (open: boolean, returnFocus = false) => {
    if (pop.hidden === !open) return
    pop.hidden = !open
    pill.setAttribute('aria-expanded', String(open))
    if (open) { const r = pill.getBoundingClientRect(); pop.style.right = `${Math.max(8, innerWidth - r.right)}px`; ($('preset') as HTMLElement).focus() }
    else if (returnFocus) pill.focus()
  }
  pill.addEventListener('click', e => { e.stopPropagation(); setPopover(pop.hidden) })
  pop.addEventListener('click', e => e.stopPropagation())
  document.addEventListener('click', () => setPopover(false))
  const renderTrainTab = () => {
    const cur = store.state.data.meta.sim
    const usingDesign = cur.active_preset === 'design' && design
    $('trainBody').innerHTML = renderDesign(design, usingDesign ? `${designLabel} · applied to the simulation` : design ? `${designLabel} · not applied (choose the "Train design" preset)` : '')
    const pr = baselineMeta.sim.presets?.edge_rail_fleet_connect
    const cl = $('claims') as HTMLElement
    const edgeActive = store.state.scenario.vehicle === 'EDGE_RAIL_ACTIVE_ANTENNA'
    cl.hidden = !edgeActive || !pr?.claims?.length
    if (!cl.hidden) {
      const vp = cur.vehicle.profiles.EDGE_RAIL_ACTIVE_ANTENNA
      const agg = store.state.scenario.policy === 'PACKET_BONDING' ? `Fleet Connect aggregates every cellular network and the satcom link at once (efficiency ${cur.wan.bonding_efficiency})` : 'aggregation off: choose Packet bonding to model Fleet Connect'
      cl.innerHTML = `<b>EDGE Rail 5G active antenna · model assumptions</b>Link budget +${vp.db_offset} dB (no coax/splitter losses, 4x4 MIMO diversity) = quality +${vp.score_offset}; throughput x${vp.capacity_factor}; ${agg}.<br>Manufacturer claims, not modelled: ${pr.claims.join(' · ')}.`
    }
  }
  ;($('designFile') as HTMLInputElement).addEventListener('change', async e => {
    const input = e.target as HTMLInputElement
    const f = input.files?.[0]; if (!f) return
    try {
      if (f.size > MAX_PROJECT_BYTES) throw new Error('Train Studio project exceeds the 5 MiB limit')
      design = derive(parseProject(await f.text()), baselineMeta.sim.train)
      design.source_file = f.name; designLabel = f.name
      presetSel.value = 'design'; applyPreset('design'); renderScenario()
      showNotice(`Loaded "${design.title}": ${design.carriages.length} carriages · ${design.cellular_units} EDGE Rail · ${design.satcom_units} satcom · ${design.aps_connected}/${design.aps_total} APs connected`)
    } catch (err) { showNotice(String((err as Error).message), true) }
    input.value = ''
  })
  $('designExample').addEventListener('click', async () => {
    try {
      const res = await fetch('examples/azuma-5car.train.json'); if (!res.ok) throw new Error('example design not found')
      design = derive(parseProject(await res.text()), baselineMeta.sim.train); design.source_file = 'azuma-5car.train.json'; designLabel = 'example · azuma-5car.train.json'
      presetSel.value = 'design'; applyPreset('design'); renderScenario()
    } catch (err) { showNotice(String((err as Error).message), true) }
  })
  $('designReset').addEventListener('click', () => { design = designFromMeta(baselineMeta); designLabel = design ? 'from pipeline run' : ''; presetSel.value = 'baseline'; applyPreset('baseline'); renderScenario() })
  presetSel.value = meta.sim.active_preset && meta.sim.presets?.[meta.sim.active_preset] ? meta.sim.active_preset : (design ? 'design' : 'baseline')
  renderTrainTab()
  renderScenario()
  $('btnInspector').addEventListener('click', () => {
    const a = $('app'); const open = a.dataset.inspector !== 'open'
    a.dataset.inspector = open ? 'open' : 'closed'
    $('btnInspector').setAttribute('aria-pressed', String(open))
    window.dispatchEvent(new Event('tls-layout'))
  })
  // Any layout change (rail panel, inspector) needs the map and the timeline to re-measure.
  window.addEventListener('tls-layout', () => setTimeout(() => { mapCtx.map.resize(); timeline.redraw() }, 30))
  $('btnTheme').addEventListener('click', () => { const t = store.state.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = t; try { localStorage.setItem('tls-theme', t) } catch {} store.set({ theme: t }) })
  const help = $('help') as HTMLElement
  let helpReturn: HTMLElement | null = null
  const openHelp = () => { renderHelp(); helpReturn = document.activeElement as HTMLElement | null; help.hidden = false; ($('helpClose') as HTMLElement).focus() }
  const closeHelp = () => { if (help.hidden) return; help.hidden = true; helpReturn?.focus() }
  help.addEventListener('keydown', trapTab(help.querySelector('.modal-card') as HTMLElement))
  $('btnHelp').addEventListener('click', openHelp)
  $('helpClose').addEventListener('click', closeHelp)
  help.addEventListener('click', e => { if (e.target === help) closeHelp() })
  if (innerWidth < 1100) { $('app').dataset.panel = 'none'; for (const b of document.querySelectorAll('.rail-btn')) b.setAttribute('aria-selected', 'false') }
  if (innerWidth < 900) { $('app').dataset.inspector = 'closed'; $('btnInspector').setAttribute('aria-pressed', 'false') }

  // ---- map ------------------------------------------------------------------------------------
  const tooltip = (info: PickingInfo) => {
    const o: any = info.object; if (!o) return null
    if (o.key && o.provider) return { html: `<b>${o.provider.toUpperCase()}</b> cell ${o.key}<br>${o.radio} · ${o.samples} obs · ${o.source}`, style: tipStyle() }
    if (o.crs) return { html: `<b>${o.name}</b> (${o.crs})<br>${(o.distance_m / 1000).toFixed(1)} km${o.scheduled ? ' · ' + o.scheduled : ''}${o.stop ? ' · stop' : ' · pass'}`, style: tipStyle() }
    if (o.lane) return { html: `<b>${o.lane === 'wan' ? 'Combined Wi-Fi' : o.lane}</b> · click to inspect sample`, style: tipStyle() }
    return null
  }
  const mapCtx = createMap($('map'), data, theme, tooltip)
  ;(window as any).__tls = { map: mapCtx.map, store, overlay: mapCtx.overlay }
  await mapCtx.ready
  mapCtx.setTerrain(store.state.layers.terrain)
  setBuildings(mapCtx.map, store.state.layers.buildings, store.state.theme)
  mapCtx.onUserInteract(() => { if (store.state.camera === 'chase' || store.state.camera === 'oblique') store.set({ camera: 'free' }) })

  // ---- panels ---------------------------------------------------------------------------------
  initRail(store)
  const inspector = initInspector(store)
  const timeline = initTimeline(store, i => seek(data.t[i]))
  const seek = (t: number) => store.set({ t: Math.max(0, Math.min(meta.duration_s, t)) })

  // transport
  const setPlayIcon = () => {
    $('playIcon').setAttribute('d', store.state.playing ? 'M2 1h3v10H2zM7 1h3v10H7z' : 'M2 1l9 5-9 5z')
    $('play').setAttribute('aria-label', store.state.playing ? 'Pause (Space)' : 'Play (Space)')
  }
  const renderState = () => { const st = store.state; setSimState(st.playing ? 'running' : st.t >= meta.duration_s ? 'finished' : 'paused', st.playing ? `${st.speed}×` : '') }
  $('play').addEventListener('click', () => { store.set({ playing: !store.state.playing }); setPlayIcon() })
  setPlayIcon()
  const speedBtns = [...document.querySelectorAll<HTMLButtonElement>('#speedSeg button')]
  const setSpeed = (v: number) => { store.set({ speed: v }); speedBtns.forEach(x => { const on = Number(x.dataset.v) === v; x.classList.toggle('on', on); x.setAttribute('aria-pressed', String(on)) }) }
  setSpeed(store.state.speed)
  $('speedSeg').addEventListener('click', e => { const b = (e.target as HTMLElement).closest('button'); if (b) setSpeed(Number(b.dataset.v)) })
  const scrub = $('scrub') as HTMLInputElement
  let scrubbing = false
  scrub.addEventListener('input', () => { scrubbing = true; seek(Number(scrub.value) / 100000 * meta.duration_s) })
  scrub.addEventListener('change', () => { scrubbing = false })
  const jump = $('jump') as HTMLSelectElement
  jump.innerHTML = '<option value="">station…</option>' + meta.stations.map((s, k) => `<option value="${k}">${s.name}</option>`).join('')
  jump.addEventListener('change', () => { const s = meta.stations[Number(jump.value)]; if (s) seek(Math.max(0, data.t[s.sample_id] - 30)); jump.value = '' })
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeHelp(); setPopover(false, true); return }
    if (!help.hidden) return                                  // the dialog is modal: no app shortcuts behind it
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
    if (e.ctrlKey || e.metaKey || e.altKey) return
    const step = e.shiftKey ? 600 : 60, rates = speedBtns.map(b => Number(b.dataset.v)), k = rates.indexOf(store.state.speed)
    if (e.key === ' ') { e.preventDefault(); store.set({ playing: !store.state.playing }); setPlayIcon() }
    else if (e.key === 'ArrowRight') seek(store.state.t + step)
    else if (e.key === 'ArrowLeft') seek(store.state.t - step)
    else if (e.key === ']') setSpeed(rates[Math.min(rates.length - 1, k + 1)])
    else if (e.key === '[') setSpeed(rates[Math.max(0, k - 1)])
    else if (e.key === 'Home') { e.preventDefault(); seek(0) }
    else if (e.key === 'End') { e.preventDefault(); seek(meta.duration_s) }
    else if (e.key === 'i' || e.key === 'I') $('btnInspector').click()
    else if (e.key === 'c' || e.key === 'C') { const views: CameraMode[] = ['chase', 'oblique', 'free', 'route']; store.set({ camera: views[(views.indexOf(store.state.camera) + 1) % views.length] }) }
    else if (e.key === '?') openHelp()
  })
  renderState()
  renderKpis(store, baselineSim)

  // ---- reactive layer rebuilds ---------------------------------------------------------------
  let runs: Run[] = buildRuns(store.state, store.state.layers.terrain)
  let staticLayers = [...ribbonLayers(store.state, runs, i => { store.set({ selected: i, playing: false }); setPlayIcon() }), ...stationLayers(store.state, store.state.layers.terrain)]
  store.on((st, changed) => {
    if (changed.has('layers')) { mapCtx.setTerrain(st.layers.terrain && realTerrain); mapCtx.setLabels(st.layers.labels); setBuildings(mapCtx.map, st.layers.buildings, st.theme) }
    if (changed.has('theme')) { mapCtx.setTheme(st.theme); mapCtx.map.once('style.load', () => { mapCtx.setLabels(st.layers.labels); setBuildings(mapCtx.map, st.layers.buildings, st.theme) }) }
    if (changed.has('sim') || changed.has('metric') || changed.has('theme') || changed.has('linkVisible') || changed.has('layers')) {
      runs = buildRuns(st, st.layers.terrain && realTerrain)
      staticLayers = [...ribbonLayers(st, runs, i => { store.set({ selected: i, playing: false }); setPlayIcon() }), ...stationLayers(st, st.layers.terrain && realTerrain)]
    }
    if (changed.has('sim')) renderKpis(store, baselineSim)
    if (changed.has('camera') && st.camera === 'route') mapCtx.fitRoute()
    if (changed.has('selected') && st.selected != null) timeline.redraw()
    if (changed.has('playing') || changed.has('t') || changed.has('speed')) renderState()
  })

  // ---- frame loop -----------------------------------------------------------------------------
  let last = performance.now(), uiAt = 0
  function frame(now: number) {
    const st = store.state
    const dt = Math.min(0.1, (now - last) / 1000); last = now
    if (st.playing) { let t = st.t + dt * st.speed; if (t >= meta.duration_s) { t = meta.duration_s; store.set({ playing: false }); setPlayIcon() } st.t = t }
    const i = indexAtTime(data.t, st.t)
    const j = Math.min(data.n - 1, i + 1)
    const frac = j > i ? Math.max(0, Math.min(1, (st.t - data.t[i]) / (data.t[j] - data.t[i]))) : 0
    const useTerrain = st.layers.terrain && realTerrain
    const zoom = mapCtx.map.getZoom()
    mapCtx.setLayers([...staticLayers, ...(st.layers.track ? trackLayers(st, i, useTerrain) : []), ...treeLayer(st, mapCtx.map, i, useTerrain, now), ...cellLayers(st, i, useTerrain), ...trainLayers(st, i, frac, useTerrain, zoom)])
    const lon = data.lon[i] + (data.lon[j] - data.lon[i]) * frac, lat = data.lat[i] + (data.lat[j] - data.lat[i]) * frac
    mapCtx.drive(st.camera, lon, lat, data.bearing[i], dt)
    if (now - uiAt > 100) {
      uiAt = now
      inspector.update(i)
      renderState()
      $('clock').textContent = fmtClock(meta.departure, st.t)
      if (!scrubbing) scrub.value = String(Math.round(st.t / meta.duration_s * 100000))
      timeline.draw(i)
    }
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

/** Browser-side export of the on-screen scenario: per-sample CSV + a JSON summary. The tender-grade pack (Word + Excel) is `tcs report`. */
function exportScenario(store: Store) {
  const { data, sim, scenario } = store.state
  const m = data.meta
  const ids = m.providers.map(p => p.id)
  const head = ['sample_id', 'distance_m', 'sim_time', 'latitude', 'longitude', 'speed_kph', 'in_tunnel', 'sky_visibility', ...ids.flatMap(id => [`${id}_quality`, `${id}_capacity_mbps`, `${id}_latency_ms`, `${id}_available`]), 'active_links', 'bonded_capacity_mbps', 'effective_latency_ms', 'packet_loss_pct', 'per_user_mbps', 'wifi_score', 'service_class', 'confidence']
  const rows = [head.join(',')]
  const cls = ['EXCELLENT', 'GOOD', 'USABLE', 'POOR', 'OUTAGE']
  for (let i = 0; i < data.n; i++) {
    const act = ids.filter((_, k) => sim.active[i] & (1 << k)).join('+')
    rows.push([i, data.distance[i].toFixed(0), fmtClock(m.departure, data.t[i]), data.lat[i].toFixed(6), data.lon[i].toFixed(6), data.speed[i].toFixed(0), data.inTunnel[i], data.sky[i].toFixed(2),
      ...ids.flatMap(id => { const L = sim.links[id]; return [L.q[i].toFixed(3), L.cap[i].toFixed(1), L.lat[i].toFixed(0), L.avail[i]] }),
      act, sim.bonded[i].toFixed(1), Number.isNaN(sim.lat[i]) ? '' : sim.lat[i].toFixed(0), sim.loss[i].toFixed(2), sim.perUser[i].toFixed(2), sim.wifi[i].toFixed(0), cls[sim.wifiClass[i]], sim.conf[i].toFixed(2)].join(','))
  }
  const stem = `tls-${m.route.id}-${m.sim.active_preset ?? 'baseline'}-${scenario.vehicle.toLowerCase()}-${scenario.policy.toLowerCase()}`
  const save = (name: string, text: string, type: string) => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000) }
  save(`${stem}.csv`, rows.join(String.fromCharCode(10)), 'text/csv')
  const s = kpiStats(store, sim)
  save(`${stem}.summary.json`, JSON.stringify({ route: m.route, generated: new Date().toISOString(), model_version: m.model_version, scenario, preset: m.sim.active_preset ?? 'baseline', train_design: m.sim.train?.design ?? null,
    sources: { geometry: m.geometry_source, terrain: m.terrain_source, coverage: m.coverage_sources, cells: m.cell_source }, kpis: s, note: 'Model predictions, not measurements. See docs/model.md and the Sources tab for provenance and confidence.' }, null, 1), 'application/json')
}

function tipStyle() { return { background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: '6px', padding: '6px 8px', fontFamily: 'var(--body)', fontSize: '12px', boxShadow: 'var(--shadow)' } }

function showNotice(msg: string, error = false, action?: { label: string; href: string }, detail?: string) {
  const n = $('notice'), m = $('noticeMsg')
  m.textContent = msg
  if (action) { const a = document.createElement('a'); a.className = 'btn'; a.href = action.href; a.textContent = action.label; m.append(document.createElement('br'), a) }
  if (detail) { const d = document.createElement('small'); d.className = 'detail'; d.textContent = `Details: ${detail}`; m.append(d) }
  n.classList.toggle('error', error); n.hidden = false
}
$('noticeClose').addEventListener('click', () => { $('notice').hidden = true })

function kpiStats(store: Store, sim: SimResult) {
  const { data } = store.state
  const share = [0, 0, 0, 0, 0]; let outageM = 0, bonded = 0
  for (let i = 0; i < data.n; i++) { share[sim.wifiClass[i]]++; if (sim.wifiClass[i] === 4 && i + 1 < data.n) outageM += data.distance[i + 1] - data.distance[i]; bonded += sim.bonded[i] }
  return { stream: (share[0] + share[1]) / data.n * 100, usable: (share[0] + share[1] + share[2]) / data.n * 100, outageKm: outageM / 1000, bondedMean: bonded / data.n, conf: sim.conf.reduce((a, b) => a + b, 0) / data.n }
}

function renderKpis(store: Store, baseline?: SimResult) {
  const { data, sim } = store.state
  const s = kpiStats(store, sim)
  const b = baseline && baseline !== sim ? kpiStats(store, baseline) : null
  let ho = 0
  for (const p of data.meta.providers) if (p.type === 'cellular') { const hp = data.base[p.id].hp; for (let i = 1; i < data.n; i++) if (hp[i] > 0 && hp[i - 1] === 0) ho++ }
  const delta = (v: number, ref: number | undefined, unit: string, lowerIsBetter = false) => {
    if (ref == null || Math.abs(v - ref) < 0.05) return ''
    const d = v - ref; const good = lowerIsBetter ? d < 0 : d > 0
    return `<small class="${good ? '' : 'neg'}" title="vs baseline">${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(Math.abs(d) < 10 ? 1 : 0)}</small>`
  }
  const tile = (label: string, value: string, unit: string, delta: string) => `<div class="kpi"><span>${label}</span><b>${value}<u>${unit}</u>${delta}</b></div>`
  const kp = [
    tile('Streaming capable', s.stream.toFixed(0), '%', delta(s.stream, b?.stream, '')),
    tile('Usable or better', s.usable.toFixed(0), '%', delta(s.usable, b?.usable, '')),
    tile('Predicted outage', s.outageKm.toFixed(1), 'km', delta(s.outageKm, b?.outageKm, '', true)),
    tile('Mean WAN', s.bondedMean.toFixed(0), 'Mbps', delta(s.bondedMean, b?.bondedMean, '')),
    tile('Handovers', String(ho), '', ''),
    tile('Confidence', s.conf.toFixed(2), '', ''),
  ]
  $('kpis').innerHTML = kp.join('')
}

function renderHelp() {
  $('helpBody').innerHTML = `
    <p>A time-aware simulation of onboard connectivity for one train service: every 50 m along the real railway the model estimates each cellular operator and the satcom link, runs the onboard link manager, and predicts the passenger Wi-Fi experience.</p>
    <ul>
      <li><b>Ribbons</b>: centre = combined Wi-Fi after the selected policy; lanes to the right = each cellular operator; the floating lane = satcom. Colour follows the selected metric; class names are always shown in the inspector and tooltips so colour is never the only signal.</li>
      <li><b>Sky window</b>: the disc above the train scales with the satcom sky-visibility score from the DEM horizon; in the outage colour there is no session (tunnel, canopy, deep cutting).</li>
      <li><b>Train and surroundings</b>: the consist follows the loaded Train Studio design (roof units included); rails, catenary, 3D buildings and woodland trees come from the basemap's own vector tiles, so they are real footprints but schematic heights.</li>
      <li><b>Scenario controls</b> re-run the link manager and Wi-Fi model in the browser; the Python pipeline produced the per-link base estimates.</li>
      <li><b>Confidence</b>: switch the colouring to Confidence to see how much of the route rests on measured, predicted or synthetic inputs.</li>
      <li><b>Keys</b>: <kbd>Space</kbd> play / pause · <kbd>←</kbd> <kbd>→</kbd> ±1 min (<kbd>Shift</kbd> ±10 min) · <kbd>[</kbd> <kbd>]</kbd> playback rate · <kbd>Home</kbd> <kbd>End</kbd> start / end · <kbd>I</kbd> inspector · <kbd>C</kbd> camera view · <kbd>?</kbd> this help · <kbd>Esc</kbd> close. Drag the map to take the camera.</li>
    </ul>
    <p>Predictions, not measurements. Ofcom coverage is operator-predicted; OpenCellID is community data; Starlink has no public route-level telemetry. See the Sources tab for what is live in this bundle.</p>`
}

main()
