import { fmtHM, type Store } from '../state'
import { CLASS_NAMES, CLASS_VARS, WIFI_CLASSES, classOf, cssVar, qClass } from '../sim/classify'
import { arrowNav } from './a11y'
import { LIVE_COVERAGE_MIN, liveCoverageShare } from '../data'
import { esc } from '../html'

const $ = (id: string) => document.getElementById(id)!

export function envText(store: Store, i: number): string {
  const d = store.state.data
  if (d.inTunnel[i]) return `${d.tunnelName[i] && d.tunnelName[i] !== 'tunnel' ? d.tunnelName[i] : 'Tunnel'}`
  if (d.canopy[i] >= 0.9) return 'Station canopy'
  if (d.cutting[i] > 8) return `Cutting · ${d.cutting[i].toFixed(0)} m`
  if (d.sky[i] < 0.6) return `Terrain shadow · sky ${(d.sky[i] * 100).toFixed(0)} %`
  if (d.urban[i] > 0.5) return 'Urban'
  return 'Open country'
}

export function initInspector(store: Store) {
  const tabs = document.querySelectorAll<HTMLButtonElement>('.tab')
  tabs.forEach(t => t.addEventListener('click', () => showTab(t.dataset.tab!)))
  arrowNav([...tabs], 'horizontal', b => showTab(b.dataset.tab!))
  const s = store.state
  const host = $('liveLinks'); host.innerHTML = ''
  // one row per link: name · class · Mbps · ms, with a full-width capacity bar underneath (units live in the header, not every cell)
  host.insertAdjacentHTML('beforeend', '<div class="lrow lhead" aria-hidden="true"><span>Link</span><span>Class</span><span>Mbps</span><span>ms</span></div>')
  const rows: Record<string, { row: HTMLElement; nm: HTMLElement; cls: HTMLElement; fill: HTMLElement; cap: HTMLElement; lat: HTMLElement }> = {}
  for (const p of s.data.meta.providers) {
    const row = document.createElement('div'); row.className = 'lrow'
    const nm = document.createElement('div'); nm.className = 'nm'; nm.innerHTML = `<i></i>${esc(p.name)}`
    const cls = document.createElement('span'); cls.className = 'pill'
    const cap = document.createElement('span'); cap.className = 'num'
    const lat = document.createElement('span'); lat.className = 'num'
    const bar = document.createElement('div'); bar.className = 'bar'; const fill = document.createElement('i'); bar.appendChild(fill)
    row.append(nm, cls, cap, lat, bar); host.appendChild(row); rows[p.id] = { row, nm, cls, fill, cap, lat }
  }
  renderSources(store)
  store.on((st, changed) => {
    if (changed.has('selected') && st.selected != null) { renderSample(store, st.selected); showTab('sample') }
    // keep an open sample card in step with a re-run scenario or a theme switch (it was left showing the old numbers / colours)
    else if ((changed.has('sim') || changed.has('theme')) && st.selected != null) renderSample(store, st.selected)
    if (changed.has('theme')) renderSources(store)
  })

  return {
    update(i: number) {
      const st = store.state, d = st.data, r = st.sim
      $('lvSpeed').textContent = String(Math.round(d.speed[i]))
      $('lvDist').textContent = `${(d.distance[i] / 1000).toFixed(1)} km`
      const nxt = d.nextStation[i]; const stn = nxt ? d.meta.stations.find(x => x.crs === nxt) : null
      $('lvNext').textContent = stn ? stn.name : 'terminus'
      $('lvEta').textContent = stn ? fmtHM(d.meta.departure, d.t[i] + d.ttn[i]) : '—'
      $('lvEnv').textContent = envText(store, i)
      d.meta.providers.forEach((p, k) => {
        const L = r.links[p.id], row = rows[p.id], on = (r.active[i] & (1 << k)) !== 0, c = classOf('quality', p.id, i, d, r)
        row.cls.textContent = CLASS_NAMES[c]; row.cls.style.borderLeftColor = cssVar(CLASS_VARS[c])
        row.fill.style.width = `${Math.min(100, L.cap[i] / 250 * 100)}%`; row.fill.style.background = cssVar(CLASS_VARS[c])
        // unavailable links show the reason across both number columns instead of two dashes
        row.cap.textContent = L.avail[i] ? String(Math.round(L.cap[i])) : L.reason[i].replace(/_/g, ' ').toLowerCase()
        row.lat.textContent = L.avail[i] ? String(Math.round(L.lat[i])) : ''
        row.cap.classList.toggle('reason', !L.avail[i]); row.lat.hidden = !L.avail[i]
        row.nm.classList.toggle('active', on)
        row.row.classList.toggle('off', !st.linkVisible[p.id])
      })
      const act = d.meta.providers.filter((_, k) => r.active[i] & (1 << k)).map(p => p.name)
      $('wanActive').textContent = act.length ? act.join(' + ') : 'none'
      $('wanCap').textContent = act.length ? `${Math.round(r.bonded[i])} Mbps` : '—'
      $('wanLat').textContent = act.length ? `${Math.round(r.lat[i])} ms · ${r.loss[i].toFixed(1)} %` : '—'
      $('wanUser').textContent = act.length ? `${r.perUser[i].toFixed(1)} Mbps × ${Math.round(r.activeUsers[i])} users` : '—'
      $('wifiScore').textContent = String(Math.round(r.wifi[i]))
      const wc = r.wifiClass[i]; const pill = $('wifiClass'); pill.textContent = WIFI_CLASSES[wc]; pill.style.borderLeftColor = cssVar(CLASS_VARS[wc <= 1 ? 0 : wc - 1])
      $('wanConf').textContent = r.conf[i].toFixed(2)
    },
  }
}

export function showTab(name: string) {
  document.querySelectorAll<HTMLButtonElement>('.tab').forEach(t => {
    const on = t.dataset.tab === name
    t.classList.toggle('on', on); t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1
  })
  for (const id of ['live', 'sample', 'train', 'sources']) (document.getElementById(`tab-${id}`) as HTMLElement).hidden = id !== name
}

function renderSample(store: Store, i: number) {
  const st = store.state, d = st.data, r = st.sim, sim = d.meta.sim
  const kv = (k: string, v: string) => `<div class="kv"><span>${k}</span><b>${v}</b></div>`
  let h = ''

  // context card
  h += `<section class="card"><h4>Sample ${i} · ${(d.distance[i] / 1000).toFixed(2)} km${d.stationNear[i] ? ' · ' + esc(d.stationNear[i]!) : ''}</h4>`
  h += kv('Position', `${d.lat[i].toFixed(5)}, ${d.lon[i].toFixed(5)}`)
    + kv('Railhead / terrain', `${d.elev[i].toFixed(0)} / ${d.terrain[i].toFixed(0)} m`)
    + kv('Environment', esc(envText(store, i)))
    + kv('Sky visibility', `${(d.sky[i] * 100).toFixed(0)} %`)
    + kv('Time · speed', `${fmtHM(d.meta.departure, d.t[i])} · ${Math.round(d.speed[i])} km/h`) + `</section>`

  // one card per link, with its reasoning inline
  const cfg = sim.cellular, vprof = sim.vehicle.profiles[st.scenario.vehicle]
  for (const p of d.meta.providers) {
    const L = r.links[p.id], b = d.base[p.id], c = classOf('quality', p.id, i, d, r)
    h += `<div class="lcard"><div class="lcard-head"><b>${esc(p.name)}</b><span class="pill" style="border-left-color:${cssVar(CLASS_VARS[c])}">${CLASS_NAMES[c]}</span></div>`
    if (p.type === 'cellular') {
      const cell = b.cell[i] ? d.cellIndex.get(b.cell[i]!) : null
      h += kv('Quality', L.q[i].toFixed(2))
        + kv('RSRP (modelled)', `${L.rsrp[i].toFixed(0)} dBm`)
        + kv('Capacity / latency', `${Math.round(L.cap[i])} Mbps / ${Math.round(L.lat[i])} ms`)
        + kv('Serving cell', cell ? `${(b.celld[i] / 1000).toFixed(1)} km · ${esc(cell.radio)}` : '—')
        + kv('Technology · confidence', `${esc(b.tech[i] ?? '—')} · ${b.conf[i].toFixed(2)}`)
    } else {
      h += kv('Sky visibility', L.q[i].toFixed(2))
        + kv('Available', L.avail[i] ? 'yes' : `no · ${L.reason[i].toLowerCase().replace(/_/g, ' ')}`)
        + kv('Capacity / latency', `${Math.round(L.cap[i])} Mbps / ${Math.round(L.lat[i])} ms`)
        + kv('Confidence', b.conf[i].toFixed(2))
    }
    // reasoning terms
    const rows: [string, string][] = []
    if (p.type === 'cellular') {
      rows.push(['coverage prior + terrain + cell distance', b.qb[i].toFixed(2)])
      const cut = Math.min(1, d.cutting[i] / cfg.terrain.cutting_full_depth_m) * cfg.terrain.cutting_penalty_max
      if (cut > 0.005) rows.push([`cutting ${d.cutting[i].toFixed(0)} m`, `−${cut.toFixed(2)}`])
      const dk = b.celld[i] / 1000, dp = Number.isNaN(dk) ? 0 : Math.max(0, dk - cfg.cell_distance.free_km) * cfg.cell_distance.penalty_per_km
      if (dp > 0.005) rows.push([`cell ${dk.toFixed(1)} km away`, `−${dp.toFixed(2)}`])
      if (vprof.score_offset) rows.push([vprof.score_offset < 0 ? 'carriage penetration loss' : `active antenna (+${vprof.db_offset} dB)`, `${vprof.score_offset > 0 ? '+' : ''}${vprof.score_offset.toFixed(2)}`])
      if (vprof.capacity_factor && vprof.capacity_factor !== 1) rows.push(['antenna / MIMO factor', `×${vprof.capacity_factor}`])
      if (L.reason[i] === 'NOT_FITTED') rows.push(['on this train', 'not fitted: no modem for this network, so the router cannot use it'])
      if (d.inTunnel[i]) rows.push(['tunnel', L.q[i] > 0.1 ? `in-tunnel coverage assumed → ${cfg.tunnels.das_score}` : `no coverage → ${cfg.tunnels.default_score}`])
      if (b.hp[i] > 0) rows.push(['handover', `+${cfg.handover.latency_spike_ms} ms · ×${cfg.handover.capacity_factor}`])
    } else {
      rows.push(['sky visibility (DEM horizon)', b.qb[i].toFixed(2)])
      if (d.canopy[i] > 0) rows.push(['station canopy', `−${(0.85 * d.canopy[i]).toFixed(2)}`])
      if (d.cutting[i] > 0.5) rows.push([`cutting horizon ${d.cutting[i].toFixed(0)} m`, `−${Math.min(0.55, d.cutting[i] / 22).toFixed(2)}`])
      if (d.urban[i] > 0.01) rows.push(['urban obstruction', `−${(0.18 * d.urban[i] * (1 - d.canopy[i])).toFixed(2)}`])
      if (st.scenario.weather !== 'nominal') rows.push([`weather · ${st.scenario.weather}`, `sky −${p.availability!.weather[st.scenario.weather].sky_penalty}`])
      rows.push(['reason code', L.reason[i]])
    }
    rows.push(['source', esc(b.src[i] ?? '').replace(/\|/g, '|<wbr>')])   // break long source chains at the separator, not mid-word
    h += `<div class="why"><div class="why-h">How it was reached</div>${rows.map(([k, v]) => `<div class="row"><span>${k}</span><code>${v}</code></div>`).join('')}</div></div>`
  }

  // combined WAN card
  const act = d.meta.providers.filter((_, k) => r.active[i] & (1 << k)).map(p => p.name).join(' + ') || 'none'
  h += `<section class="card"><h4>Onboard WAN · ${esc(st.scenario.policy.toLowerCase().replace(/_/g, ' '))}</h4>`
  h += kv('Active links', esc(act))
    + kv('Bonded', `${Math.round(r.bonded[i])} Mbps`)
    + kv('Latency · loss', `${Number.isNaN(r.lat[i]) ? '—' : Math.round(r.lat[i]) + ' ms'} · ${r.loss[i].toFixed(1)} %`)
    + kv('Wi-Fi score', `${Math.round(r.wifi[i])} · ${WIFI_CLASSES[r.wifiClass[i]]}`)
    + kv('Confidence', r.conf[i].toFixed(2))
    + kv('Model', esc(d.meta.model_version)) + `</section>`

  $('sampleHint').hidden = true
  $('sampleBody').innerHTML = h
}

function renderSources(store: Store) {
  const m = store.state.data.meta
  const live = (flag: boolean, liveLabel = 'live', synthLabel = 'synthetic') => `<em class="${flag ? 'live' : 'synth'}">${flag ? liveLabel : synthLabel}</em>`
  const geomLive = m.geometry_source === 'osm' || m.geometry_source === 'file'
  const straight = m.geometry_source === 'osm' ? m.geometry_straight_legs ?? [] : []   // legs with no rail path, drawn straight
  const terrLive = m.terrain_source !== 'synthetic_terrain'
  const covShare = liveCoverageShare(m), covLive = covShare >= LIVE_COVERAGE_MIN
  const cellLive = m.cell_source === 'opencellid'
  const items = [
    ['Route geometry', geomLive && !straight.length, m.geometry_source === 'osm' ? 'OpenStreetMap rail network, routed station-to-station (ODbL). Tunnel / cutting / embankment / bridge / maxspeed tags carried per 50 m sample.'
      + (straight.length ? ` No rail path was found for ${esc(straight.map(s => s.replace('-', '–')).join(', '))}: drawn as a straight line there, without tunnels, cuttings or line speeds.` : '') : m.geometry_source === 'file' ? 'Infrastructure-manager / curated centreline file.' : 'Spline through approximate station coordinates. Run <code>tcs run</code> without <code>--offline</code> to fetch the OSM centreline.', straight.length ? 'partly live' : undefined],
    ['Terrain & sky visibility', terrLive, terrLive ? `${esc(m.terrain_source)}: 30 m DEM, 16-ray horizon per sample, solid-angle sky fraction above the terminal's minimum elevation.` : 'Procedural terrain. 3D terrain rendering is disabled until real elevation is available.'],
    ['Cellular coverage prior', covLive, covLive ? `${esc(m.coverage_sources.join(', '))} — operator predictions on Ofcom's 50 m grid mapped to a model score; never presented as measured RSRP.`
      : covShare > 0 ? `Only ${Math.round(covShare * 100)} % of the route has Ofcom predictions (usually the Ofcom call quota ran out part-way); the rest is a neutral stand-in. Rebuild the route once the quota resets.`
      : 'Synthetic prior (noise field). Set <code>OFCOM_API_KEY</code>, or enable Connected Nations open data, to replace it.', covShare > 0 && !covLive ? 'partly live' : undefined],
    ['Cell sites', cellLive, cellLive ? 'OpenCellID corridor extract (CC BY-SA 4.0). Logical cells; top-5 candidates per sample; serving cell with hysteresis.' : 'Synthetic site layout. Set <code>OPENCELLID_TOKEN</code> to use the community database.'],
    ['Satcom', false, `Predictive obstruction model (${esc(m.providers.find(p => p.type === 'satcom')?.terminal ?? 'performance')} terminal). No public route-level Starlink RF telemetry exists; confidence capped at 0.35 until terminal telemetry is ingested.`, 'predictive'],
    ['Calibration', false, 'No measurements loaded. <code>tcs calibrate &lt;csv&gt;</code> fits score→RSRP and per-operator bias from Ofcom drive tests, the Ofcom train study or Network Survey logs.', 'none'],
  ] as [string, boolean, string, string?][]
  let h = `<div class="src-list">`
  for (const [title, ok, text, alt] of items) h += `<div class="src"><h5>${title}${live(ok, 'live', alt ?? 'synthetic')}</h5><p>${text}</p></div>`
  h += `<div class="src"><h5>Model ${esc(m.model_version)}</h5><p>${m.n_samples.toLocaleString()} samples at ${m.route.sample_spacing_m} m · ${(m.length_m / 1000).toFixed(1)} km · scenario switching runs the same link-manager and Wi-Fi model in the browser.</p></div>`
  if (m.warnings?.length) h += `<div class="src"><h5>Warnings</h5><p>${m.warnings.map(esc).join('<br>')}</p></div>`
  h += `</div>`
  $('sourcesBody').innerHTML = h
}

export { qClass }
