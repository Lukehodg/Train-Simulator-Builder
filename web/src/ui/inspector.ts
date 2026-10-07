import { fmtHM, type Store } from '../state'
import { CLASS_NAMES, CLASS_VARS, WIFI_CLASSES, classOf, cssVar, qClass } from '../sim/classify'
import { dasAt } from '../sim/model'
import { arrowNav } from './a11y'
import { isGB, LIVE_COVERAGE_MIN, liveCoverageShare } from '../data'
import { esc } from '../html'
import type { CalibrationMeta, Meta } from '../types'

const $ = (id: string) => document.getElementById(id)!

export function envText(store: Store, i: number): string {
  const d = store.state.data
  if (d.inTunnel[i]) return `${d.tunnelName[i] && d.tunnelName[i] !== 'tunnel' ? d.tunnelName[i] : 'Tunnel'}`
  if (d.canopy[i] >= 0.9) return 'Station canopy'
  if (d.cutting[i] > 8) return `Cutting · ${d.cutting[i].toFixed(0)} m`
  if (d.overhead[i] > 0.1) return 'Under a bridge'
  if (d.sky[i] < 0.6) return `${d.lidar[i] ? 'Trees and lineside' : 'Terrain shadow'} · sky ${(d.sky[i] * 100).toFixed(0)} %`
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
      const mc = b.meas ? b.meas[i] : NaN
      if (Number.isFinite(mc) && Math.abs(mc) >= 0.5) rows.push(['measured correction (earlier trips here)', `${mc > 0 ? '+' : '−'}${Math.abs(mc).toFixed(1)} dB`])
      if (vprof.score_offset) rows.push([vprof.score_offset < 0 ? 'carriage penetration loss' : `active antenna (+${vprof.db_offset} dB)`, `${vprof.score_offset > 0 ? '+' : ''}${vprof.score_offset.toFixed(2)}`])
      if (vprof.capacity_factor && vprof.capacity_factor !== 1) rows.push(['antenna / MIMO factor', `×${vprof.capacity_factor}`])
      if (L.reason[i] === 'NOT_FITTED') rows.push(['on this train', 'not fitted: no modem for this network, so the router cannot use it'])
      if (d.inTunnel[i]) rows.push(['tunnel', dasAt(cfg.tunnels.das_tunnels ?? [], d.tunnelName[i], d.distance[i] / 1000) ? `in-tunnel coverage assumed → ${cfg.tunnels.das_score}`
        : cfg.tunnels.portal_decay_m ? `signal from the portals, fading inside → ${L.q[i].toFixed(2)}` : `no coverage → ${cfg.tunnels.default_score}`])
      if (b.hp[i] > 0) rows.push(['handover', `+${cfg.handover.latency_spike_ms} ms · ×${cfg.handover.capacity_factor}`])
    } else {
      if (d.lidar[i]) {   // LiDAR skyline: cutting walls and trees are already in it; a station roof or bridge covers a share of the sky
        rows.push(['sky visibility (LiDAR skyline + terrain)', b.qb[i].toFixed(2)])
        const roof = Math.max(0.85 * d.canopy[i], d.overhead[i])
        if (roof > 0.005) rows.push([d.overhead[i] >= 0.85 * d.canopy[i] ? 'bridge over the line' : 'station canopy', `×${(1 - roof).toFixed(2)}`])
      } else {
        rows.push(['sky visibility (DEM horizon)', b.qb[i].toFixed(2)])
        if (d.canopy[i] > 0) rows.push(['station canopy', `−${(0.85 * d.canopy[i]).toFixed(2)}`])
        if (d.cutting[i] > 0.5) rows.push([`cutting horizon ${d.cutting[i].toFixed(0)} m`, `−${Math.min(0.55, d.cutting[i] / 22).toFixed(2)}`])
      }
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

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const month = (iso: string) => `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${iso.slice(0, 4)}`
const period = (p?: { from: string; to: string }) => p ? `${month(p.from)} – ${month(p.to)}` : ''

function calibrationText(cal: CalibrationMeta | null | undefined): string {
  if (!cal) return 'No measurements loaded. <code>tcs calibrate-national</code> fits every route to a national measurement set (e.g. Network Rail\'s Yellow Train logs); <code>tcs calibrate &lt;csv&gt;</code> fits one route to its own drive tests or modem logs.'
  if (cal.scope === 'route') return 'Fitted to field measurements attached to this route (<code>tcs calibrate</code>): per-operator bias by section and score→RSRP.'
  const r = cal.route?.calibrated, b = cal.route?.uncalibrated
  return `${esc(cal.source ?? 'Field measurements')}: per-operator score bias, cutting loss and how far signal carries into tunnels, fitted over ${cal.routes} routes `
    + `on ${period(cal.fit_period)} and tested on ${period(cal.test_period)}, which the fit never saw. `
    + (r?.points && b?.points
      ? `On this route, ${r.points.toLocaleString()} measured points: RSRP within ±6 dB at ${Math.round((r.within_6db ?? 0) * 100)} %, mean error ${r.mae_db?.toFixed(1)} dB (${b.mae_db?.toFixed(1)} dB uncalibrated), bias ${(r.bias_db ?? 0) >= 0 ? '+' : '−'}${Math.abs(r.bias_db ?? 0).toFixed(1)} dB. `
      : cal.route?.measurements?.fit ? 'This route was measured only in the fit period, so there is no independent test on it. '
      : 'No measurements along this route: the national fit applies as it is. ')
    + 'LTE only, and networks have grown since, so it checks the model rather than today\'s coverage.'
    + currentText(cal)
}

function currentText(cal: CalibrationMeta): string {
  let h = ''
  const cc = cal.current_check, here = cc?.route
  if (cc?.overall?.points) {
    h += ` <br><br>Checked against ${esc(cc.source ?? 'later measurements')} 4G, ${period({ from: cc.fit_period!.from, to: cc.test_period!.to })}: `
      + (here?.points ? `on this route, ${here.points.toLocaleString()} points, mean error ${here.mae_db?.toFixed(1)} dB` : `not measured on this route; all routes, mean error ${cc.overall.mae_db?.toFixed(1)} dB`)
      + ` once their level (${(cc.level_offset_db ?? 0) >= 0 ? '+' : '−'}${Math.abs(cc.level_offset_db ?? 0).toFixed(0)} dB, the scanner's uncorrected antenna and cable) is matched.`
    const wc = cc.with_measured_corrections, wr = wc?.route
    if (wc?.overall?.points) h += ` With corrections from the earlier trips' measurements: ${wr?.points ? `${wr.mae_db?.toFixed(1)} dB on this route` : `${wc.overall.mae_db?.toFixed(1)} dB on all routes`}.`
  }
  const fg = cal.five_g, nets = fg?.route?.networks
  if (fg?.networks && nets) {
    const shown = Object.entries(nets).filter(([n]) => fg.networks![n] > 0)        // a network never seen: its 5G bands were not scanned
    h += ` <br><br>5G usable along ${shown.map(([n, v]) => `${esc(n)} ${Math.round(v * 100)} %`).join(', ')} of measured points (${period(fg.period)}; ${esc((fg.bands ?? []).join(', '))} scanned, not 3.4–3.8 GHz, where Three and Vodafone carry most 5G).`
  }
  return h
}

const SURVEYS: Record<string, string> = { lidar_ea: 'Environment Agency', lidar_wales: 'Welsh Government', lidar_scotland: 'Scottish Remote Sensing Portal' }

function lidarText(m: Meta): string {
  if (!m.lidar_share) return ''
  const by = Object.entries(m.lidar_sources ?? {}).map(([k, v]) => `${esc(SURVEYS[k] ?? k)}${Object.keys(m.lidar_sources!).length > 1 ? ` ${Math.round(v * 100)} %` : ''}`).join(', ')
  return ` Within 60 m of the track, open 2 m LiDAR (${by}; OGL) on ${Math.round(m.lidar_share * 100)} % of the route: cutting walls and embankments from the ground model, `
    + 'and trees, buildings and bridges over the line on the skyline from the surface model.'
}

function mastsText(m: Meta): string {
  const f = m.fitted_masts
  if (!f?.masts) return ''
  return ` ${f.masts.toLocaleString()} masts placed from Network Rail Global View 4G logs (from how their signal rises and falls along the track) `
    + `stand in for OpenCellID's positions of the same masts, which are typically about 1 km out; they serve ${Math.round(f.serving_share * 100)} % of the route.`
}

function correctionsText(m: Meta): string {
  const c = m.measured_corrections!
  const sets = c.sources.map(s => `${esc(s.preset === 'global_view_4g' ? 'Network Rail Global View 4G' : s.preset === 'yellow_train' ? 'Network Rail Yellow Train' : s.preset)}`
    + `${s.from ? ` (${esc(s.from)} to ${esc(s.to ?? '')})` : ''}${s.weight < 1 ? ` at ${s.weight}× weight` : ''}`).join(' and ')
  return `Where trains measured this route, the model's error at those measurements, smoothed over about 100 m along the track, `
    + `corrects its prediction: ${Math.round(c.share * 100)} % of the open-air route and networks here, from ${sets || 'scanner logs'}. `
    + 'Signal at a spot repeats from trip to trip, so earlier passes are the best guide to later ones; on later trips this cut the average error from 9.7 to 8.3 dB.'
}

function renderSources(store: Store) {
  const m = store.state.data.meta
  const live = (flag: boolean, liveLabel = 'live', synthLabel = 'synthetic') => `<em class="${flag ? 'live' : 'synth'}">${flag ? liveLabel : synthLabel}</em>`
  const geomLive = m.geometry_source === 'osm' || m.geometry_source === 'file'
  const straight = m.geometry_source === 'osm' ? m.geometry_straight_legs ?? [] : []   // legs with no rail path, drawn straight
  const terrLive = m.terrain_source !== 'synthetic_terrain'
  const covShare = liveCoverageShare(m), covLive = covShare >= LIVE_COVERAGE_MIN
  const sitesPrior = m.coverage_sources.includes('opencellid_sites')   // the US prior: an estimate, not an operator prediction
  const cellLive = m.cell_source === 'opencellid'
  const items = [
    ['Route geometry', geomLive && !straight.length, m.geometry_source === 'ntad' ? 'US DOT National Transportation Atlas Database line for this Amtrak service, used because the OpenStreetMap track was unavailable for this build: generalised, with no tunnel, cutting or line-speed tags (tunnels are placed from the route file, approximately).' : m.geometry_source === 'osm' ? 'OpenStreetMap rail network, routed station-to-station (ODbL). Tunnel / cutting / embankment / bridge / maxspeed tags carried per 50 m sample.'
      + (straight.length ? ` No rail path was found for ${esc(straight.map(s => s.replace('-', '–')).join(', '))}: drawn as a straight line there, without tunnels, cuttings or line speeds.` : '') : m.geometry_source === 'file' ? 'Infrastructure-manager / curated centreline file.' : 'Spline through approximate station coordinates. Run <code>tcs run</code> without <code>--offline</code> to fetch the OSM centreline.', straight.length || m.geometry_source === 'ntad' ? 'partly live' : undefined],
    ['Terrain & sky visibility', terrLive, terrLive ? `${esc(m.terrain_source)}: 30 m DEM, 16-ray horizon per sample, solid-angle sky fraction above the terminal's minimum elevation.${lidarText(m)}` : 'Procedural terrain. 3D terrain rendering is disabled until real elevation is available.'],
    ['Cellular coverage prior', covLive && !sitesPrior, sitesPrior && covLive ? 'Estimated from how far the track is from each network\'s nearest OpenCellID sites (with a lift for nearby 5G sites and for dense sites). There is no operator coverage prediction behind it and no US measurements have calibrated it yet, so treat it as indicative; confidence is held at 0.40.'
      : covLive ? `${esc(m.coverage_sources.join(', '))} — operator predictions on Ofcom's 50 m grid mapped to a model score; never presented as measured RSRP.`
      : covShare > 0 ? `Only ${Math.round(covShare * 100)} % of the route has Ofcom predictions (usually the Ofcom call quota ran out part-way); the rest is a neutral stand-in. Rebuild the route once the quota resets.`
      : isGB(m) ? 'Synthetic prior (noise field). Set <code>OFCOM_API_KEY</code>, or enable Connected Nations open data, to replace it.'
      : 'Synthetic prior (noise field). Set <code>OPENCELLID_TOKEN</code> so the prior can be estimated from the networks\' cell sites.', sitesPrior && covLive ? 'estimate' : covShare > 0 && !covLive ? 'partly live' : undefined],
    ['Cell sites', cellLive, cellLive ? 'OpenCellID corridor extract (CC BY-SA 4.0). Logical cells; top-5 candidates per sample; serving cell with hysteresis.' + mastsText(m) : 'Synthetic site layout. Set <code>OPENCELLID_TOKEN</code> to use the community database.'],
    ['Satcom', false, `Predictive obstruction model (${esc(m.providers.find(p => p.type === 'satcom')?.terminal ?? 'performance')} terminal). No public route-level Starlink RF telemetry exists; confidence capped at 0.35 until terminal telemetry is ingested.`, 'predictive'],
    ['Calibration', !!m.calibration, calibrationText(m.calibration), m.calibration ? undefined : 'none'],
    ...(m.measured_corrections ? [['Measured corrections', true, correctionsText(m), 'measured'] as [string, boolean, string, string]] : []),
  ] as [string, boolean, string, string?][]
  let h = `<div class="src-list">`
  for (const [title, ok, text, alt] of items) h += `<div class="src"><h5>${title}${live(ok, title === 'Calibration' ? 'calibrated' : title === 'Measured corrections' ? 'measured' : 'live', alt ?? 'synthetic')}</h5><p>${text}</p></div>`
  h += `<div class="src"><h5>Model ${esc(m.model_version)}</h5><p>${m.n_samples.toLocaleString()} samples at ${m.route.sample_spacing_m} m · ${(m.length_m / 1000).toFixed(1)} km · scenario switching runs the same link-manager and Wi-Fi model in the browser.</p></div>`
  if (m.warnings?.length) h += `<div class="src"><h5>Warnings</h5><p>${m.warnings.map(esc).join('<br>')}</p></div>`
  h += `</div>`
  $('sourcesBody').innerHTML = h
}

export { qClass }
