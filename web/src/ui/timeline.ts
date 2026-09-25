import { fmtHM, type Store } from '../state'
import { CLASS_NAMES, CLASS_VARS, CONF_VARS, WIFI_CLASSES, classOf, confStep, cssVar, wifiToClass } from '../sim/classify'
import { indexAtDistance } from '../data'
import { envText } from './inspector'

const GL = 78, GR = 12          // gutters: lane labels on the left, breathing room on the right

/**
 * Journey timeline: one aligned track per link over a shared distance axis.
 * Drag to zoom into a section, double-click to reset, hover for a crosshair reading across every track.
 */
export function initTimeline(store: Store, onSeek: (i: number) => void) {
  const canvas = document.getElementById('tl') as HTMLCanvasElement
  const ctx = canvas.getContext('2d')!
  const base = document.createElement('canvas')
  const tip = document.getElementById('tlTip')!
  let W = 0, H = 0, DPR = 1
  let hoverM: number | null = null
  let view: [number, number] = [0, 0]            // visible distance range in metres; [0,0] = whole route
  let drag: { from: number; to: number } | null = null

  const lanesOf = () => {
    const out: { key: string; label: string }[] = [{ key: 'wan', label: 'Wi-Fi' }]
    for (const p of store.state.data.meta.providers) out.push({ key: p.id, label: p.name })
    return out
  }
  // Lane heights derive from the canvas so the chart always fits the dock, whatever its height.
  const geom = () => {
    const lanes = lanesOf()
    const top = 6, gap = 2, axis = 15
    const avail = Math.max(40, H - top - axis)
    const h1 = Math.max(20, Math.min(42, avail * 0.3))
    const h = Math.max(7, (avail - h1 - gap * lanes.length) / (lanes.length - 1))
    const rows: { key: string; label: string; y: number; h: number }[] = []
    let y = top
    lanes.forEach((l, n) => { const hh = n === 0 ? h1 : h; rows.push({ ...l, y, h: hh }); y += hh + gap })
    return { rows, axisY: y + 1 }
  }
  const total = () => store.state.data.meta.length_m
  const range = (): [number, number] => (view[1] > view[0] ? view : [0, total()])
  const mToX = (m: number) => { const [a, b] = range(); return GL + ((m - a) / (b - a)) * (W - GL - GR) }
  const xToM = (x: number) => { const [a, b] = range(); return Math.min(b, Math.max(a, a + ((x - GL) / (W - GL - GR)) * (b - a))) }

  /** Round tick spacing for the visible span. */
  function tickStep(span: number): number {
    const target = span / 8
    for (const s of [1, 2, 5, 10, 20, 25, 50, 100, 200, 500]) if (target <= s * 1000) return s * 1000
    return 1_000_000
  }

  function drawBase() {
    const st = store.state, d = st.data, r = st.sim
    const rect = canvas.getBoundingClientRect(); if (!rect.width) return
    DPR = Math.min(devicePixelRatio, 2); W = rect.width; H = rect.height
    canvas.width = base.width = W * DPR; canvas.height = base.height = H * DPR
    const c = base.getContext('2d')!
    c.setTransform(DPR, 0, 0, DPR, 0, 0); c.clearRect(0, 0, W, H)
    const { rows, axisY } = geom()
    const ink = cssVar('--text'), muted = cssVar('--muted'), line = cssVar('--border'), faint = cssVar('--surface-3')
    const cls = CLASS_VARS.map(cssVar), conf = CONF_VARS.map(cssVar)
    const cols = W - GL - GR
    const [m0, m1] = range(), perPx = (m1 - m0) / cols

    c.font = '500 9.5px "IBM Plex Mono", monospace'; c.textBaseline = 'middle'
    for (const ln of rows) {
      const vis = st.linkVisible[ln.key]
      c.fillStyle = vis ? muted : faint; c.textAlign = 'right'; c.fillText(ln.label, GL - 9, ln.y + ln.h / 2)
      let runStart = GL, runCol: string | null = null
      for (let px = 0; px <= cols; px++) {
        let col: string | null = null
        if (px < cols) {
          const k0 = indexAtDistance(d.distance, m0 + px * perPx), k1 = indexAtDistance(d.distance, m0 + (px + 1) * perPx)
          let worst = -1, cv = 1
          for (let i = k0; i <= k1; i++) {
            if (st.metric === 'conf') cv = Math.min(cv, ln.key === 'wan' ? r.conf[i] : d.base[ln.key].conf[i])
            else worst = Math.max(worst, classOf(st.metric, ln.key, i, d, r))
          }
          col = st.metric === 'conf' ? conf[confStep(cv)] : cls[worst]
        }
        if (col !== runCol) {
          if (runCol) { c.globalAlpha = vis ? 1 : 0.22; c.fillStyle = runCol; c.fillRect(runStart, ln.y, GL + px - runStart, ln.h); c.globalAlpha = 1 }
          runStart = GL + px; runCol = col
        }
      }
    }
    // bonded capacity line across the Wi-Fi strip
    { const ln = rows[0]; c.beginPath()
      for (let px = 0; px < cols; px++) { const i = indexAtDistance(d.distance, m0 + px * perPx); const y = ln.y + ln.h - 2 - Math.min(1, r.bonded[i] / 300) * (ln.h - 4); px ? c.lineTo(GL + px, y) : c.moveTo(GL + px, y) }
      c.strokeStyle = cssVar('--surface'); c.lineWidth = 3; c.stroke(); c.strokeStyle = ink; c.lineWidth = 1.3; c.stroke() }

    // axis: km ticks, stations, tunnels
    c.strokeStyle = line; c.beginPath(); c.moveTo(GL, axisY); c.lineTo(W - GR, axisY); c.stroke()
    const step = tickStep(m1 - m0)
    c.font = '500 9px "IBM Plex Mono", monospace'; c.textAlign = 'center'; c.fillStyle = muted
    for (let m = Math.ceil(m0 / step) * step; m <= m1; m += step) {
      const x = mToX(m)
      c.strokeStyle = line; c.beginPath(); c.moveTo(x, axisY); c.lineTo(x, axisY + 3); c.stroke()
      c.fillStyle = muted; c.fillText(String(Math.round(m / 1000)), x, axisY + 10)
    }
    c.textAlign = 'right'; c.fillStyle = muted; c.fillText('km', GL - 9, axisY + 10)
    c.textAlign = 'center'
    let lastX = -100
    for (const s of d.meta.stations) {
      if (s.distance_m < m0 || s.distance_m > m1) continue
      const x = mToX(s.distance_m)
      c.strokeStyle = s.stop ? line : faint; c.beginPath(); c.moveTo(x, 4); c.lineTo(x, axisY); c.stroke()
      if (x - lastX > 26) { c.fillStyle = s.stop ? ink : muted; c.fillText(s.crs, x, H - 4); lastX = x }
    }
    c.fillStyle = cssVar('--c-outage')
    for (const t of d.meta.tunnels) {
      if (t.to_m < m0 || t.from_m > m1) continue
      c.fillRect(mToX(t.from_m) - 0.5, axisY - 3, Math.max(1.5, mToX(t.to_m) - mToX(t.from_m)), 3)
    }
  }

  function draw(i: number) {
    if (!W) drawBase()
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(base, 0, 0); ctx.setTransform(DPR, 0, 0, DPR, 0, 0)
    const { axisY } = geom(); const d = store.state.data
    const [m0, m1] = range()
    if (drag) {
      const x0 = mToX(Math.min(drag.from, drag.to)), x1 = mToX(Math.max(drag.from, drag.to))
      ctx.fillStyle = cssVar('--accent'); ctx.globalAlpha = .16; ctx.fillRect(x0, 2, x1 - x0, axisY - 2); ctx.globalAlpha = 1
      ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(x0, 2); ctx.lineTo(x0, axisY); ctx.moveTo(x1, 2); ctx.lineTo(x1, axisY); ctx.stroke()
    }
    const pm = d.distance[i]
    if (pm >= m0 && pm <= m1) {
      const x = mToX(pm), acc = cssVar('--accent')
      ctx.strokeStyle = acc; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, 4); ctx.lineTo(x, axisY); ctx.stroke()
      ctx.fillStyle = acc; ctx.beginPath(); ctx.moveTo(x - 5, 1); ctx.lineTo(x + 5, 1); ctx.lineTo(x, 7); ctx.closePath(); ctx.fill()
    }
    if (hoverM != null && hoverM >= m0 && hoverM <= m1) {
      const hx = mToX(hoverM)
      ctx.strokeStyle = cssVar('--muted'); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(hx, 4); ctx.lineTo(hx, axisY); ctx.stroke(); ctx.setLineDash([])
    }
    const sel = store.state.selected
    if (sel != null && d.distance[sel] >= m0 && d.distance[sel] <= m1) {
      const sx = mToX(d.distance[sel]); ctx.fillStyle = cssVar('--text')
      ctx.beginPath(); ctx.moveTo(sx - 4, axisY + 1); ctx.lineTo(sx + 4, axisY + 1); ctx.lineTo(sx, axisY - 5); ctx.closePath(); ctx.fill()
    }
    if (view[1] > view[0]) {
      const label = `${(view[0] / 1000).toFixed(0)}–${(view[1] / 1000).toFixed(0)} km · double-click to reset`
      ctx.font = '500 9px "IBM Plex Mono", monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      const w = ctx.measureText(label).width + 12
      ctx.fillStyle = cssVar('--accent'); ctx.globalAlpha = .12; ctx.fillRect(GL + 4, 3, w, 14); ctx.globalAlpha = 1
      ctx.fillStyle = cssVar('--accent'); ctx.fillText(label, GL + 10, 10)
    }
  }

  function showTip(x: number) {
    const st = store.state, d = st.data, r = st.sim, i = indexAtDistance(d.distance, hoverM!)
    const cls = CLASS_VARS.map(cssVar)
    let h = `<div class="row hd"><span>${(d.distance[i] / 1000).toFixed(1)} km · ${fmtHM(d.meta.departure, d.t[i])}</span><b>${envText(store, i)}</b></div>`
    h += `<div class="row"><span><i style="background:${cls[wifiToClass(r.wifiClass[i])]}"></i>Wi-Fi · ${WIFI_CLASSES[r.wifiClass[i]]}</span><b>${Math.round(r.bonded[i])} Mbps</b></div>`
    for (const p of d.meta.providers) {
      const L = r.links[p.id], c = classOf('quality', p.id, i, d, r)
      h += `<div class="row"><span><i style="background:${cls[c]}"></i>${p.name} · ${CLASS_NAMES[c]}</span><b>${L.avail[i] ? Math.round(L.cap[i]) + ' Mbps' : L.reason[i].toLowerCase().replace(/_/g, ' ')}</b></div>`
    }
    tip.innerHTML = h; tip.style.display = 'block'
    const tw = tip.offsetWidth
    tip.style.left = `${Math.min(W - tw - 4, Math.max(0, x - tw / 2))}px`; tip.style.top = '-6px'
  }

  // ---- interaction -------------------------------------------------------------------------
  canvas.addEventListener('pointerdown', e => {
    const x = e.clientX - canvas.getBoundingClientRect().left
    if (x < GL || x > W - GR) return
    canvas.setPointerCapture(e.pointerId)
    drag = { from: xToM(x), to: xToM(x) }
  })
  canvas.addEventListener('pointermove', e => {
    const x = e.clientX - canvas.getBoundingClientRect().left
    if (x < GL || x > W - GR) { hoverM = null; tip.style.display = 'none'; return }
    hoverM = xToM(x)
    if (drag) { drag.to = hoverM; tip.style.display = 'none'; return }
    showTip(x)
  })
  canvas.addEventListener('pointerup', e => {
    if (!drag) return
    const x = e.clientX - canvas.getBoundingClientRect().left
    const spanPx = Math.abs(mToX(drag.to) - mToX(drag.from))
    const [a, b] = [Math.min(drag.from, drag.to), Math.max(drag.from, drag.to)]
    drag = null
    if (spanPx > 8) { view = [a, b]; drawBase() }                        // brush → zoom to selection
    else if (x >= GL && x <= W - GR) {                                    // click → seek + inspect
      const i = indexAtDistance(store.state.data.distance, xToM(x))
      onSeek(i); store.set({ selected: i })
    }
  })
  canvas.addEventListener('dblclick', () => { view = [0, 0]; drawBase() })
  canvas.addEventListener('pointerleave', () => { hoverM = null; drag = null; tip.style.display = 'none' })
  canvas.addEventListener('wheel', e => {                                 // wheel zooms about the cursor
    if (!W) return
    e.preventDefault()
    const x = e.clientX - canvas.getBoundingClientRect().left
    if (x < GL || x > W - GR) return
    const [a, b] = range(), at = xToM(x), f = e.deltaY > 0 ? 1.25 : 0.8
    const span = Math.min(total(), Math.max(1000, (b - a) * f))
    let n0 = at - (at - a) * (span / (b - a)), n1 = n0 + span
    if (n0 < 0) { n1 -= n0; n0 = 0 }
    if (n1 > total()) { n0 -= n1 - total(); n1 = total() }
    view = span >= total() ? [0, 0] : [Math.max(0, n0), Math.min(total(), n1)]
    drawBase()
  }, { passive: false })

  new ResizeObserver(() => drawBase()).observe(canvas)
  store.on((_s, changed) => { if (changed.has('sim') || changed.has('metric') || changed.has('theme') || changed.has('linkVisible') || changed.has('data')) drawBase() })
  return { draw, redraw: drawBase }
}
