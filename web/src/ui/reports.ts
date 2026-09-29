/** The Report menu: download links for the evidence packs (Word report + Excel appendix) that `tcs report-all` builds
 *  beside each route bundle, in data/<route>/reports/ with an index.json. UI only; nothing is generated in the browser. */
import { CONFIG } from '../config'
import { esc } from '../html'

interface ReportFile { file: string; bytes: number }
interface ReportItem { scenario: string; label: string; docx?: ReportFile; xlsx?: ReportFile }
interface ReportIndex { route: string; generated_at: string; reports: ReportItem[] }

const size = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`)
const KINDS = [
  { key: 'docx', name: 'Word report', icon: 'M4 1.5h5L12 4.5V14.5H4zM9 1.5v3h3M6 8h4M6 10.5h4' },
  { key: 'xlsx', name: 'Excel appendix', icon: 'M2.5 3h11v10h-11zM2.5 6.5h11M2.5 9.8h11M6.5 3v10' },
] as const

async function loadIndex(routeId: string): Promise<ReportIndex | null> {
  try {
    const r = await fetch(`${CONFIG.dataRoot}/${encodeURIComponent(routeId)}/reports/index.json`)
    if (!r.ok) return null
    const idx = await r.json()                               // a server that answers every path with the app page lands in the catch
    return Array.isArray(idx?.reports) && idx.reports.length ? idx : null
  } catch { return null }
}

export function initReports(o: {
  routeId: string
  /** The pack id matching the on-screen scenario, or null when the screen shows something no pack covers. */
  matching: (ids: string[]) => string | null
  /** One-line description of the on-screen scenario (the scenario pill's text). */
  describe: () => string
  onExport: () => void
  /** Called just before the menu opens, to close the other header popover. */
  onOpen: () => void
}) {
  const btn = document.getElementById('btnReport')!, pop = document.getElementById('reportPopover')!
  const body = document.getElementById('reportBody')!, foot = document.getElementById('reportFoot')!
  let idx: ReportIndex | null | undefined                     // undefined: not fetched yet
  let loading: Promise<void> | null = null

  const exportBtn = (label: string) => `<button class="btn" type="button" data-act="export"><svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2v8M4.5 6.5 8 10l3.5-3.5M3 13h10"/></svg>${label}</button>`
  const render = () => {
    if (idx === undefined) { body.innerHTML = '<p class="hint rp-msg">Looking for this route’s evidence pack…</p>'; foot.innerHTML = ''; return }
    if (idx === null) {
      body.innerHTML = `<p class="rp-msg">No evidence pack was published with this route yet. The monthly publish builds one for every route (<code>tcs report-all</code>).</p>
        <div class="rp-note">Meanwhile, Export saves the scenario on screen as CSV + JSON.${exportBtn('Export on-screen scenario')}</div>`
      foot.innerHTML = '<kbd>Esc</kbd> closes.'
      return
    }
    const match = o.matching(idx.reports.map(r => r.scenario))
    const base = `${CONFIG.dataRoot}/${encodeURIComponent(o.routeId)}/reports/`
    body.innerHTML = idx.reports.map((r, k) => `
      <div class="rp-item${r.scenario === match ? ' on' : ''}" role="group" aria-labelledby="rpLabel${k}">
        <div class="rp-label"><span id="rpLabel${k}">${esc(r.label)}</span>${r.scenario === match ? '<span class="rp-tag">On screen</span>' : ''}</div>
        <div class="rp-files">${KINDS.map(kd => { const f = r[kd.key]; return f ? `
          <a class="btn rp-file" href="${base}${encodeURIComponent(f.file)}" download aria-label="${kd.name}, ${esc(r.label)}, ${size(f.bytes)}">
            <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="${kd.icon}"/></svg><span>${kd.name}</span><small>${size(f.bytes)}</small></a>` : '' }).join('')}
        </div>
      </div>`).join('') + (match ? '' : `
      <div class="rp-note"><b>The scenario on screen has no pack</b><span>On screen: ${esc(o.describe())}. The packs above cover the standard scenarios. Export saves the on-screen one as CSV + JSON; a Word + Excel pack for it needs <code>tcs report</code>.</span>${exportBtn('Export on-screen scenario')}</div>`)
    const built = new Date(idx.generated_at)
    const when = Number.isNaN(built.getTime()) ? '' : `Built ${built.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })} from the same route data as this map. `
    foot.innerHTML = `${when}Model predictions, not measurements. <kbd>Esc</kbd> closes.`
  }

  const set = (open: boolean, returnFocus = false) => {
    if (pop.hidden === !open) return
    if (open) o.onOpen()
    pop.hidden = !open
    btn.setAttribute('aria-expanded', String(open))
    if (!open) { if (returnFocus) btn.focus(); return }
    const r = btn.getBoundingClientRect(); pop.style.right = `${Math.max(8, innerWidth - r.right)}px`
    render()
    const focusFirst = () => (pop.querySelector<HTMLElement>('.rp-item.on a') ?? pop.querySelector<HTMLElement>('a, button') ?? pop).focus()   // the pack on screen first
    if (idx !== undefined) { focusFirst(); return }
    pop.focus()
    loading ??= loadIndex(o.routeId).then(v => { idx = v })
    loading.then(() => { if (!pop.hidden) { render(); if (pop.contains(document.activeElement) || document.activeElement === document.body) focusFirst() } })
  }
  btn.addEventListener('click', e => { e.stopPropagation(); set(pop.hidden) })
  pop.addEventListener('click', e => {
    e.stopPropagation()
    if ((e.target as HTMLElement).closest('[data-act="export"]')) { o.onExport(); set(false, true) }
  })
  document.addEventListener('click', () => set(false))
  return { set }
}
