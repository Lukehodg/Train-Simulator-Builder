import { test, expect } from '@playwright/test'
import { readFileSync } from 'node:fs'
import path from 'node:path'

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('response', response => {
    if (response.url().endsWith('/reports/index.json')) return   // a route without evidence packs is a normal state (the Report menu says so)
    if (response.url().startsWith('http://127.0.0.1:4178/') && response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`)
  })
  ;(page as any).appErrors = errors
  await page.route('**/*', async route => {
    const url = new URL(route.request().url())
    if (url.hostname !== '127.0.0.1') {
      if (url.pathname.endsWith('style.json')) return route.fulfill({ json: { version: 8, sources: {}, layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#dbe3e9' } }] } })
      return route.abort()
    }
    const file = url.pathname.split('/').pop()!
    if (/\/data\/[^/]+\/(meta.json|route.arrow|cells.arrow)$/.test(url.pathname)) {
      return route.fulfill({ body: readFileSync(path.resolve('tests/.generated', file)), contentType: file.endsWith('.json') ? 'application/json' : 'application/octet-stream' })
    }
    if (url.pathname === '/data/index.json') return route.fulfill({ json: { routes: [{ id: 'ecml_kgx_edb', name: 'Test ECML', origin: 'London', destination: 'Edinburgh', length_km: 590 }, { id: 'second', name: 'Second test route', origin: 'A', destination: 'B', length_km: 590 }] } })
    return route.continue()
  })
  await page.goto('/')
  await expect(page.locator('#kpis .kpi')).toHaveCount(6)
  await expect(page.locator('#map canvas').first()).toBeVisible()
})

test.afterEach(async ({ page }) => { expect((page as any).appErrors).toEqual([]) })

test('playback, keyboard, scrubbing and station selection', async ({ page }) => {
  const clock = page.locator('#clock')
  const start = await clock.textContent()
  await page.locator('#play').click()
  await expect(clock).not.toHaveText(start!)
  await page.locator('#play').click()
  await page.locator('#scrub').fill('20000')
  const scrubbed = await page.locator('#scrub').inputValue()
  expect(scrubbed).toBe('20000')
  await page.locator('#clock').click()   // move focus off the scrubber so arrow keys reach the app
  const before = await clock.textContent()
  await page.keyboard.press('ArrowRight')
  await expect(clock).not.toHaveText(before!)
  await page.locator('#jump').selectOption('1')
  await expect(page.locator('#jump')).toHaveValue('')
})

test('scenario, theme, panels and help remain interactive', async ({ page }) => {
  await page.locator('#scenarioPill').click()
  await page.locator('#preset').selectOption('edge_rail_fleet_connect')
  await page.locator('#weather').selectOption('storm')
  await expect(page.locator('#scenarioSummary')).toContainText('storm')
  await page.keyboard.press('Escape')
  await expect(page.locator('#scenarioPopover')).toBeHidden()
  const theme = await page.locator('html').getAttribute('data-theme')
  await page.locator('#btnTheme').click()
  await expect(page.locator('html')).not.toHaveAttribute('data-theme', theme!)
  await page.locator('#btnInspector').click()
  await expect(page.locator('#app')).toHaveAttribute('data-inspector', 'closed')
  await page.locator('#btnInspector').click()
  await page.locator('#btnHelp').click()
  await expect(page.locator('#help')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.locator('#help')).toBeHidden()
})

test('valid design, rejected malformed design, reset and export', async ({ page }) => {
  await page.locator('[data-tab="train"]').click()
  await page.locator('#designExample').click()
  await expect(page.locator('#trainBody')).toContainText('Azuma')
  await expect(page.locator('#scenarioSummary')).toContainText('Train design')
  const summary = await page.locator('#scenarioSummary').textContent()
  await page.locator('#designFile').setInputFiles({ name: 'invalid.train.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ app: 'motion-applied-train-studio', version: 1, project: { cars: [{ aps: -1 }] } })) })
  await expect(page.locator('#notice')).toContainText('integer from 0 to 1000')
  await expect(page.locator('#scenarioSummary')).toHaveText(summary!)
  await page.locator('#designReset').click()
  await expect(page.locator('#scenarioSummary')).toContainText('Baseline')
  const downloadPromise = page.waitForEvent('download')
  await page.locator('#btnExport').click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toMatch(/\.(csv|json)$/)
  const stream = await download.createReadStream()
  const chunks = []; for await (const chunk of stream!) chunks.push(chunk)
  expect(Buffer.concat(chunks).length).toBeGreaterThan(100)
})

// `tcs report-all` places Word + Excel packs for the standard scenarios beside each route bundle, with an index.json.
test('the Report menu offers the pre-built packs and flags a scenario they do not cover', async ({ page }) => {
  const pack = (sc: string, label: string, xlsxBytes: number) => ({ scenario: sc, label,
    docx: { file: `evidence_ecml_kgx_edb_${sc}.docx`, bytes: 266861 }, xlsx: { file: `evidence_ecml_kgx_edb_${sc}.xlsx`, bytes: xlsxBytes } })
  const reports = [pack('baseline', 'Baseline (config defaults)', 4_100_000), pack('edge_rail_fleet_connect', 'EDGE Rail 5G active antenna + Fleet Connect', 708216)]
  await page.route(url => /\/data\/ecml_kgx_edb\/reports\//.test(url.pathname), route => {
    if (route.request().url().endsWith('/index.json')) return route.fulfill({ json: { route: 'ecml_kgx_edb', generated_at: '2026-09-27T12:00:00+00:00', reports } })
    return route.fulfill({ body: 'pack', contentType: 'application/octet-stream' })
  })
  const btn = page.locator('#btnReport'), menu = page.locator('#reportPopover')
  await btn.click()
  await expect(menu).toBeVisible()
  await expect(btn).toHaveAttribute('aria-expanded', 'true')
  await expect(menu.locator('.rp-item')).toHaveCount(2)
  await expect(menu.locator('.rp-item.on')).toContainText('Baseline')
  await expect(menu.locator('.rp-note')).toHaveCount(0)
  await expect(menu).toContainText(/Built 27 Sept? 2026/)
  const first = menu.locator('.rp-file').first()
  await expect(first).toBeFocused()                          // the pack matching the screen
  await expect(first).toHaveAttribute('href', 'data/ecml_kgx_edb/reports/evidence_ecml_kgx_edb_baseline.docx')
  await expect(menu.locator('.rp-file').nth(1)).toContainText('4.1 MB')
  const [download] = await Promise.all([page.waitForEvent('download'), first.click()])
  expect(download.suggestedFilename()).toBe('evidence_ecml_kgx_edb_baseline.docx')
  await page.keyboard.press('Escape')
  await expect(menu).toBeHidden()
  await expect(btn).toBeFocused()

  await page.locator('#scenarioPill').click()
  await page.locator('#preset').selectOption('edge_rail_fleet_connect')
  await btn.click()                                          // one header popover at a time
  await expect(page.locator('#scenarioPopover')).toBeHidden()
  await expect(menu.locator('.rp-item.on')).toContainText('EDGE Rail')
  await expect(menu.locator('.rp-item.on .rp-file').first()).toBeFocused()
  await page.keyboard.press('Escape')

  await page.locator('#scenarioPill').click()
  await page.locator('#weather').selectOption('storm')
  await btn.click()
  await expect(menu.locator('.rp-item.on')).toHaveCount(0)
  await expect(menu.locator('.rp-note')).toContainText('no pack')
  await expect(menu.locator('.rp-note')).toContainText('storm')
  const [csv] = await Promise.all([page.waitForEvent('download'), menu.getByRole('button', { name: 'Export on-screen scenario' }).click()])
  expect(csv.suggestedFilename()).toMatch(/\.(csv|json)$/)
  await expect(menu).toBeHidden()
})

test('the Report menu explains when a route has no evidence pack', async ({ page }) => {
  await page.route(url => url.pathname.endsWith('/reports/index.json'), route => route.fulfill({ status: 404, body: 'Not found' }))
  await page.locator('#btnReport').click()
  await expect(page.locator('#reportPopover')).toContainText('No evidence pack was published')
  await expect(page.locator('#reportPopover .rp-file')).toHaveCount(0)
  await expect(page.locator('#reportPopover').getByRole('button', { name: 'Export on-screen scenario' })).toBeFocused()
  await page.locator('#clock').click()                       // a click outside closes it
  await expect(page.locator('#reportPopover')).toBeHidden()
})

test('route selection reloads and timeline click inspects a sample', async ({ page }) => {
  await page.locator('#routePick').selectOption('second')
  await expect(page).toHaveURL(/route=second/)
  await expect(page.locator('#kpis .kpi')).toHaveCount(6)
  await page.locator('#tl').click({ position: { x: 200, y: 20 } })
  await expect(page.locator('#sampleBody')).not.toBeEmpty()
})

// The basemap is a third-party style. Offline, behind a proxy that blocks it, or during an outage, the viewer
// must still start on a plain map instead of waiting for it forever.
for (const [label, handle] of [
  ['is refused', (route: any) => route.abort()],
  ['never answers', () => {}],                                        // never answered: the style timeout has to kick in
] as const) {
  test(`the viewer starts when the basemap ${label}`, async ({ page }) => {
    test.setTimeout(60_000)
    await page.route(url => url.pathname.endsWith('style.json'), handle as any)   // registered last, so it overrides the stub
    await page.reload()
    await expect(page.locator('#kpis .kpi')).toHaveCount(6, { timeout: 25_000 })
    await expect(page.locator('#simState')).toHaveAttribute('data-state', 'paused')
    await expect(page.locator('#noticeMsg')).toContainText('basemap could not be loaded')
    await page.locator('#play').click()
    await expect(page.locator('#simState')).toHaveAttribute('data-state', 'running')
    await page.locator('#btnTheme').click()                  // a theme switch keeps the plain map rather than retrying the dead basemap
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
    await expect(page.locator('#simState')).toHaveAttribute('data-state', 'running')
  })
}

// Routes built with real elevation switch MapLibre's 3D terrain on, which exercises deck.gl's terrain code path
// (it used to throw on every render with MapLibre 6). The terrain tiles themselves are blocked here, as offline.
test('the viewer runs with 3D terrain switched on', async ({ page }) => {
  const meta = JSON.parse(readFileSync(path.resolve('tests/.generated', 'meta.json'), 'utf-8'))
  await page.route(url => /\/data\/[^/]+\/meta\.json$/.test(url.pathname), route => route.fulfill({ json: { ...meta, terrain_source: 'copernicus_glo30' } }))
  await page.reload()
  await expect(page.locator('#kpis .kpi')).toHaveCount(6)
  await page.locator('.rail-btn[data-panel="layers"]').click()
  await expect(page.locator('#ly_terrain')).toBeEnabled()
  await expect(page.locator('#ly_terrain')).toBeChecked()
  await page.locator('#play').click()
  await page.waitForTimeout(1500)                            // several map renders with terrain on; afterEach checks for page errors
  await expect(page.locator('#simState')).toHaveAttribute('data-state', 'running')
})

// Station, provider and route names come from OSM, OpenCelliD and route config; none of it may become markup.
test('names from the route data are shown as text, never as markup', async ({ page }) => {
  const meta = JSON.parse(readFileSync(path.resolve('tests/.generated', 'meta.json'), 'utf-8'))
  const tag = (id: string) => `<img id="${id}" src="data:," onerror="window.__xss=1">`
  meta.stations[1].name = `Halt ${tag('xss-station')}`
  meta.providers[0].name = `${meta.providers[0].name} ${tag('xss-provider')}`
  meta.warnings = [...(meta.warnings ?? []), `Check ${tag('xss-warning')}`]
  await page.route(url => /\/data\/[^/]+\/meta\.json$/.test(url.pathname), route => route.fulfill({ json: meta }))
  await page.route(url => url.pathname === '/data/index.json', route => route.fulfill({ json: { routes: [{ id: 'ecml_kgx_edb', name: `Test ${tag('xss-route')}`, origin: 'London', destination: 'Edinburgh', length_km: 590 }] } }))
  await page.reload()
  await expect(page.locator('#kpis .kpi')).toHaveCount(6)
  await page.locator('.tab[data-tab="sources"]').click()
  await expect(page.locator('#sourcesBody')).toContainText('Check <img id="xss-warning"')
  await expect(page.locator('#jump option').nth(2)).toHaveText(/Halt <img id="xss-station"/)
  await expect(page.locator('#routePick option').first()).toHaveText(/Test <img id="xss-route"/)
  await expect(page.locator('#liveLinks')).toContainText('<img id="xss-provider"')
  await expect(page.locator('[id^="xss-"]')).toHaveCount(0)
  expect(await page.evaluate(() => (window as any).__xss)).toBeUndefined()
})

// `tcs run --policy / --weather` records the scenario it simulated; the viewer opens on it, not on the defaults.
test('the viewer opens on the policy and weather the route was built with', async ({ page }) => {
  const meta = JSON.parse(readFileSync(path.resolve('tests/.generated', 'meta.json'), 'utf-8'))
  meta.sim = { ...meta.sim, weather: 'storm', wan: { ...meta.sim.wan, policy: 'FAILOVER' } }
  await page.route(url => /\/data\/[^/]+\/meta\.json$/.test(url.pathname), route => route.fulfill({ json: meta }))
  await page.reload()
  await expect(page.locator('#kpis .kpi')).toHaveCount(6)
  await expect(page.locator('#weather')).toHaveValue('storm')
  await expect(page.locator('#policy')).toHaveValue('FAILOVER')
})

test('cell sites are shown by default', async ({ page }) => {
  await page.locator('.rail-btn[data-panel="layers"]').click()
  await expect(page.locator('#ly_cells')).toBeChecked()
  await page.locator('#ly_cells').uncheck()                  // and can still be switched off
  await expect(page.locator('#ly_cells')).not.toBeChecked()
})
