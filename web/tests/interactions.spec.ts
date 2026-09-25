import { test, expect } from '@playwright/test'
import { readFileSync } from 'node:fs'
import path from 'node:path'

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('response', response => { if (response.url().startsWith('http://127.0.0.1:4178/') && response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`) })
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
  await page.locator('#routeTitle').click()
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

test('route selection reloads and timeline click inspects a sample', async ({ page }) => {
  await page.locator('#routePick').selectOption('second')
  await expect(page).toHaveURL(/route=second/)
  await expect(page.locator('#kpis .kpi')).toHaveCount(6)
  await page.locator('#tl').click({ position: { x: 200, y: 20 } })
  await expect(page.locator('#sampleBody')).not.toBeEmpty()
})
