// Parity check: the in-browser model (src/sim/model.ts) must reproduce the Python model (tcs/model/*) from the exported
// bundle, for every link policy x vehicle profile x weather. Expected values come from tests/parity_expected.py.
//
//   npm run test:parity -- <parity_expected.json> [<public dir with data/<route>/>]
//
// The TypeScript sources are loaded through Vite's SSR loader, so the check runs the exact modules the viewer ships.
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const webDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const [expectedPath, publicDir = resolve(webDir, 'public')] = process.argv.slice(2)
if (!expectedPath) {
  console.error('usage: npm run test:parity -- <parity_expected.json> [<public dir>]')
  process.exit(2)
}
const expected = JSON.parse(readFileSync(expectedPath, 'utf8'))
const TOL = { bonded: 0.01, wifi: 0.01, conf: 0.001 }   // Mbps, score points, confidence (Python side is rounded to 4 dp)

// loadRoute() fetches `data/<route>/...` relative to the page; serve those requests from the public folder.
globalThis.fetch = async url => {
  try {
    const buf = readFileSync(resolve(publicDir, String(url)))
    return { ok: true, json: async () => JSON.parse(buf.toString('utf8')), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
  } catch {
    return { ok: false }
  }
}

// No dependency pre-bundling scan (it would crawl index.html and the Train Builder) and no file watcher: SSR imports only.
const server = await createServer({
  root: webDir, configFile: false, logLevel: 'error', appType: 'custom',
  server: { middlewareMode: true, hmr: false, watch: null }, optimizeDeps: { noDiscovery: true, include: [] },
})
let failures = 0
try {
  const { loadRoute } = await server.ssrLoadModule('/src/data.ts')
  const { simulate } = await server.ssrLoadModule('/src/sim/model.ts')
  const data = await loadRoute(expected.route_id)
  const ids = data.meta.providers.map(p => p.id)
  if (data.n !== expected.n || ids.join() !== expected.providers.join()) {
    throw new Error(`bundle does not match the expected file: ${data.n} samples [${ids}] vs ${expected.n} [${expected.providers}]`)
  }
  const thresholds = Object.values(data.meta.sim.passenger_wifi.classes).map(Number)
  const nearThreshold = s => thresholds.some(t => Math.abs(s - t) < TOL.wifi)
  const linkNames = mask => ids.filter((_, k) => mask & (1 << k)).join('+') || '-'

  for (const sc of expected.scenarios) {
    const r = simulate(data, { policy: sc.policy, vehicle: sc.vehicle, weather: sc.weather })
    const worst = { bonded: 0, wifi: 0, conf: 0 }
    let clsBad = 0, activeBad = 0, first = null
    for (let i = 0; i < data.n; i++) {
      const d = { bonded: Math.abs(r.bonded[i] - sc.bonded[i]), wifi: Math.abs(r.wifi[i] - sc.wifi[i]), conf: Math.abs(r.conf[i] - sc.conf[i]) }
      let bad = false
      for (const k of Object.keys(d)) { worst[k] = Math.max(worst[k], d[k]); if (d[k] > TOL[k]) bad = true }
      if (r.active[i] !== sc.active[i]) { activeBad++; bad = true }
      if (r.wifiClass[i] !== sc.cls[i] && !nearThreshold(sc.wifi[i])) { clsBad++; bad = true }
      if (bad && first === null) {
        first = `sample ${i}: bonded ${r.bonded[i].toFixed(3)} vs ${sc.bonded[i]}, wifi ${r.wifi[i].toFixed(3)} vs ${sc.wifi[i]}, ` +
          `links ${linkNames(r.active[i])} vs ${linkNames(sc.active[i])}`
      }
    }
    const name = `${sc.policy} / ${sc.vehicle} / ${sc.weather}`
    if (first === null) continue
    failures++
    console.error(`FAIL ${name}: max diff bonded ${worst.bonded.toFixed(4)} Mbps, wifi ${worst.wifi.toFixed(4)}, conf ${worst.conf.toFixed(4)}; ` +
      `${activeBad} link-set and ${clsBad} class mismatches; first at ${first}`)
  }
  const total = expected.scenarios.length
  console.log(`${failures ? 'FAIL' : 'ok'} browser model vs Python: ${total - failures}/${total} scenarios match on ${data.n} samples of ${expected.route_id}`)
} finally {
  await server.close()
}
process.exit(failures ? 1 : 0)
