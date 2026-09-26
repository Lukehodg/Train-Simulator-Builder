// Executed by pytest with freshly computed Python fixtures on stdin.
import { readFileSync } from 'node:fs'
import assert from 'node:assert/strict'
import ts from 'typescript'

const source = readFileSync(new URL('../src/sim/model.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } })
const { simulate } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
const cases = JSON.parse(readFileSync(0, 'utf8'))
for (const { data, scenario, expected } of cases) {
  const result = simulate(data, scenario)
  for (const [pid, values] of Object.entries(expected.rsrp ?? {})) {
    for (let i = 0; i < data.n; i++) assert.ok(Math.abs(result.links[pid].rsrp[i] - values[i]) < 0.002, `RSRP ${pid} at ${i}`)
  }
  for (let i = 0; i < data.n; i++) {
    assert.ok(Math.abs(result.bonded[i] - expected.bonded[i]) < 0.002, `${scenario.policy} WAN at ${i}`)
    assert.ok(Math.abs(result.wifi[i] - expected.wifi[i]) < 0.002, `${scenario.policy} Wi-Fi at ${i}`)
    assert.equal(['EXCELLENT', 'GOOD', 'USABLE', 'POOR', 'OUTAGE'][result.wifiClass[i]], expected.cls[i])
  }
}
console.log(`${cases.length} Python/browser parity scenarios passed`)
