import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const { outputText } = ts.transpileModule(readFileSync(new URL('../src/train.ts', import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext } })
const { parseProject, derive } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
const design = project => JSON.stringify({ app: 'motion-applied-train-studio', version: 1, project })

for (const value of [-1, 0.5, null, '3', true, 1001, Infinity]) {
  test(`reject invalid equipment count ${String(value)}`, () => {
    assert.throws(() => parseProject(design({ cars: [{ aps: value }] })), /integer/)
  })
}
test('reject malformed structures and duplicate equipment ids', () => {
  for (const project of [[], {}, { cars: {} }, { cars: [] }, { cars: [{ custom: [] }] }, { cars: [{}], types: [{ id: 'x', name: 'X' }, { id: 'x', name: 'X' }] }]) assert.throws(() => parseProject(design(project)))
})
test('valid example derives without mutation', () => {
  const project = parseProject(readFileSync(new URL('../public/examples/azuma-5car.train.json', import.meta.url), 'utf8'))
  const before = JSON.stringify(project)
  assert.equal(derive(project, {}).carriages.length, 5)
  assert.equal(JSON.stringify(project), before)
})
test('reject oversized input before JSON parsing', () => {
  assert.throws(() => parseProject(' '.repeat(5 * 1024 * 1024 + 1)), /5 MiB/)
})
