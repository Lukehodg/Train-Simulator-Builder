import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

// Each module is transpiled to a data: URL; relative imports are pointed at their own data: URLs (it has no base to resolve them).
const moduleUrl = (file, deps = {}) => {
  let { outputText } = ts.transpileModule(readFileSync(new URL(`../src/${file}`, import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext } })
  for (const [spec, url] of Object.entries(deps)) outputText = outputText.replaceAll(`'${spec}'`, `'${url}'`)
  return `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`
}
const htmlUrl = moduleUrl('html.ts')
const { esc } = await import(htmlUrl)
const { parseProject, derive, renderDesign } = await import(moduleUrl('train.ts', { './html': htmlUrl }))
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
test('text is escaped for markup and attributes', () => {
  assert.equal(esc(`<img src=x onerror="a('b')">&`), '&#60;img src=x onerror=&#34;a(&#39;b&#39;)&#34;&#62;&#38;')
  assert.equal(esc(null), '')
  assert.equal(esc(42), '42')
})
test('a design title from an uploaded file is shown as text', () => {
  const project = parseProject(readFileSync(new URL('../public/examples/azuma-5car.train.json', import.meta.url), 'utf8'))
  const html = renderDesign(derive({ ...project, title: '<img src=x onerror=alert(1)>' }, {}), '')
  assert.ok(html.includes('&#60;img src=x onerror=alert(1)&#62;') && !html.includes('<img'))
})
