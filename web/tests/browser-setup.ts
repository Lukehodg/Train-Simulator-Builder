import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'

export default function setup() {
  const root = path.resolve(process.cwd(), '..')
  const local = path.join(root, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python')
  const python = process.env.TCS_TEST_PYTHON || (existsSync(local) ? local : 'python')
  execFileSync(python, [path.join(root, 'tests/build_browser_fixture.py')], { cwd: root, stdio: 'inherit', timeout: 60000 })
}
