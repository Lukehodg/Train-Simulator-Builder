import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests', testMatch: '**/*.spec.ts', timeout: 30000,
  workers: 1, fullyParallel: false,
  globalSetup: './tests/browser-setup.ts',
  use: { baseURL: 'http://127.0.0.1:4178', viewport: { width: 1440, height: 1000 },
    reducedMotion: 'reduce', trace: 'retain-on-failure',
    launchOptions: { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] } },
  webServer: { command: 'npm run dev -- --host 127.0.0.1 --port 4178 --strictPort', url: 'http://127.0.0.1:4178', reuseExistingServer: false },
})
