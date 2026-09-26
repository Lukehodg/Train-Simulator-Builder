import { defineConfig } from 'vite'

export default defineConfig({
  base: './',   // relative asset paths so the packaged site works from any folder
  server: { port: 5173, strictPort: false },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 2500 },
})
