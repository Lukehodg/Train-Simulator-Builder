import { defineConfig } from 'vite'

export default defineConfig({
  base: './',   // relative asset paths so the packaged site works from any folder
  server: { port: 5173, strictPort: false },
  build: { target: 'es2022', chunkSizeWarningLimit: 2500 },
})
