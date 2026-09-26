import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: { outDir: '../src/floodvalue/app/static', emptyOutDir: true },
  // make site on :8000 serves the operator API and media (/report, /media, /api/media) on one origin
  server: { proxy: { '/api': 'http://127.0.0.1:8000', '/report': 'http://127.0.0.1:8000', '/media': 'http://127.0.0.1:8000' } },
});
