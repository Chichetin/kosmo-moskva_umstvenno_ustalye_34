import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// In dev, /report must open the citizen page, not fall back to the operator app.
const reportSlash: Plugin = {
  name: 'report-slash',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url === '/report' || req.url?.startsWith('/report?')) {
        res.statusCode = 301;
        res.setHeader('Location', req.url.replace('/report', '/report/'));
        res.end();
        return;
      }
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), reportSlash],
  base: '/',
  build: {
    outDir: '../src/floodvalue/app/static',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        report: resolve(__dirname, 'report/index.html'),
      },
    },
  },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
});
