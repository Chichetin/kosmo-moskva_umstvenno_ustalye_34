import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build goes straight into the Python package so `make app` works without Node (D-013).
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/floodvalue/app/static", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  server: { proxy: { "/api": "http://127.0.0.1:8050" } },
});
