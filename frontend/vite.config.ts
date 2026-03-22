import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

const srcDir = fileURLToPath(new URL("./src", import.meta.url));

// Use './' for GitHub Pages project sites; set VITE_BASE in CI if you use a subpath.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": srcDir },
  },
  base: process.env.VITE_BASE ?? "./",
  server: {
    port: 5173,
  },
});
