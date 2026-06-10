import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  // "/" for the dockerized/nginx deploy; set VITE_BASE=/Projet-ia/ for GitHub Pages.
  base: process.env.VITE_BASE ?? "/",
});
