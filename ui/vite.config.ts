import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// EN: Vite config for React + TS
// FA: تنظیمات وایت برای ری‌اکت و تایپ‌اسکریپت
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
