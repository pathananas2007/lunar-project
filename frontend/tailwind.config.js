/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        lunar: { 900: "#0a0e1a", 800: "#111827", 700: "#1f2937" },
        accent: "#38bdf8"
      }
    }
  },
  plugins: []
}
