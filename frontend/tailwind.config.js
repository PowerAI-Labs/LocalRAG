/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#1a1b1e',
          sidebar: '#1f2024',
          input: '#2c2d31',
        }
      }
    },
  },
  plugins: [],
}