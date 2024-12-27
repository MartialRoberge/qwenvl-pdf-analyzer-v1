/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        finance: {
          50: '#f5f8fa',
          100: '#e8f1f5',
          200: '#d1e3eb',
          300: '#a3c7d6',
          400: '#6b9fb8',
          500: '#457b96',
          600: '#2d5f78',
          700: '#1f4558',
          800: '#162f3c',
          900: '#0e1c24',
        },
      },
      animation: {
        'bounce-slow': 'bounce 3s infinite',
      }
    },
  },
  plugins: [],
}
