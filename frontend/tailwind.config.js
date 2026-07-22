/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        petnutri: {
          cream: '#FBF7EF',
          green: '#5B6B3A',
          'green-dark': '#4A5830',
          orange: '#7A3B1E',
          'orange-light': '#C96A3E',
          brown: '#6B4226',
          text: '#2B2620',
          muted: '#8A8375',
        },
      },
      fontFamily: {
        display: ['"Poppins"', 'system-ui', 'sans-serif'],
        body: ['"Inter"', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        xl2: '1.25rem',
      },
    },
  },
  plugins: [],
}
