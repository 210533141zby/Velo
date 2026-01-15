/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom palette overhaul
        primary: {
          DEFAULT: '#D06847', // Deep Terracotta
          hover: '#B55739',   // Darker Terracotta
        },
        surface: {
          DEFAULT: '#FFFFFF', // Pure White
          subtle: '#F7F7F7',  // Off-white for structure
        },
        text: {
          main: '#171717',    // Neutral-900
          sub: '#737373',     // Neutral-500
        },
        border: {
          line: '#F5F5F5',    // Neutral-100
        },
        // Keeping claude for backward compatibility if needed, but updated to match new palette
        claude: {
          DEFAULT: '#D06847',
          500: '#D06847',
          600: '#B55739',
        },
        stone: {
          ...require('tailwindcss/colors').stone,
          800: 'rgba(55, 53, 47)',
          900: 'rgba(31, 30, 29)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        serif: ['"Merriweather"', 'Georgia', 'serif'], // Editorial
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      borderRadius: {
        'xl': '0.75rem',
        '2xl': '1rem',
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'float': '0 10px 30px -10px rgba(0, 0, 0, 0.1)',
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.stone.800'),
            maxWidth: 'none',
            h1: {
              fontFamily: theme('fontFamily.serif'),
              fontWeight: '700',
              color: theme('colors.stone.900'),
            },
            h2: {
              fontFamily: theme('fontFamily.serif'),
              fontWeight: '600',
              color: theme('colors.stone.900'),
            },
            code: {
              color: theme('colors.claude.600'),
              backgroundColor: theme('colors.stone.100'),
              borderRadius: '0.375rem',
            },
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
