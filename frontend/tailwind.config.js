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
              color: '#171717',
            },
            h2: {
              fontFamily: theme('fontFamily.serif'),
              fontWeight: '600',
              color: '#171717',
            },
            h3: {
              fontFamily: theme('fontFamily.serif'),
              fontWeight: '600',
              color: '#171717',
            },
            h4: {
              fontFamily: theme('fontFamily.serif'),
              fontWeight: '600',
              color: '#171717',
            },
            // 1. 修复代码块 (Code Block) - 改为 Notion 风格的浅灰底
            pre: {
              backgroundColor: '#F3F4F6', // bg-gray-100
              color: '#1F2937',           // text-gray-800
              border: '1px solid #E5E7EB', // border-gray-200
              borderRadius: '0.5rem',
              padding: '1rem',
            },
            'pre code': {
              backgroundColor: 'transparent',
              color: 'inherit',
              fontSize: '0.875rem',
              fontFamily: theme('fontFamily.mono'),
            },
            // 2. 修复行内代码 (Inline Code)
            'code::before': { content: '""' }, // 去掉丑陋的反引号
            'code::after': { content: '""' },
            code: {
              backgroundColor: '#F3F4F6',
              color: '#D06847', // 使用陶土色高亮行内代码
              padding: '0.2rem 0.4rem',
              borderRadius: '0.25rem',
              fontWeight: '500',
            },
            // 3. 修复引用 (Blockquote) - 陶土色竖线
            blockquote: {
              borderLeftColor: '#D06847', // 陶土色
              borderLeftWidth: '4px',
              backgroundColor: '#FFFBF5', // 极淡的暖色背景
              fontStyle: 'normal',
              padding: '0.5rem 1rem',
              color: '#4B5563',
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
