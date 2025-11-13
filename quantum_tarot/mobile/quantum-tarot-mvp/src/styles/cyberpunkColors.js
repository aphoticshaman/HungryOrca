/**
 * CYBERPUNK NEON COLOR PALETTE
 * LPMUD-style color codes + cyberpunk terminal aesthetics
 */

export const NEON_COLORS = {
  // Primary cyberpunk neons
  cyan: '#00FFFF',
  magenta: '#FF00FF',
  yellow: '#FFFF00',
  green: '#00FF00',

  // High-intensity variants (LPMUD $HI*$ codes)
  hiCyan: '#00FFFF',
  hiMagenta: '#FF00FF',
  hiYellow: '#FFFF00',
  hiWhite: '#FFFFFF',
  hiGreen: '#00FF00',
  hiRed: '#FF0000',
  hiBlue: '#0000FF',

  // Dimmed/normal variants
  dimCyan: '#008B8B',
  dimMagenta: '#8B008B',
  dimYellow: '#8B8B00',
  dimWhite: '#AAAAAA',
  dimGreen: '#008000',
  dimRed: '#8B0000',

  // Terminal background/accents
  bgBlack: '#000000',
  bgDarkCyan: '#001a1a',
  bgDarkMagenta: '#1a001a',

  // Glow colors (for shadows)
  glowCyan: 'rgba(0, 255, 255, 0.8)',
  glowMagenta: 'rgba(255, 0, 255, 0.8)',
  glowYellow: 'rgba(255, 255, 0, 0.8)',
  glowWhite: 'rgba(255, 255, 255, 0.6)',
  glowGreen: 'rgba(0, 255, 0, 0.8)',
};

/**
 * LPMUD COLOR CODES
 * Parse $HI*$ style color codes from text MUDs
 */
export const LPMUD_CODES = {
  '$HIY$': { color: NEON_COLORS.hiYellow, glow: NEON_COLORS.glowYellow },
  '$HIW$': { color: NEON_COLORS.hiWhite, glow: NEON_COLORS.glowWhite },
  '$HIC$': { color: NEON_COLORS.hiCyan, glow: NEON_COLORS.glowCyan },
  '$HIM$': { color: NEON_COLORS.hiMagenta, glow: NEON_COLORS.glowMagenta },
  '$HIG$': { color: NEON_COLORS.hiGreen, glow: NEON_COLORS.glowGreen },
  '$HIR$': { color: NEON_COLORS.hiRed, glow: 'rgba(255, 0, 0, 0.8)' },
  '$HIB$': { color: NEON_COLORS.hiBlue, glow: 'rgba(0, 0, 255, 0.8)' },

  // Normal intensity
  '$YEL$': { color: NEON_COLORS.dimYellow, glow: null },
  '$WHT$': { color: NEON_COLORS.dimWhite, glow: null },
  '$CYN$': { color: NEON_COLORS.dimCyan, glow: null },
  '$MAG$': { color: NEON_COLORS.dimMagenta, glow: null },
  '$GRN$': { color: NEON_COLORS.dimGreen, glow: null },
  '$RED$': { color: NEON_COLORS.dimRed, glow: null },

  // Reset
  '$NOR$': { color: NEON_COLORS.dimWhite, glow: null },
  '$RESET$': { color: NEON_COLORS.dimWhite, glow: null },
};

/**
 * Parse LPMUD color codes from a string
 * Returns array of text segments with color info
 */
export function parseLPMUDColors(text) {
  const segments = [];
  let currentIndex = 0;
  let currentColor = NEON_COLORS.dimWhite;
  let currentGlow = null;

  // Regex to match $CODE$ patterns
  const codeRegex = /\$[A-Z]+\$/g;
  let match;

  while ((match = codeRegex.exec(text)) !== null) {
    // Add text before this code
    if (match.index > currentIndex) {
      segments.push({
        text: text.substring(currentIndex, match.index),
        color: currentColor,
        glow: currentGlow
      });
    }

    // Update current color based on code
    const code = match[0];
    if (LPMUD_CODES[code]) {
      currentColor = LPMUD_CODES[code].color;
      currentGlow = LPMUD_CODES[code].glow;
    }

    currentIndex = match.index + code.length;
  }

  // Add remaining text
  if (currentIndex < text.length) {
    segments.push({
      text: text.substring(currentIndex),
      color: currentColor,
      glow: currentGlow
    });
  }

  return segments;
}

/**
 * Cyberpunk glitch effect colors
 */
export const GLITCH_COLORS = [
  NEON_COLORS.cyan,
  NEON_COLORS.magenta,
  NEON_COLORS.yellow,
  NEON_COLORS.hiWhite,
];

/**
 * Matrix rain characters
 */
export const MATRIX_CHARS = [
  '0', '1', 'ｱ', 'ｲ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｸ', 'ｹ', 'ｺ',
  'ｻ', 'ｼ', 'ｽ', 'ｾ', 'ｿ', 'ﾀ', 'ﾁ', 'ﾂ', 'ﾃ', 'ﾄ', 'ﾅ', 'ﾆ',
  'ﾇ', 'ﾈ', 'ﾉ', 'ﾊ', 'ﾋ', 'ﾌ', 'ﾍ', 'ﾎ', 'ﾏ', 'ﾐ', 'ﾑ', 'ﾒ',
  'Z', 'X', 'C', 'V', 'B', 'N', 'M', ':', '.', '"', '=', '*',
  '+', '-', '<', '>', '¦', '|', '╌', '╍', '╎', '╏'
];

export default NEON_COLORS;
