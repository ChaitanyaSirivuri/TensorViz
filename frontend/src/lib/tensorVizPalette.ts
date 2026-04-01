/** Neon-on-charcoal accents aligned with primary cyan (~#22d3ee / HSL 188). */

const DARK_FILLS = ["#22d3ee", "#e879f9", "#4ade80", "#fbbf24", "#fb7185", "#38bdf8"];

const LIGHT_FILLS = ["#0891b2", "#c026d3", "#16a34a", "#d97706", "#e11d48", "#0284c7"];

export function getTensorAccentColor(tensorIndex: number, darkMode: boolean): string {
  const palette = darkMode ? DARK_FILLS : LIGHT_FILLS;
  return palette[((tensorIndex % palette.length) + palette.length) % palette.length];
}
