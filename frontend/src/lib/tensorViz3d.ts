/** Matches backend layout cell pitch (center-to-center). */
export const CELL_PITCH = 1;

/** Gap between voxel faces ≈ this fraction of voxel edge length (target ~20–30%). */
export const GAP_RATIO = 0.25;

/** Voxel edge: pitch = edge + gap, gap = GAP_RATIO × edge → edge = pitch / (1 + GAP_RATIO). */
export const VOXEL_SIZE = CELL_PITCH / (1 + GAP_RATIO);

export type Vec3 = [number, number, number];

export function centerOffsetFromPositions(positions: Vec3[]): Vec3 {
  if (!positions.length) return [0, 0, 0];
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of positions) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  return [-(min[0] + max[0]) / 2, -(min[1] + max[1]) / 2, -(min[2] + max[2]) / 2];
}
