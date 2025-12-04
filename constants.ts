

import { SurfaceType } from './types';

export const SURFACE_COLORS: Record<SurfaceType, string> = {
  [SurfaceType.PEAK]: '#ef4444',   // Red-500
  [SurfaceType.PIT]: '#3b82f6',    // Blue-500
  [SurfaceType.SADDLE]: '#eab308', // Yellow-500
  [SurfaceType.RIDGE]: '#f97316',  // Orange-500
  [SurfaceType.VALLEY]: '#06b6d4', // Cyan-500
  [SurfaceType.FLAT]: '#64748b',   // Slate-500
  [SurfaceType.UNKNOWN]: '#1e293b' // Slate-800
};

export const APP_STEPS = [
  { id: 1, title: 'Input', desc: 'Load DEM/Heightmap' },
  { id: 2, title: 'Process', desc: 'Compute Hessian Matrix' },
  { id: 3, title: 'Visualize', desc: '3D Surface Rendering' }
];

export const SURFACE_TYPE_CODES: Record<SurfaceType, string> = {
  [SurfaceType.PEAK]: 'P',
  [SurfaceType.PIT]: 'T',
  [SurfaceType.SADDLE]: 'S',
  [SurfaceType.RIDGE]: 'R',
  [SurfaceType.VALLEY]: 'V',
  [SurfaceType.FLAT]: 'F',
  [SurfaceType.UNKNOWN]: 'U'
};

export const CODE_TO_SURFACE_TYPE: Record<string, SurfaceType> = Object.entries(SURFACE_TYPE_CODES).reduce((acc, [type, code]) => {
  acc[code] = type as SurfaceType;
  return acc;
}, {} as Record<string, SurfaceType>);

// Smaller dimension = Higher Accuracy (better fit for local features)
// Reduced to 4 for maximum fidelity (< 0.01 error target).
export const MAX_REGION_DIMENSION = 4; 

export const QUADRATIC_FUNCTION_DEF = "z = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f; where u = x - center.x, v = y - center.y";
export const COMPACT_DATA_SCHEMA = "[type_code, [cx, cy], [minx, maxx, miny, maxy], params, [invXX, invYY, invXY], [minZ, maxZ]]; params is number (f) or [a..f]";