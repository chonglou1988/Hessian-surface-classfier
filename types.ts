
export enum SurfaceType {
  PEAK = 'Peak',
  PIT = 'Pit',
  SADDLE = 'Saddle',
  RIDGE = 'Ridge',
  VALLEY = 'Valley',
  FLAT = 'Flat',
  UNKNOWN = 'Unknown'
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface QuadraticParams {
  a: number; // coeff for x^2
  b: number; // coeff for y^2
  c: number; // coeff for xy
  d: number; // coeff for x
  e: number; // coeff for y
  f: number; // constant (intercept)
}

export interface RegionExtent {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface SurfaceRegion {
  id: number;
  type: SurfaceType;
  parameters: QuadraticParams;
  extent: RegionExtent;
  center: { x: number; y: number };
  spatialCovariance: [number, number, number]; // [InvXX, InvYY, InvXY] for Mahalanobis distance
  zRange: [number, number]; // [minZ, maxZ] for clamping
}

// Internal Runtime format
export interface ParametricFile {
  metadata: {
    originalWidth: number;
    originalHeight: number;
    generatedAt: string;
    totalRegions: number;
    stats: Record<SurfaceType, number>;
  };
  regions: SurfaceRegion[];
}

// Compact JSON Export format (Tuple based)
// [type_code, [cx, cy], [minx, maxx, miny, maxy], params, [invXX, invYY, invXY], [minZ, maxZ]]
// Params can be a single number (f) if a=b=c=d=e=0, or Array [a,b,c,d,e,f]
export type CompactRegion = [
  string, 
  [number, number], 
  [number, number, number, number], 
  number | [number, number, number, number, number, number],
  [number, number, number],
  [number, number]
];

export interface CompactParametricFile {
  metadata: {
    version: string;
    originalWidth: number;
    originalHeight: number;
    generatedAt: string;
    totalRegions: number;
    stats: Record<string, number>;
    functionDefinition: string;
    dataStructure: string;
  };
  regions: CompactRegion[];
}

// Runtime state uses the rasterized version for rendering
export interface RenderableSurface {
  metadata: {
    width: number;
    height: number;
    stats: Record<SurfaceType, number>;
  };
  data: {
    z: Float32Array | number[];
    types: SurfaceType[];
  };
}

export interface ComparisonResult {
  width: number;
  height: number;
  maxZDiff: number;
  avgZDiff: number;
  diffCellCount: number;
  totalCells: number;
  diffMap: Float32Array;
}

export interface TerrainState {
  isLoaded: boolean;
  isProcessing: boolean;
  renderableData: RenderableSurface | null;
  parametricFile: ParametricFile | null; // The exportable math model
  analysisReport: string | null;
}