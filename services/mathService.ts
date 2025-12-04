import { SurfaceType, ParametricFile, SurfaceRegion, QuadraticParams, RegionExtent, RenderableSurface, ComparisonResult } from '../types';
import { MAX_REGION_DIMENSION } from '../constants';

// --- Matrix Math Helpers for Least Squares ---

/**
 * Solves Ax = b using Gaussian elimination.
 * A is a flattened N*N matrix, b is length N array.
 */
const solveLinearSystem = (A: number[][], b: number[]): number[] | null => {
  const n = b.length;
  // Augment matrix
  const M = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    // Pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) {
        maxRow = k;
      }
    }
    
    // Swap
    [M[i], M[maxRow]] = [M[maxRow], M[i]];

    if (Math.abs(M[i][i]) < 1e-10) return null; // Singular

    // Normalize pivot row
    for (let k = i + 1; k <= n; k++) {
      M[i][k] /= M[i][i];
    }

    // Eliminate other rows
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = M[k][i];
        for (let j = i; j <= n; j++) {
          M[k][j] -= factor * M[i][j];
        }
      }
    }
  }

  const x = new Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = M[i][n];
  }
  return x;
};

/**
 * Fits a quadratic surface z = ax^2 + by^2 + cxy + dx + ey + f
 * Returns coefficients [a, b, c, d, e, f]
 */
const fitQuadraticSurface = (points: {x: number, y: number, z: number}[], centerX: number, centerY: number): QuadraticParams => {
  // We center coordinates relative to the region center to improve numerical stability
  // Equation: z = a*u^2 + b*v^2 + c*uv + d*u + e*v + f
  // where u = x - centerX, v = y - centerY

  const n = points.length;
  if (n < 6) {
    // Not enough points, return flat plane at avg height
    const avgZ = points.reduce((s, p) => s + p.z, 0) / n || 0;
    return { a: 0, b: 0, c: 0, d: 0, e: 0, f: avgZ };
  }

  // Build Linear System M * coeffs = R
  // 6 unknowns: a, b, c, d, e, f
  const M = Array(6).fill(0).map(() => Array(6).fill(0));
  const R = Array(6).fill(0);

  for (const p of points) {
    const u = p.x - centerX;
    const v = p.y - centerY;
    const z = p.z;
    const u2 = u * u;
    const v2 = v * v;
    const uv = u * v;

    // The vector of predictors for this point: [u^2, v^2, uv, u, v, 1]
    const vec = [u2, v2, uv, u, v, 1];

    for (let i = 0; i < 6; i++) {
      R[i] += vec[i] * z;
      for (let j = 0; j < 6; j++) {
        M[i][j] += vec[i] * vec[j];
      }
    }
  }

  // --- Ridge Regression (Regularization) ---
  // Reduced to 1e-8 to allow tight fitting to data points (minimizing Z-difference).
  const lambda = 1e-8; 
  for(let i=0; i<6; i++) M[i][i] += lambda;

  const result = solveLinearSystem(M, R);

  if (!result) {
    // Fallback if singular
    const avgZ = points.reduce((s, p) => s + p.z, 0) / n;
    return { a: 0, b: 0, c: 0, d: 0, e: 0, f: avgZ };
  }

  return {
    a: result[0],
    b: result[1],
    c: result[2],
    d: result[3],
    e: result[4],
    f: result[5]
  };
};

/**
 * Merges small isolated regions into their neighbors to simplify the mesh and reduce file size.
 */
const refineSurfaceTypes = (typesArr: any[], width: number, height: number) => {
    const numPoints = width * height;
    const visited = new Uint8Array(numPoints);
    const dx = [1, -1, 0, 0];
    const dy = [0, 0, 1, -1];
    
    // Threshold: Regions smaller than this will be merged into neighbors.
    // Reduced to 2 (keeping almost all small details)
    const MERGE_THRESHOLD = 2; 

    for (let i = 0; i < numPoints; i++) {
        if (visited[i]) continue;

        const currentType = typesArr[i];
        const queue = [i];
        visited[i] = 1;
        const componentIndices = [i];

        let head = 0;
        while (head < queue.length) {
            const idx = queue[head++];
            const cx = idx % width;
            const cy = Math.floor(idx / width);

            for (let d = 0; d < 4; d++) {
                const nx = cx + dx[d];
                const ny = cy + dy[d];
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    const nIdx = ny * width + nx;
                    if (!visited[nIdx] && typesArr[nIdx] === currentType) {
                        visited[nIdx] = 1;
                        queue.push(nIdx);
                        componentIndices.push(nIdx);
                    }
                }
            }
        }

        // If component is small, merge it into the dominant neighbor
        if (componentIndices.length < MERGE_THRESHOLD) {
            const neighborCounts: Record<string, number> = {};
            let maxCount = 0;
            let bestType = null;

            for (const cIdx of componentIndices) {
                 const cx = cIdx % width;
                 const cy = Math.floor(cIdx / width);
                 for (let d = 0; d < 4; d++) {
                    const nx = cx + dx[d];
                    const ny = cy + dy[d];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const nIdx = ny * width + nx;
                        if (typesArr[nIdx] !== currentType) {
                            const nType = typesArr[nIdx];
                            neighborCounts[nType] = (neighborCounts[nType] || 0) + 1;
                            if (neighborCounts[nType] > maxCount) {
                                maxCount = neighborCounts[nType];
                                bestType = nType;
                            }
                        }
                    }
                 }
            }

            if (bestType) {
                for (const cIdx of componentIndices) {
                    typesArr[cIdx] = bestType;
                }
            }
        }
    }
};

/**
 * Applies a Gaussian Blur to the heightmap.
 * Kernel size is matched to MAX_REGION_DIMENSION to effectively suppress noise at the tile scale.
 */
const applyGaussianBlur = (data: Float32Array, width: number, height: number): Float32Array => {
  const output = new Float32Array(data.length);
  
  // Fixed 3x3 kernel (minimal smoothing)
  const kernelSize = 3;
  const radius = 1;
  const sigma = 1.0; 
  
  const kernel = new Float32Array(kernelSize * kernelSize);
  let kernelSum = 0;

  for (let y = -radius; y <= radius; y++) {
      for (let x = -radius; x <= radius; x++) {
          const exponent = -(x * x + y * y) / (2 * sigma * sigma);
          const value = Math.exp(exponent) / (2 * Math.PI * sigma * sigma);
          const idx = (y + radius) * kernelSize + (x + radius);
          kernel[idx] = value;
          kernelSum += value;
      }
  }

  // Normalize kernel
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kernelSum;
  
  for(let y=0; y<height; y++) {
    for(let x=0; x<width; x++) {
       let sum = 0;
       // Handle edges by clamping
       for(let ky=-radius; ky<=radius; ky++) {
         for(let kx=-radius; kx<=radius; kx++) {
            const nx = Math.min(width-1, Math.max(0, x+kx));
            const ny = Math.min(height-1, Math.max(0, y+ky));
            const val = data[ny*width + nx];
            const weight = kernel[(ky+radius)*kernelSize + (kx+radius)];
            sum += val * weight;
         }
       }
       output[y*width+x] = sum;
    }
  }
  return output;
};

// --- Main Processing Logic ---

export const processTerrain = async (
  heightMap: Float32Array,
  width: number,
  height: number,
  scaleZ: number = 1.0
): Promise<{ renderable: RenderableSurface, parametric: ParametricFile }> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const numPoints = width * height;
      const zArr = new Float32Array(numPoints);
      const typesArr = new Array(numPoints);
      
      const stats: Record<SurfaceType, number> = {
        [SurfaceType.PEAK]: 0,
        [SurfaceType.PIT]: 0,
        [SurfaceType.SADDLE]: 0,
        [SurfaceType.RIDGE]: 0,
        [SurfaceType.VALLEY]: 0,
        [SurfaceType.FLAT]: 0,
        [SurfaceType.UNKNOWN]: 0,
      };

      // Populate Z Array (Raw)
      for (let i = 0; i < numPoints; i++) {
        zArr[i] = heightMap[i] * scaleZ;
      }

      // Smoothed version for derivatives using large kernel
      const smoothedZ = applyGaussianBlur(zArr, width, height);

      const getZ = (x: number, y: number, source: Float32Array) => {
        const cx = Math.max(0, Math.min(width - 1, x));
        const cy = Math.max(0, Math.min(height - 1, y));
        return source[cy * width + cx];
      };

      // 1. Classification Phase
      const h = 1.0; 
      const eps = 0.002; // Threshold for Zero Curvature
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const z = getZ(x, y, smoothedZ); // Use smoothed Z for derivatives

          // Derivatives (Central Difference)
          const fx = (getZ(x + 1, y, smoothedZ) - getZ(x - 1, y, smoothedZ)) / (2 * h);
          const fy = (getZ(x, y + 1, smoothedZ) - getZ(x, y - 1, smoothedZ)) / (2 * h);
          const fxx = (getZ(x + 1, y, smoothedZ) - 2 * z + getZ(x - 1, y, smoothedZ)) / (h * h);
          const fyy = (getZ(x, y + 1, smoothedZ) - 2 * z + getZ(x, y - 1, smoothedZ)) / (h * h);
          const fxy = (getZ(x + 1, y + 1, smoothedZ) - getZ(x + 1, y - 1, smoothedZ) - getZ(x - 1, y + 1, smoothedZ) + getZ(x - 1, y - 1, smoothedZ)) / (4 * h * h);

          const denom = 1 + fx * fx + fy * fy;
          const denomSq = denom * denom;
          const K = denomSq > 1e-9 ? (fxx * fyy - fxy * fxy) / denomSq : 0;
          const H = denom > 1e-9 ? ((1 + fx * fx) * fyy - 2 * fx * fy * fxy + (1 + fy * fy) * fxx) / (2 * Math.pow(denom, 1.5)) : 0;

          let type = SurfaceType.UNKNOWN;

          if (K > eps) {
            if (H < -eps) type = SurfaceType.PEAK;
            else if (H > eps) type = SurfaceType.PIT;
          } else if (K < -eps) {
            type = SurfaceType.SADDLE;
          } else {
            if (H < -eps) type = SurfaceType.RIDGE;
            else if (H > eps) type = SurfaceType.VALLEY;
            else type = SurfaceType.FLAT;
          }

          typesArr[idx] = type;
        }
      }

      // 1.5 Refinement
      refineSurfaceTypes(typesArr, width, height);

      // Re-calculate Stats
      for (let i = 0; i < numPoints; i++) {
          stats[typesArr[i] as SurfaceType]++;
      }

      // 2. Segmentation & Fitting Phase (With Spatial Subdivision)
      const visited = new Uint8Array(numPoints);
      const regions: SurfaceRegion[] = [];
      let regionIdCounter = 1;

      const dx = [1, -1, 0, 0];
      const dy = [0, 0, 1, -1];

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          if (visited[idx] === 1) continue;

          const currentType = typesArr[idx];
          
          // BFS to find Connected Component
          const queue = [idx];
          visited[idx] = 1;
          
          const componentPoints: {x: number, y: number, z: number}[] = [];
          const compExtent: RegionExtent = { minX: x, maxX: x, minY: y, maxY: y };
          
          let head = 0;
          while(head < queue.length) {
            const currIdx = queue[head++];
            const cx = currIdx % width;
            const cy = Math.floor(currIdx / width);
            const zVal = zArr[currIdx]; // Use ORIGINAL Z for fitting, not smoothed
            
            componentPoints.push({ x: cx, y: cy, z: zVal });
            
            if (cx < compExtent.minX) compExtent.minX = cx;
            if (cx > compExtent.maxX) compExtent.maxX = cx;
            if (cy < compExtent.minY) compExtent.minY = cy;
            if (cy > compExtent.maxY) compExtent.maxY = cy;

            for(let i=0; i<4; i++) {
              const nx = cx + dx[i];
              const ny = cy + dy[i];
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nIdx = ny * width + nx;
                if (visited[nIdx] === 0 && typesArr[nIdx] === currentType) {
                  visited[nIdx] = 1;
                  queue.push(nIdx);
                }
              }
            }
          }

          // Spatial Subdivision: Break large components into MAX_REGION_DIMENSION chunks
          const tiles: Record<string, {x: number, y: number, z: number}[]> = {};
          
          for (const p of componentPoints) {
            const tx = Math.floor((p.x - compExtent.minX) / MAX_REGION_DIMENSION);
            const ty = Math.floor((p.y - compExtent.minY) / MAX_REGION_DIMENSION);
            const key = `${tx}_${ty}`;
            if (!tiles[key]) tiles[key] = [];
            tiles[key].push(p);
          }

          // Process each tile as a separate region
          for (const key in tiles) {
            const tilePoints = tiles[key];
            if (tilePoints.length === 0) continue;

            // Compute Tile Stats
            let minZ = Infinity;
            let maxZ = -Infinity;
            let tMinX = Infinity, tMaxX = -Infinity, tMinY = Infinity, tMaxY = -Infinity;
            let sumX = 0, sumY = 0;

            for(const p of tilePoints) {
                if (p.z < minZ) minZ = p.z;
                if (p.z > maxZ) maxZ = p.z;
                if (p.x < tMinX) tMinX = p.x;
                if (p.x > tMaxX) tMaxX = p.x;
                if (p.y < tMinY) tMinY = p.y;
                if (p.y > tMaxY) tMaxY = p.y;
                sumX += p.x;
                sumY += p.y;
            }

            if (minZ === Infinity) minZ = 0;
            if (maxZ === -Infinity) maxZ = 0;

            const centerX = sumX / tilePoints.length;
            const centerY = sumY / tilePoints.length;

            const params = fitQuadraticSurface(tilePoints, centerX, centerY);

            // Compute Covariance
            let sxx = 0, syy = 0, sxy = 0;
            const N = tilePoints.length;
            if (N > 2) {
                for(const p of tilePoints) {
                    const dx = p.x - centerX;
                    const dy = p.y - centerY;
                    sxx += dx * dx;
                    syy += dy * dy;
                    sxy += dx * dy;
                }
                sxx /= N;
                syy /= N;
                sxy /= N;
            } else {
                sxx = 1; syy = 1; sxy = 0;
            }

            const det = sxx * syy - sxy * sxy;
            let invXX = 1, invYY = 1, invXY = 0;
            if (Math.abs(det) > 1e-6) {
                invXX = syy / det;
                invYY = sxx / det;
                invXY = -sxy / det;
            }

            regions.push({
                id: regionIdCounter++,
                type: currentType,
                center: { x: centerX, y: centerY },
                extent: { minX: tMinX, maxX: tMaxX, minY: tMinY, maxY: tMaxY },
                parameters: params,
                spatialCovariance: [invXX, invYY, invXY],
                zRange: [minZ, maxZ]
            });
          }
        }
      }

      const renderable: RenderableSurface = {
        metadata: { width, height, stats },
        data: { z: zArr, types: typesArr }
      };

      const parametric: ParametricFile = {
        metadata: {
          originalWidth: width,
          originalHeight: height,
          generatedAt: new Date().toISOString(),
          totalRegions: regions.length,
          stats
        },
        regions: regions
      };

      resolve({ renderable, parametric });
    }, 100);
  });
};

/**
 * Reconstructs a visual heightmap using Moving Least Squares (Accumulation Buffer).
 * Blends overlapping local functions to create a smooth surface.
 */
export const reconstructTerrainFromParams = (file: ParametricFile): RenderableSurface => {
  const w = file.metadata.originalWidth;
  const h = file.metadata.originalHeight;
  
  // Accumulation Buffers
  const sumZ = new Float32Array(w * h).fill(0);
  const sumWeights = new Float32Array(w * h).fill(0);
  
  // For Classification: Determine best type by max weight voting
  const bestWeight = new Float32Array(w * h).fill(-1);
  const outTypes = new Array(w * h).fill(SurfaceType.UNKNOWN);

  for (const region of file.regions) {
    const { minX, maxX, minY, maxY } = region.extent;
    const { x: cx, y: cy } = region.center;
    const { a, b, c, d, e, f } = region.parameters;
    const [invXX, invYY, invXY] = region.spatialCovariance || [1, 1, 0];
    const [minZ, maxZ] = region.zRange || [-99999, 99999];

    const widthR = maxX - minX;
    const heightR = maxY - minY;
    
    // Tight padding for small tiles to prevent over-smoothing (blurring)
    const padding = Math.ceil(Math.max(widthR, heightR) * 0.5) + 2;
    
    // Influence Radius squared for weight calculation
    const radiusSq = (widthR/2 + padding) * (widthR/2 + padding) + (heightR/2 + padding)*(heightR/2 + padding);

    const rMinX = Math.max(0, Math.floor(minX - padding));
    const rMaxX = Math.min(w - 1, Math.ceil(maxX + padding));
    const rMinY = Math.max(0, Math.floor(minY - padding));
    const rMaxY = Math.min(h - 1, Math.ceil(maxY + padding));

    for (let y = rMinY; y <= rMaxY; y++) {
      for (let x = rMinX; x <= rMaxX; x++) {
        const u = x - cx;
        const v = y - cy;
        const idx = y * w + x;
        
        // Mahalanobis Distance for Anisotropic Weighting
        const distSq = (u * u * invXX) + (v * v * invYY) + (2 * u * v * invXY);
        
        const euclidDistSq = (x - cx)*(x - cx) + (y - cy)*(y - cy);
        
        if (euclidDistSq < radiusSq) {
             const ratio = euclidDistSq / radiusSq;
             // Quartic kernel for very smooth falloff
             const weight = (1 - ratio) * (1 - ratio); 
             
             let zVal = a*u*u + b*v*v + c*u*v + d*u + e*v + f;
             
             // Clamp to known range to prevent wild spikes at edges of fit
             if (zVal < minZ) zVal = minZ;
             if (zVal > maxZ) zVal = maxZ;

             sumZ[idx] += zVal * weight;
             sumWeights[idx] += weight;

             if (weight > bestWeight[idx]) {
                 bestWeight[idx] = weight;
                 outTypes[idx] = region.type;
             }
        }
      }
    }
  }

  const finalZ = new Float32Array(w * h);
  for(let i=0; i < w*h; i++) {
      if (sumWeights[i] > 0.0001) {
          finalZ[i] = sumZ[i] / sumWeights[i];
      } else {
          finalZ[i] = 0; // Void or background
      }
  }

  return {
    metadata: {
      width: w,
      height: h,
      stats: file.metadata.stats
    },
    data: {
      z: finalZ,
      types: outTypes
    }
  };
};

export const compareSurfaces = (
  original: Float32Array,
  reconstructed: Float32Array,
  width: number,
  height: number
): ComparisonResult => {
  let maxDiff = 0;
  let sumDiff = 0;
  let diffCount = 0;
  const total = width * height;
  const diffMap = new Float32Array(total);

  for(let i=0; i<total; i++) {
    const d = Math.abs(original[i] - reconstructed[i]);
    diffMap[i] = d;
    
    if (d > 1e-4) { // Ignore extremely small float precision noise
        diffCount++;
    }
    if (d > maxDiff) maxDiff = d;
    sumDiff += d;
  }

  return {
    width,
    height,
    maxZDiff: maxDiff,
    avgZDiff: sumDiff / total,
    diffCellCount: diffCount,
    totalCells: total,
    diffMap: diffMap
  };
};

export const generateHeatMap = (data: Float32Array, width: number, height: number, maxVal: number, minVisible: number = 0): string => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return '';

    const imgData = ctx.createImageData(width, height);
    const pixels = imgData.data;

    // The gradient represents the range from [minVisible] to [maxVal].
    // If minVisible is 1.0, then Blue is 1.0. 
    // If maxVal is 5.0, then Red is 5.0.
    const range = maxVal - minVisible;

    for (let i = 0; i < data.length; i++) {
        const val = data[i];

        if (val < minVisible) {
            // Mask out errors below threshold (Transparent)
            pixels[i * 4] = 0;
            pixels[i * 4 + 1] = 0;
            pixels[i * 4 + 2] = 0;
            pixels[i * 4 + 3] = 0;
        } else {
            // Dynamic Gradient: Blue -> Red based on position in [minVisible, maxVal]
            let norm = 0;
            if (range > 1e-6) {
                 norm = (val - minVisible) / range;
            } else {
                 norm = 1.0; // If range is 0 (max=min), show as Red/High
            }
            
            // Clamp norm to 0-1
            norm = Math.min(1, Math.max(0, norm));

            // HSL: 240 (Blue) -> 0 (Red)
            const hue = (1.0 - norm) * 240;
            
            // Simple HSL to RGB conversion helper inline
            const s = 1.0, l = 0.5;
            const c = (1 - Math.abs(2 * l - 1)) * s;
            const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
            const m = l - c / 2;
            let r=0, g=0, b=0;
    
            if (0 <= hue && hue < 60) { r = c; g = x; b = 0; }
            else if (60 <= hue && hue < 120) { r = x; g = c; b = 0; }
            else if (120 <= hue && hue < 180) { r = 0; g = c; b = x; }
            else if (180 <= hue && hue < 240) { r = 0; g = x; b = c; }
            else if (240 <= hue && hue < 300) { r = x; g = 0; b = c; }
            else { r = c; g = 0; b = x; }
    
            pixels[i * 4] = (r + m) * 255;
            pixels[i * 4 + 1] = (g + m) * 255;
            pixels[i * 4 + 2] = (b + m) * 255;
            pixels[i * 4 + 3] = 255; // Alpha Opaque
        }
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL();
};

export const generateGeminiPrompt = (parametricData: ParametricFile): string => {
  const { stats, totalRegions } = parametricData.metadata;
  
  // Pick a few representative regions (e.g., largest extent)
  const sortedRegions = [...parametricData.regions].sort((a, b) => {
        const areaA = (a.extent.maxX - a.extent.minX) * (a.extent.maxY - a.extent.minY);
        const areaB = (b.extent.maxX - b.extent.minX) * (b.extent.maxY - b.extent.minY);
        return areaB - areaA;
  });
  
  const sampleRegions = sortedRegions.slice(0, 5);

  const regionDesc = sampleRegions.map(r => 
    `- A ${r.type} region (approx ${r.extent.maxX - r.extent.minX}x${r.extent.maxY - r.extent.minY} pixels) modeled by quadratic: ${r.parameters.a.toFixed(4)}x²...`
  ).join('\n');

  return `
    Analyze the following topographic surface classification.
    Total Sub-Regions: ${totalRegions}
    Stats: ${JSON.stringify(stats)}
    Sample Local Features:
    ${regionDesc}
    
    1. Describe terrain roughness.
    2. Explain landscape formation implications.
    3. Suggest land use suitability.
  `;
};