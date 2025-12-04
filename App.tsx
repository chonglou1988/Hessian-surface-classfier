import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from "@google/genai";
import { 
  ArrowUpTrayIcon, 
  BeakerIcon, 
  DocumentArrowDownIcon, 
  CpuChipIcon,
  ChartBarIcon,
  ArrowPathIcon,
  ScaleIcon,
  XCircleIcon,
  PhotoIcon,
  MagnifyingGlassPlusIcon
} from '@heroicons/react/24/outline';
import TerrainCanvas from './components/TerrainCanvas';
import { processTerrain, generateGeminiPrompt, reconstructTerrainFromParams, compareSurfaces, generateHeatMap } from './services/mathService';
import { TerrainState, SurfaceType, ParametricFile, RenderableSurface, CompactParametricFile, CompactRegion, SurfaceRegion, QuadraticParams, ComparisonResult } from './types';
import { SURFACE_COLORS, SURFACE_TYPE_CODES, CODE_TO_SURFACE_TYPE, QUADRATIC_FUNCTION_DEF, COMPACT_DATA_SCHEMA } from './constants';
// @ts-ignore
import { fromBlob } from 'geotiff';

// --- Serialization Helpers for Compact JSON ---

const isPracticallyZero = (n: number) => Math.abs(n) < 1e-6;

const serializeParametricFile = (file: ParametricFile): CompactParametricFile => {
  const compactRegions: CompactRegion[] = file.regions.map(r => {
    let params: number | [number, number, number, number, number, number];
    const { a, b, c, d, e, f } = r.parameters;
    
    if (isPracticallyZero(a) && isPracticallyZero(b) && isPracticallyZero(c) && isPracticallyZero(d) && isPracticallyZero(e)) {
      params = Number(f.toPrecision(5));
    } else {
      params = [
        Number(a.toPrecision(5)),
        Number(b.toPrecision(5)),
        Number(c.toPrecision(5)),
        Number(d.toPrecision(5)),
        Number(e.toPrecision(5)),
        Number(f.toPrecision(5))
      ];
    }
    
    // Covariance and ZRange
    const cov = r.spatialCovariance || [1, 1, 0];
    const zRange = r.zRange || [-9999, 9999];

    return [
      SURFACE_TYPE_CODES[r.type] || 'U',
      [Number(r.center.x.toFixed(2)), Number(r.center.y.toFixed(2))],
      [r.extent.minX, r.extent.maxX, r.extent.minY, r.extent.maxY],
      params,
      [Number(cov[0].toPrecision(5)), Number(cov[1].toPrecision(5)), Number(cov[2].toPrecision(5))],
      [Number(zRange[0].toPrecision(5)), Number(zRange[1].toPrecision(5))]
    ];
  });

  const statsWithCodes: Record<string, number> = {};
  Object.keys(file.metadata.stats).forEach(k => {
    const type = k as SurfaceType;
    const code = SURFACE_TYPE_CODES[type];
    if (code) statsWithCodes[code] = file.metadata.stats[type];
  });

  return {
    metadata: {
      version: "1.2", // Bumped version for new schema
      originalWidth: file.metadata.originalWidth,
      originalHeight: file.metadata.originalHeight,
      generatedAt: file.metadata.generatedAt,
      totalRegions: file.metadata.totalRegions,
      stats: statsWithCodes,
      functionDefinition: QUADRATIC_FUNCTION_DEF,
      dataStructure: COMPACT_DATA_SCHEMA
    },
    regions: compactRegions
  };
};

const deserializeParametricFile = (compact: CompactParametricFile): ParametricFile => {
  let idCounter = 1;
  const regions: SurfaceRegion[] = compact.regions.map(r => {
    const [code, center, extent, params, cov, zRange] = r;
    const type = CODE_TO_SURFACE_TYPE[code] || SurfaceType.UNKNOWN;
    
    let qParams: QuadraticParams;
    if (typeof params === 'number') {
      qParams = { a: 0, b: 0, c: 0, d: 0, e: 0, f: params };
    } else {
      qParams = {
        a: params[0],
        b: params[1],
        c: params[2],
        d: params[3],
        e: params[4],
        f: params[5]
      };
    }

    return {
      id: idCounter++,
      type,
      center: { x: center[0], y: center[1] },
      extent: { minX: extent[0], maxX: extent[1], minY: extent[2], maxY: extent[3] },
      parameters: qParams,
      spatialCovariance: cov ? [cov[0], cov[1], cov[2]] : [1, 1, 0],
      zRange: zRange ? [zRange[0], zRange[1]] : [-99999, 99999]
    };
  });

  const stats: Record<SurfaceType, number> = {
    [SurfaceType.PEAK]: 0,
    [SurfaceType.PIT]: 0,
    [SurfaceType.SADDLE]: 0,
    [SurfaceType.RIDGE]: 0,
    [SurfaceType.VALLEY]: 0,
    [SurfaceType.FLAT]: 0,
    [SurfaceType.UNKNOWN]: 0,
  };

  Object.keys(compact.metadata.stats).forEach(code => {
    const type = CODE_TO_SURFACE_TYPE[code];
    if (type) stats[type] = compact.metadata.stats[code];
  });

  return {
    metadata: {
      originalWidth: compact.metadata.originalWidth,
      originalHeight: compact.metadata.originalHeight,
      generatedAt: compact.metadata.generatedAt,
      totalRegions: compact.metadata.totalRegions,
      stats
    },
    regions
  };
};

// --- Main App ---

const App: React.FC = () => {
  const [terrainState, setTerrainState] = useState<TerrainState>({
    isLoaded: false,
    isProcessing: false,
    renderableData: null,
    parametricFile: null,
    analysisReport: null
  });
  
  const [displayMode, setDisplayMode] = useState<'elevation' | 'classification'>('classification');
  const [exaggeration, setExaggeration] = useState<number>(1.5);
  
  // Compare Mode State
  const [showCompare, setShowCompare] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [comparisonImage, setComparisonImage] = useState<string | null>(null);
  const [expandedImage, setExpandedImage] = useState<string | null>(null);
  const [compareFiles, setCompareFiles] = useState<{dem: File|null, json: File|null}>({dem: null, json: null});
  const [heatmapThreshold, setHeatmapThreshold] = useState<number>(0.0); // Now acts as MINIMUM VISIBLE threshold

  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Helper: Normalize Data ---
  const normalizeTerrainData = (
      data: Float32Array | Float64Array | Int16Array | Uint8Array | number[], 
      width: number, 
      height: number,
      nodataValue: number = -9999
  ): Float32Array => {
      let floatData: Float32Array;
      if (data instanceof Float32Array) {
        floatData = data;
      } else {
        floatData = new Float32Array(data.length);
        for(let i=0; i<data.length; i++) floatData[i] = data[i];
      }

      // NO DOWNSAMPLING
      const processedData = floatData;
      const newW = width;
      const newH = height;

      const out = new Float32Array(newW * newH);
      let min = Infinity;
      let max = -Infinity;
      
      for(let i=0; i<processedData.length; i++) {
          const val = processedData[i];
          if (val === nodataValue || val < -50000 || val > 50000 || isNaN(val)) continue;
          if (val < min) min = val;
          if (val > max) max = val;
      }
      
      if (min === Infinity) { min = 0; max = 1; }
      if (max === min) { max = min + 1; }

      const range = max - min;
      const TARGET_RANGE = 50; 

      for(let i=0; i<out.length; i++) {
          let val = processedData[i];
          if (val === nodataValue || val < -50000 || val > 50000 || isNaN(val)) {
              out[i] = 0;
          } else {
               if (val < min) val = min;
               if (val > max) val = max;
               out[i] = ((val - min) / range) * TARGET_RANGE;
          }
      }
      return out;
  }
  
  const prepareTerrain = (
      rawData: Float32Array | number[] | any, 
      w: number, 
      h: number, 
      nodata?: number
  ) => {
      let floatData: Float32Array;
      if (rawData instanceof Float32Array) floatData = rawData;
      else floatData = Float32Array.from(rawData);

      const normalized = normalizeTerrainData(floatData, w, h, nodata);
      
      return { data: normalized, width: w, height: h };
  }

  // --- File Parsers ---

  const parseTiff = async (file: File): Promise<{ data: Float32Array; width: number; height: number }> => {
    try {
        const tiff = await fromBlob(file);
        const image = await tiff.getImage();
        const width = image.getWidth();
        const height = image.getHeight();
        const rasters = await image.readRasters() as any;
        
        let rawData: Float32Array;
        if (rasters.length > 0 && rasters[0].length === width * height) {
             rawData = rasters[0];
        } else if (rasters.length === width * height) {
             rawData = rasters;
        } else {
            throw new Error("Invalid TIFF raster format");
        }
        return prepareTerrain(rawData, width, height);
    } catch (e) {
        console.error("TIFF parse error:", e);
        throw new Error("Failed to parse TIFF.");
    }
  };

  const parseAsciiGrid = async (file: File): Promise<{ data: Float32Array; width: number; height: number }> => {
    const text = await file.text();
    const lines = text.split('\n');
    let ncols = 0, nrows = 0, nodata = -9999;
    const dataValues: number[] = [];
    
    let headerParsed = false;
    let headerRows = 0;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const parts = line.split(/\s+/);
        if (parts.length < 2) continue;

        const key = parts[0].toLowerCase();
        const val = parseFloat(parts[1]);

        if (key === 'ncols') ncols = val;
        else if (key === 'nrows') nrows = val;
        else if (key === 'nodata_value') nodata = val;
        else if (!isNaN(parseFloat(parts[0]))) {
            headerParsed = true;
            headerRows = i;
            break;
        }
    }

    if (ncols === 0 || nrows === 0) throw new Error("Invalid ASCII Grid header");

    for (let i = headerRows; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const parts = line.split(/\s+/);
        parts.forEach(p => {
             const val = parseFloat(p);
             if (!isNaN(val)) dataValues.push(val);
        });
    }

    return prepareTerrain(dataValues, ncols, nrows, nodata);
  };

  const processImageToHeightMap = (file: File): Promise<{ data: Float32Array; width: number; height: number }> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        const w = img.width;
        const h = img.height;
        
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (!ctx) { reject("Canvas context error"); return; }
        
        ctx.drawImage(img, 0, 0, w, h);
        const imgData = ctx.getImageData(0, 0, w, h);
        const data = imgData.data;
        const heightMap = new Float32Array(w * h);
        
        for (let i = 0; i < data.length; i += 4) {
          const heightVal = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114); 
          heightMap[i / 4] = heightVal; 
        }
        const result = prepareTerrain(heightMap, w, h);
        resolve(result);
      };
      img.onerror = reject;
      img.src = url;
    });
  };

  // --- Comparison Logic ---

  useEffect(() => {
    if (comparisonResult) {
        // Pass heatmapThreshold as the minimum visible value.
        const heatMapUrl = generateHeatMap(
            comparisonResult.diffMap, 
            comparisonResult.width, 
            comparisonResult.height, 
            comparisonResult.maxZDiff,
            heatmapThreshold // This now serves as MIN_VISIBLE
        );
        setComparisonImage(heatMapUrl);
    }
  }, [comparisonResult, heatmapThreshold]);

  const handleCompareFiles = async () => {
    const { dem, json } = compareFiles;
    if (!dem || !json) {
        alert("Please select both files.");
        return;
    }

    try {
        setTerrainState(prev => ({...prev, isProcessing: true}));
        setComparisonResult(null);
        setComparisonImage(null);
        
        // 1. Parse DEM (Normalizes it)
        let demData: Float32Array;
        let w: number, h: number;
        
        if (dem.name.match(/\.(asc|dem|txt)$/i)) {
            const result = await parseAsciiGrid(dem);
            demData = result.data;
            w = result.width; h = result.height;
        } else if (dem.name.match(/\.(tif|tiff)$/i)) {
            const result = await parseTiff(dem);
            demData = result.data;
            w = result.width; h = result.height;
        } else if (dem.type.startsWith('image/')) {
            const result = await processImageToHeightMap(dem);
            demData = result.data;
            w = result.width; h = result.height;
        } else {
            throw new Error("Unsupported DEM format");
        }

        // 2. Parse JSON
        const text = await json.text();
        const jsonObj = JSON.parse(text);
        let parametric: ParametricFile;
        if (jsonObj.metadata && jsonObj.metadata.dataStructure) {
            parametric = deserializeParametricFile(jsonObj as CompactParametricFile);
        } else {
            parametric = jsonObj as ParametricFile;
        }

        // 3. Reconstruct Surface from JSON
        const reconstructed = reconstructTerrainFromParams(parametric);

        // 4. Validate Dimensions
        if (reconstructed.metadata.width !== w || reconstructed.metadata.height !== h) {
            throw new Error(`Dimensions mismatch: DEM is ${w}x${h}, JSON is ${reconstructed.metadata.width}x${reconstructed.metadata.height}`);
        }

        // 5. Calculate Comparison
        const result = compareSurfaces(demData, reconstructed.data.z as Float32Array, w, h);
        setComparisonResult(result);
        setHeatmapThreshold(0.0); // Reset to show all errors initially

        setTerrainState(prev => ({...prev, isProcessing: false}));

    } catch (e: any) {
        console.error(e);
        alert("Comparison Failed: " + e.message);
        setTerrainState(prev => ({...prev, isProcessing: false}));
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    await processFile(file);
  };

  const processFile = async (file: File) => {
    setTerrainState(prev => ({ ...prev, isProcessing: true, analysisReport: null }));

    try {
      if (file.name.endsWith('.json')) {
         const text = await file.text();
         let parametric: ParametricFile;
         
         const json = JSON.parse(text);

         // Check if compact or standard
         if (json.metadata && json.metadata.dataStructure) {
           parametric = deserializeParametricFile(json as CompactParametricFile);
         } else if (json.regions && Array.isArray(json.regions)) {
           parametric = json as ParametricFile;
         } else {
             throw new Error("Invalid JSON format.");
         }

         // Reconstruct the visual surface from the math parameters
         const reconstructed = reconstructTerrainFromParams(parametric);

         setTerrainState(prev => ({
             ...prev,
             isLoaded: true,
             isProcessing: false,
             renderableData: reconstructed,
             parametricFile: parametric
         }));
         return;
      } 
      
      let heightMapData: Float32Array;
      let w: number, h: number;
      let scaleZ = 1.0;

      if (file.name.match(/\.(asc|dem|txt)$/i)) {
         const result = await parseAsciiGrid(file);
         heightMapData = result.data;
         w = result.width;
         h = result.height;
      }
      else if (file.name.match(/\.(tif|tiff)$/i)) {
         const result = await parseTiff(file);
         heightMapData = result.data;
         w = result.width;
         h = result.height;
      }
      else if (file.type.startsWith('image/')) {
        const result = await processImageToHeightMap(file);
        heightMapData = result.data;
        w = result.width;
        h = result.height;
      } else {
        alert("Unsupported format.");
        setTerrainState(prev => ({ ...prev, isProcessing: false }));
        return;
      }

      // 1. Classification & 2. Segmentation/Fitting
      const { renderable, parametric } = await processTerrain(heightMapData, w, h, scaleZ);
      
      setTerrainState(prev => ({
        ...prev,
        isLoaded: true,
        isProcessing: false,
        renderableData: renderable,
        parametricFile: parametric
      }));

    } catch (err) {
      console.error(err);
      alert(`Error processing file: ${err instanceof Error ? err.message : "Unknown error"}`);
      setTerrainState(prev => ({ ...prev, isProcessing: false }));
    }
  };

  const handleGenerateReport = async () => {
    if (!terrainState.parametricFile) return;
    
    if (!process.env.API_KEY) {
        alert("API Key not found.");
        return;
    }

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const prompt = generateGeminiPrompt(terrainState.parametricFile);
        
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt
        });

        setTerrainState(prev => ({...prev, analysisReport: response.text}));
    } catch (error) {
        console.error("Gemini Error:", error);
        alert("Failed to generate AI report.");
    }
  };

  const handleDownload = () => {
    if (!terrainState.parametricFile) return;
    try {
        // Use Compact Serialization
        const compact = serializeParametricFile(terrainState.parametricFile);
        const jsonStr = JSON.stringify(compact, null, 0); // null, 0 removes indentation/whitespace
        
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `surface_model_v1.2_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) {
        console.error("Export Error:", e);
        alert("Failed to export file.");
    }
  };

  const resetApp = () => {
    setTerrainState({
        isLoaded: false,
        isProcessing: false,
        renderableData: null,
        parametricFile: null,
        analysisReport: null
    });
    setComparisonResult(null);
    setComparisonImage(null);
    setShowCompare(false);
    setCompareFiles({dem: null, json: null});
  };

  // --- Render Logic ---

  const renderHome = () => (
    <div className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white p-6 relative overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(#475569 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
        <div className="max-w-4xl w-full z-10 flex flex-col items-center text-center space-y-10">
            <div className="space-y-4">
                <div className="inline-flex items-center justify-center p-4 bg-blue-500/10 rounded-2xl ring-1 ring-blue-500/50 mb-4 shadow-[0_0_20px_rgba(59,130,246,0.3)]">
                    <CpuChipIcon className="w-12 h-12 text-blue-400" />
                </div>
                <h1 className="text-5xl font-extrabold bg-gradient-to-b from-white to-slate-400 bg-clip-text text-transparent">
                    Hessian Surface Classifier
                </h1>
                <p className="text-lg text-slate-400 max-w-2xl mx-auto">
                    A scientific workflow: Input DEM → Compute Hessian → Classify → <span className="text-blue-400 font-bold">Fit Quadratic Functions</span> → Visualize.
                </p>
            </div>
            
            {/* Standard Mode */}
            {!showCompare && (
                <div className="w-full max-w-xl space-y-4">
                    <input 
                        type="file" 
                        accept=".tif,.tiff,.asc,.dem,image/*,.json,.txt" 
                        ref={fileInputRef} 
                        className="hidden" 
                        onChange={handleFileUpload}
                    />
                    <button 
                        onClick={() => fileInputRef.current?.click()}
                        disabled={terrainState.isProcessing}
                        className="group w-full py-6 rounded-2xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 transition-all shadow-xl flex flex-col items-center justify-center gap-3"
                    >
                        {terrainState.isProcessing ? (
                            <div className="flex items-center gap-3">
                                <span className="text-xl font-medium animate-pulse">Computing Surface Functions...</span>
                            </div>
                        ) : (
                            <>
                                <div className="flex items-center gap-2">
                                    <ArrowUpTrayIcon className="w-6 h-6" />
                                    <span className="text-xl font-bold">Select Input / Load JSON Model</span>
                                </div>
                                <span className="text-sm text-blue-200 opacity-80">
                                    Supports DEMs (.tif, .asc) or Parametric JSON models
                                </span>
                            </>
                        )}
                    </button>
                    
                    <button 
                        onClick={() => setShowCompare(true)}
                        className="w-full py-3 bg-transparent border border-slate-700 hover:bg-slate-800 text-slate-400 rounded-lg flex items-center justify-center gap-2 text-sm"
                    >
                        <ScaleIcon className="w-4 h-4" /> Compare Model Accuracy
                    </button>
                </div>
            )}

            {/* Compare Mode */}
            {showCompare && (
                <div className="w-full max-w-5xl bg-slate-900 border border-slate-700 rounded-2xl p-6 relative shadow-2xl">
                    <button onClick={() => {setShowCompare(false); setComparisonResult(null); setComparisonImage(null);}} className="absolute top-4 right-4 text-slate-500 hover:text-white"><XCircleIcon className="w-6 h-6" /></button>
                    <h3 className="text-xl font-bold mb-6 text-slate-200 flex items-center gap-2"><ScaleIcon className="w-5 h-5 text-blue-500" /> Compare Accuracy</h3>
                    
                    <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 text-left">
                                <label className="block text-xs text-slate-500 font-bold uppercase mb-2">1. Original DEM</label>
                                <input 
                                    type="file" 
                                    className="text-xs text-slate-300 file:mr-2 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                                    accept=".asc,.tif,.dem"
                                    onChange={(e) => setCompareFiles(prev => ({...prev, dem: e.target.files?.[0] || null}))}
                                />
                            </div>
                            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 text-left">
                                <label className="block text-xs text-slate-500 font-bold uppercase mb-2">2. Generated JSON</label>
                                <input 
                                    type="file" 
                                    className="text-xs text-slate-300 file:mr-2 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                                    accept=".json"
                                    onChange={(e) => setCompareFiles(prev => ({...prev, json: e.target.files?.[0] || null}))}
                                />
                            </div>
                        </div>

                        {terrainState.isProcessing ? (
                             <div className="py-4 text-blue-400 animate-pulse text-sm font-bold">Calculating differences...</div>
                        ) : (
                             <button 
                                onClick={handleCompareFiles}
                                className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg shadow-lg"
                             >
                                Run Comparison
                             </button>
                        )}

                        {comparisonResult && (
                            <div className="space-y-4">
                                <div className="mt-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700 grid grid-cols-3 gap-2 text-center">
                                    <div>
                                        <div className="text-[10px] uppercase text-slate-500 font-bold">Max Z Diff</div>
                                        <div className="text-lg font-mono text-red-400">{comparisonResult.maxZDiff.toFixed(4)}</div>
                                    </div>
                                    <div>
                                        <div className="text-[10px] uppercase text-slate-500 font-bold">Avg Z Diff</div>
                                        <div className="text-lg font-mono text-blue-400">{comparisonResult.avgZDiff.toFixed(4)}</div>
                                    </div>
                                    <div>
                                        <div className="text-[10px] uppercase text-slate-500 font-bold">Diff Cells</div>
                                        <div className="text-lg font-mono text-yellow-400">
                                            {((comparisonResult.diffCellCount / comparisonResult.totalCells) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {comparisonImage && (
                                    <div className="bg-black p-2 rounded-lg border border-slate-800 flex flex-col items-center">
                                        <div className="flex justify-between w-full mb-2 items-center">
                                            <div className="text-[10px] text-slate-500 uppercase font-bold flex items-center gap-1"><PhotoIcon className="w-3 h-3"/> Difference Heatmap</div>
                                            <div className="flex items-center">
                                                <div className="flex gap-1 items-center text-[10px]">
                                                    <span className="text-blue-500 font-bold">{heatmapThreshold.toFixed(2)}m</span>
                                                    <div className="w-16 h-2 rounded bg-gradient-to-r from-blue-600 via-green-500 to-red-600"></div>
                                                    <span className="text-red-500 font-bold">{comparisonResult.maxZDiff.toFixed(2)}m</span>
                                                </div>
                                                <div className="ml-4 text-[10px] text-slate-400">
                                                    (Errors &lt; {heatmapThreshold.toFixed(2)}m are hidden)
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div className="w-full px-4 py-2 border-b border-slate-800 mb-2">
                                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                                <span>Min Visible Error</span>
                                                <span>{heatmapThreshold.toFixed(2)}m</span>
                                            </div>
                                            <input 
                                                type="range" 
                                                min="0" 
                                                max={Math.ceil(comparisonResult.maxZDiff * 10) / 10} 
                                                step="0.01" 
                                                value={heatmapThreshold} 
                                                onChange={(e) => setHeatmapThreshold(parseFloat(e.target.value))} 
                                                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                            />
                                        </div>

                                        <div 
                                          className="relative w-full group cursor-zoom-in"
                                          onClick={() => setExpandedImage(comparisonImage)}
                                        >
                                          <img 
                                            src={comparisonImage} 
                                            alt="Difference Heatmap" 
                                            className="w-full h-96 object-contain bg-slate-900 rounded border border-slate-700" 
                                          />
                                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
                                             <div className="flex items-center gap-2 text-white font-bold bg-slate-800/80 px-4 py-2 rounded-full">
                                                <MagnifyingGlassPlusIcon className="w-5 h-5"/> Click to Expand
                                             </div>
                                          </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>

        {/* Expanded Image Modal */}
        {expandedImage && (
            <div 
                className="fixed inset-0 z-50 bg-black/95 backdrop-blur-sm flex items-center justify-center p-8 cursor-zoom-out"
                onClick={() => setExpandedImage(null)}
            >
                 <img src={expandedImage} className="max-w-full max-h-full object-contain rounded-lg shadow-2xl border border-slate-800" />
                 <button className="absolute top-6 right-6 text-slate-400 hover:text-white p-2 rounded-full hover:bg-slate-800 transition-colors">
                    <XCircleIcon className="w-10 h-10" />
                 </button>
                 <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-slate-900/80 px-4 py-2 rounded-full border border-slate-700 text-slate-300 text-sm pointer-events-none">
                    Click anywhere to close
                 </div>
            </div>
        )}
    </div>
  );

  const renderViewer = () => (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans overflow-hidden">
      <div className="w-80 flex-shrink-0 border-r border-slate-800 bg-slate-900 flex flex-col z-10 shadow-xl">
        <div className="p-6 border-b border-slate-800 flex justify-between items-center">
            <div>
                <h1 className="text-lg font-bold text-blue-400">Surface Viewer</h1>
                <p className="text-[10px] text-slate-500">
                    {terrainState.parametricFile?.metadata.totalRegions} Regions Fitted
                </p>
            </div>
            <button onClick={resetApp} className="p-2 hover:bg-slate-800 rounded-full text-slate-400"><ArrowPathIcon className="w-4 h-4" /></button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
             {terrainState.renderableData && (
                <div className="space-y-6">
                    <div>
                        <h2 className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-3 flex items-center gap-2">
                            <ChartBarIcon className="w-4 h-4" /> Feature Stats
                        </h2>
                        <div className="space-y-2 text-xs">
                            {(Object.keys(terrainState.renderableData.metadata.stats) as SurfaceType[]).map((type) => {
                                const count = terrainState.renderableData!.metadata.stats[type];
                                const total = terrainState.renderableData!.metadata.width * terrainState.renderableData!.metadata.height;
                                const pct = total > 0 ? (count / total) * 100 : 0;
                                return (
                                    <div key={type} className="flex items-center gap-2">
                                        <div className="w-16 text-slate-400">{type}</div>
                                        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                                            <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: SURFACE_COLORS[type] }}></div>
                                        </div>
                                        <div className="w-8 text-right text-slate-500">{Math.round(pct)}%</div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div className="space-y-4 pt-4 border-t border-slate-800">
                        <div className="flex rounded-md shadow-sm bg-slate-800 p-1">
                            <button onClick={() => setDisplayMode('classification')} className={`flex-1 text-xs py-1.5 rounded transition-all ${displayMode === 'classification' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}>Class</button>
                            <button onClick={() => setDisplayMode('elevation')} className={`flex-1 text-xs py-1.5 rounded transition-all ${displayMode === 'elevation' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}>Elev</button>
                        </div>
                        <div>
                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                <span>Exaggeration</span>
                                <span>{exaggeration.toFixed(1)}x</span>
                            </div>
                            <input type="range" min="0.1" max="5" step="0.1" value={exaggeration} onChange={(e) => setExaggeration(parseFloat(e.target.value))} className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"/>
                        </div>
                    </div>

                    <div className="pt-4 border-t border-slate-800 flex flex-col gap-3">
                        <button onClick={handleGenerateReport} className="w-full py-2.5 px-3 bg-purple-900/30 hover:bg-purple-900/50 border border-purple-500/30 text-purple-200 rounded-md text-xs font-medium flex items-center justify-center gap-2">
                            <CpuChipIcon className="w-4 h-4" /> Analyze with AI
                        </button>
                        <button onClick={handleDownload} className="w-full py-2.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 rounded-md text-xs font-medium flex items-center justify-center gap-2">
                            <DocumentArrowDownIcon className="w-4 h-4" /> Export Math Model (JSON)
                        </button>
                    </div>
                </div>
            )}
        </div>
      </div>

      <div className="flex-1 flex flex-col relative bg-black">
        <div className="flex-1 relative">
             <TerrainCanvas 
                renderableData={terrainState.renderableData} 
                displayMode={displayMode} 
                exaggeration={exaggeration}
            />
        </div>
        {terrainState.analysisReport && (
            <div className="absolute top-6 right-6 w-96 max-h-[85vh] bg-slate-900/95 backdrop-blur-xl border border-purple-500/30 shadow-2xl rounded-2xl overflow-hidden flex flex-col z-50">
                <div className="p-4 bg-purple-900/20 border-b border-purple-500/20 flex justify-between items-center">
                    <h3 className="font-bold text-purple-200 flex items-center gap-2 text-sm"><BeakerIcon className="w-4 h-4" /> Analysis</h3>
                    <button onClick={() => setTerrainState(prev => ({...prev, analysisReport: null}))} className="text-slate-400 hover:text-white">✕</button>
                </div>
                <div className="p-6 overflow-y-auto text-sm text-slate-300 space-y-3 font-light">
                    {terrainState.analysisReport.split('\n').map((line, i) => line.trim() ? <p key={i}>{line}</p> : <br key={i} />)}
                </div>
            </div>
        )}
      </div>
    </div>
  );

  return terrainState.isLoaded ? renderViewer() : renderHome();
};

export default App;