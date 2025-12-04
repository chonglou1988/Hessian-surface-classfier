
import React, { useMemo, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { RenderableSurface, SurfaceType } from '../types';
import { SURFACE_COLORS } from '../constants';

declare global {
  namespace JSX {
    interface IntrinsicElements {
      mesh: any;
      bufferGeometry: any;
      bufferAttribute: any;
      meshStandardMaterial: any;
      ambientLight: any;
      directionalLight: any;
      hemisphereLight: any;
      gridHelper: any;
    }
  }
}

interface TerrainMeshProps {
  data: RenderableSurface;
  displayMode: 'elevation' | 'classification';
  exaggeration: number;
}

const TerrainMesh: React.FC<TerrainMeshProps> = ({ data, displayMode, exaggeration }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const geometryRef = useRef<THREE.BufferGeometry>(null);
  const { width, height } = data.metadata;

  const colorMap = useMemo(() => {
    const map = new Map<SurfaceType, THREE.Color>();
    (Object.keys(SURFACE_COLORS) as SurfaceType[]).forEach(type => {
      map.set(type, new THREE.Color(SURFACE_COLORS[type]));
    });
    return map;
  }, []);

  const { positions, colors, indices } = useMemo(() => {
    const numPoints = width * height;
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);
    
    const numTriangles = (width - 1) * (height - 1) * 2;
    const indices = new Uint32Array(numTriangles * 3);
    
    const offsetX = width / 2;
    const offsetY = height / 2;

    const { z, types } = data.data;

    // Generate Vertices and Colors
    for (let i = 0; i < numPoints; i++) {
        const x = i % width;
        const y = Math.floor(i / width);

        // Position
        const px = x - offsetX;
        const py = -(y - offsetY); 
        const pz = z[i] * exaggeration; 
        
        positions[i * 3] = px;
        positions[i * 3 + 1] = py;
        positions[i * 3 + 2] = pz;

        // Color
        let r = 0, g = 0, b = 0;
        if (displayMode === 'classification') {
            const type = types[i];
            const c = colorMap.get(type) || colorMap.get(SurfaceType.UNKNOWN)!;
            r = c.r; g = c.g; b = c.b;
        } else {
            // Elevation Coloring (Blue to White)
            const hVal = z[i] / 10;
            const hue = 0.6;
            const sat = 0.5;
            const lum = Math.min(1, Math.max(0.1, hVal * 0.1 + 0.1));
            
            const tempColor = new THREE.Color().setHSL(hue, sat, lum);
            r = tempColor.r; g = tempColor.g; b = tempColor.b;
        }

        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }

    // Generate Indices
    let indexPtr = 0;
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width - 1; x++) {
        const a = y * width + x;
        const b = y * width + (x + 1);
        const c = (y + 1) * width + x;
        const d = (y + 1) * width + (x + 1);

        // Triangle 1
        indices[indexPtr++] = a;
        indices[indexPtr++] = c;
        indices[indexPtr++] = b;

        // Triangle 2
        indices[indexPtr++] = b;
        indices[indexPtr++] = c;
        indices[indexPtr++] = d;
      }
    }

    return { positions, colors, indices };
  }, [data, displayMode, exaggeration, width, height, colorMap]);

  useEffect(() => {
    if (geometryRef.current) {
        // Force the GPU to update the positions when exaggeration changes
        if (geometryRef.current.attributes.position) {
            geometryRef.current.attributes.position.needsUpdate = true;
        }
        geometryRef.current.computeVertexNormals();
    }
  }, [positions, exaggeration]); 

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]} receiveShadow castShadow>
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          count={positions.length / 3}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          array={colors}
          count={colors.length / 3}
          itemSize={3}
        />
        <bufferAttribute
          attach="index"
          array={indices}
          count={indices.length}
          itemSize={1}
        />
      </bufferGeometry>
      <meshStandardMaterial 
        vertexColors 
        side={THREE.DoubleSide} 
        roughness={0.7}
        metalness={0.2}
        wireframe={false}
      />
    </mesh>
  );
};

interface TerrainCanvasProps {
  renderableData: RenderableSurface | null;
  displayMode: 'elevation' | 'classification';
  exaggeration: number;
}

const TerrainCanvas: React.FC<TerrainCanvasProps> = ({ renderableData, displayMode, exaggeration }) => {
  if (!renderableData) return null;

  return (
    <div className="w-full h-full relative">
        <Canvas shadows gl={{ antialias: true, preserveDrawingBuffer: true }}>
            <PerspectiveCamera makeDefault position={[0, 100, 200]} fov={45} />
            <OrbitControls 
                enableDamping 
                dampingFactor={0.05} 
                maxPolarAngle={Math.PI / 2 - 0.05} 
                minDistance={10}
                maxDistance={800}
            />
            
            <ambientLight intensity={0.2} />
            <directionalLight position={[50, 100, 50]} intensity={1.5} castShadow shadow-mapSize={[1024, 1024]} />
            <hemisphereLight intensity={0.4} groundColor="#000000" />

            <TerrainMesh 
                data={renderableData} 
                displayMode={displayMode}
                exaggeration={exaggeration}
            />
            
            <Environment preset="night" />
            <gridHelper args={[Math.max(renderableData.metadata.width, renderableData.metadata.height) * 2, 40, 0x333333, 0x111111]} position={[0, -0.1, 0]} />
        </Canvas>
        
        <div className="absolute bottom-6 left-6 bg-slate-900/80 backdrop-blur-md p-4 rounded-xl border border-slate-700/50 text-xs shadow-2xl pointer-events-none select-none z-10">
            <h4 className="font-bold mb-3 text-slate-300 uppercase tracking-wider text-[10px]">Surface Classification</h4>
            <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                <div className="flex items-center gap-3"><div className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div> <span className="text-slate-300">Peak</span></div>
                <div className="flex items-center gap-3"><div className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]"></div> <span className="text-slate-300">Pit</span></div>
                <div className="flex items-center gap-3"><div className="w-2.5 h-2.5 rounded-full bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.5)]"></div> <span className="text-slate-300">Saddle</span></div>
                <div className="flex items-center gap-3"><div className="w-2.5 h-2.5 rounded-full bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.5)]"></div> <span className="text-slate-300">Ridge</span></div>
                <div className="flex items-center gap-3"><div className="w-2.5 h-2.5 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.5)]"></div> <span className="text-slate-300">Valley</span></div>
                <div className="flex items-center gap-3"><div className="w-2.5 h-2.5 rounded-full bg-slate-500"></div> <span className="text-slate-400">Flat</span></div>
            </div>
        </div>
    </div>
  );
};

export default TerrainCanvas;
