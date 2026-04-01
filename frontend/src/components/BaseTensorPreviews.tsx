import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";

import { Button } from "@/components/ui/button";
import type { TensorState } from "@/lib/api";
import { getTensorAccentColor } from "@/lib/tensorVizPalette";
import { centerOffsetFromPositions, VOXEL_SIZE, type Vec3 } from "@/lib/tensorViz3d";

function StaticVoxels({
  tensor,
  darkMode,
  tensorIndex,
}: {
  tensor: TensorState;
  darkMode: boolean;
  tensorIndex: number;
}) {
  const offset = useMemo(() => {
    const pos = tensor.elements.map((e) => e.position as Vec3);
    return centerOffsetFromPositions(pos);
  }, [tensor.elements]);

  const fill = getTensorAccentColor(tensorIndex, darkMode);
  const stroke = "#020617";
  const bg = darkMode ? "#0b0f12" : "#f8fafc";

  return (
    <>
      <color attach="background" args={[bg]} />
      <PerspectiveCamera makeDefault position={[5, 4, 6]} fov={42} />
      <OrbitControls enableZoom enablePan={false} minPolarAngle={0.15} maxPolarAngle={Math.PI - 0.2} />
      <ambientLight intensity={1} />
      <group>
        {tensor.elements.map((e) => {
          const p = e.position as Vec3;
          const x = p[0] + offset[0];
          const y = p[1] + offset[1];
          const z = p[2] + offset[2];
          return (
            <group key={e.id} position={[x, y, z]}>
              <mesh>
                <boxGeometry args={[VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE]} />
                <meshBasicMaterial color={fill} toneMapped={false} />
              </mesh>
              <lineSegments raycast={() => null}>
                <edgesGeometry args={[new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)]} />
                <lineBasicMaterial color={stroke} transparent opacity={0.9} />
              </lineSegments>
            </group>
          );
        })}
      </group>
    </>
  );
}

export function BaseTensorPreviews({
  bases,
  darkMode,
}: {
  bases: TensorState[] | undefined;
  darkMode: boolean;
}) {
  const list = bases?.length ? bases : [];
  const [page, setPage] = useState(0);

  useEffect(() => {
    setPage(0);
  }, [bases]);

  const idx = list.length ? Math.min(page, list.length - 1) : 0;
  const current = list[idx];

  if (!current) return null;

  return (
    <div className="pointer-events-auto absolute right-3 top-3 z-20 flex w-[min(220px,40vw)] flex-col gap-1 rounded-lg border border-border bg-background/95 p-2 shadow-md backdrop-blur-sm">
      <div className="flex items-center justify-between gap-1">
        <span className="text-[10px] font-medium text-muted-foreground">Input tensors</span>
        {list.length > 1 && (
          <span className="font-mono text-[10px] tabular-nums text-foreground">
            {idx + 1} / {list.length}
          </span>
        )}
      </div>
      {list.length > 1 && (
        <div className="flex gap-1">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-7 flex-1 px-1"
            onClick={() => setPage((p) => (p - 1 + list.length) % list.length)}
            aria-label="Previous base tensor"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-7 flex-1 px-1"
            onClick={() => setPage((p) => (p + 1) % list.length)}
            aria-label="Next base tensor"
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}
      <div className="text-[10px] text-muted-foreground">
        shape {JSON.stringify(current.shape)}
      </div>
      <div className="h-[130px] w-full overflow-hidden rounded-md border border-border">
        <Canvas
          key={`${idx}-${current.shape.join("x")}`}
          dpr={[1, 2]}
          gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
        >
          <StaticVoxels tensor={current} darkMode={darkMode} tensorIndex={idx} />
        </Canvas>
      </div>
    </div>
  );
}
