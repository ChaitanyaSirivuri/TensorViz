import { Html, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { animated, useSpring } from "@react-spring/three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type { VisualizeResponse } from "@/lib/api";
import { getTensorAccentColor } from "@/lib/tensorVizPalette";
import { centerOffsetFromPositions, VOXEL_SIZE, type Vec3 } from "@/lib/tensorViz3d";

export type MergedBlock = {
  id: number;
  value: number;
  from: Vec3;
  to: Vec3;
  multiIndex: number[];
  baseIndex: number;
};

function mergeBlocks(res: VisualizeResponse): MergedBlock[] {
  const elementsPerBase =
    res.bases.length > 1 ? res.bases[0]?.elements.length ?? 0 : 0;
  const multiSource = res.bases.length > 1 && elementsPerBase > 0;

  const beforeMap = new Map<number, { position: Vec3; value: number; multi_index: number[] }>();
  for (const e of res.before.elements) {
    beforeMap.set(e.id, {
      position: e.position as Vec3,
      value: e.value,
      multi_index: e.multi_index,
    });
  }
  const out: MergedBlock[] = [];
  for (const e of res.after.elements) {
    const prev = beforeMap.get(e.id);
    const from = prev?.position ?? (e.position as Vec3);
    const baseIndex = multiSource ? Math.floor(e.id / elementsPerBase) : 0;
    out.push({
      id: e.id,
      value: e.value,
      from,
      to: e.position as Vec3,
      multiIndex: [...e.multi_index],
      baseIndex,
    });
  }
  return out;
}

function addVec(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function centerOffset(blocks: MergedBlock[]): Vec3 {
  if (!blocks.length) return [0, 0, 0];
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const b of blocks) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], b.to[i], b.from[i]);
      max[i] = Math.max(max[i], b.to[i], b.from[i]);
    }
  }
  return [-(min[0] + max[0]) / 2, -(min[1] + max[1]) / 2, -(min[2] + max[2]) / 2];
}

type BBox = { min: Vec3; max: Vec3 };
type CameraView = { position: Vec3; target: Vec3 };
type TickLine = { start: Vec3; end: Vec3 };

function bboxFromPositions(positions: Vec3[]): BBox {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of positions) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  return { min, max };
}

function cameraViewForPositions(positions: Vec3[]): CameraView {
  if (!positions.length) {
    return { position: [8, 6, 10], target: [0, 0, 0] };
  }

  const offset = centerOffsetFromPositions(positions);
  const centered = positions.map((p) => addVec(p, offset));
  const box = bboxFromPositions(centered);
  const spanX = box.max[0] - box.min[0] + VOXEL_SIZE;
  const spanY = box.max[1] - box.min[1] + VOXEL_SIZE;
  const spanZ = box.max[2] - box.min[2] + VOXEL_SIZE;
  const span = Math.max(spanX, spanY, spanZ, 1);

  return {
    position: [span * 1.6 + 0.8, span * 1.2 + 0.6, span * 2.2 + 1.4],
    target: [0, 0, 0],
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpVec(a: Vec3, b: Vec3, t: number): Vec3 {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function pointOnLine(line: TickLine, t: number): Vec3 {
  return lerpVec(line.start, line.end, t);
}

function axisTickLine(rank: number, dim: number, global: BBox, margin: number): TickLine | null {
  const frontZ = global.max[2] + margin * 0.4;
  const leftX = global.min[0] - margin * 1.05;
  const farLeftX = global.min[0] - margin * 2.1;
  const rightX = global.max[0] + margin * 1.0;
  const bottomY = global.min[1] - margin * 1.2;
  const topY = global.max[1] + margin * 1.0;

  if (rank === 1) {
    return {
      start: [global.min[0], bottomY, 0],
      end: [global.max[0], bottomY, 0],
    };
  }

  if (rank === 2) {
    if (dim === 0) {
      return {
        start: [leftX, global.max[1], 0],
        end: [leftX, global.min[1], 0],
      };
    }
    if (dim === 1) {
      return {
        start: [global.min[0], bottomY, 0],
        end: [global.max[0], bottomY, 0],
      };
    }
  }

  if (rank === 3) {
    if (dim === 0) {
      return {
        start: [rightX, bottomY, global.max[2]],
        end: [rightX, bottomY, global.min[2]],
      };
    }
    if (dim === 1) {
      return {
        start: [leftX, global.max[1], frontZ],
        end: [leftX, global.min[1], frontZ],
      };
    }
    if (dim === 2) {
      return {
        start: [global.min[0], bottomY, frontZ],
        end: [global.max[0], bottomY, frontZ],
      };
    }
  }

  if (rank === 4) {
    if (dim === 0) {
      return {
        start: [global.min[0], topY, frontZ],
        end: [global.max[0], topY, frontZ],
      };
    }
    if (dim === 1) {
      return {
        start: [rightX, bottomY, global.max[2]],
        end: [rightX, bottomY, global.min[2]],
      };
    }
    if (dim === 2) {
      return {
        start: [leftX, global.max[1], frontZ],
        end: [leftX, global.min[1], frontZ],
      };
    }
    if (dim === 3) {
      return {
        start: [global.min[0], bottomY, frontZ],
        end: [global.max[0], bottomY, frontZ],
      };
    }
  }

  if (rank === 5) {
    if (dim === 0) {
      return {
        start: [farLeftX, global.max[1], frontZ],
        end: [farLeftX, global.min[1], frontZ],
      };
    }
    if (dim === 1) {
      return {
        start: [global.min[0], topY, frontZ],
        end: [global.max[0], topY, frontZ],
      };
    }
    if (dim === 2) {
      return {
        start: [rightX, bottomY, global.max[2]],
        end: [rightX, bottomY, global.min[2]],
      };
    }
    if (dim === 3) {
      return {
        start: [leftX, global.max[1], frontZ],
        end: [leftX, global.min[1], frontZ],
      };
    }
    if (dim === 4) {
      return {
        start: [global.min[0], bottomY, frontZ],
        end: [global.max[0], bottomY, frontZ],
      };
    }
  }

  return null;
}

/** Axis tick: dimension index d, label k, world position for Html */
function buildAxisTicks(shape: number[], centered: { multiIndex: number[]; pos: Vec3 }[]): { key: string; text: string; pos: Vec3 }[] {
  const rank = shape.length;
  const margin = 0.62;
  const out: { key: string; text: string; pos: Vec3 }[] = [];
  if (!centered.length) return out;

  const global = bboxFromPositions(centered.map((c) => c.pos));

  for (let d = 0; d < rank; d++) {
    const size = shape[d];
    const line = axisTickLine(rank, d, global, margin);
    if (!line) continue;

    for (let k = 0; k < size; k++) {
      const t = size === 1 ? 0.5 : k / (size - 1);
      const pos = pointOnLine(line, t);
      out.push({ key: `d${d}-k${k}`, text: String(k), pos });
    }
  }
  return out;
}

function AxisTicks({
  shape,
  blocks,
  offset,
  darkMode,
}: {
  shape: number[];
  blocks: MergedBlock[];
  offset: Vec3;
  darkMode: boolean;
}) {
  const centered = useMemo(
    () =>
      blocks.map((b) => ({
        multiIndex: b.multiIndex,
        pos: addVec(b.to, offset),
      })),
    [blocks, offset],
  );

  const ticks = useMemo(() => buildAxisTicks(shape, centered), [shape, centered]);

  const fg = darkMode ? "#e0f7fa" : "#0f172a";

  return (
    <group>
      {ticks.map((t) => (
        <Html key={t.key} position={t.pos} center style={{ pointerEvents: "none" }} distanceFactor={8}>
          <div
            className="select-none font-mono text-[12px] font-semibold tabular-nums"
            style={{
              color: fg,
              textShadow: darkMode ? "0 0 6px rgba(0,0,0,0.9)" : "0 0 4px rgba(255,255,255,0.9)",
            }}
          >
            {t.text}
          </div>
        </Html>
      ))}
    </group>
  );
}

function CameraRig({ view, resetNonce }: { view: CameraView; resetNonce: number }) {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    camera.position.set(...view.position);
    camera.lookAt(...view.target);
    controls.target.set(...view.target);
    controls.update();
    if (typeof controls.saveState === "function") {
      controls.saveState();
    }
  }, [view, resetNonce]);

  return (
    <>
      <PerspectiveCamera ref={cameraRef} makeDefault position={view.position} fov={48} />
      <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.08} />
    </>
  );
}

function AnimatedCube({
  from,
  to,
  value,
  offset,
  run,
  darkMode,
  fill,
}: {
  from: Vec3;
  to: Vec3;
  value: number;
  offset: Vec3;
  run: boolean;
  darkMode: boolean;
  fill: string;
}) {
  const f = useMemo(() => from.map((v, i) => v + offset[i]) as Vec3, [from, offset]);
  const t = useMemo(() => to.map((v, i) => v + offset[i]) as Vec3, [to, offset]);

  const { position } = useSpring({
    from: { position: f },
    to: { position: run ? t : f },
    config: { tension: 90, friction: 26, mass: 0.9 },
    immediate: false,
  });

  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const { camera } = useThree();
  const [showValue, setShowValue] = useState(false);

  useFrame(() => {
    if (!meshRef.current) return;
    const d = camera.position.distanceTo(meshRef.current.getWorldPosition(new THREE.Vector3()));
    setShowValue(d < 9 || hovered);
  });

  const stroke = "#020617";

  return (
    <animated.group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={[VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE]} />
        <meshBasicMaterial color={fill} toneMapped={false} />
      </mesh>
      <lineSegments raycast={() => null}>
        <edgesGeometry args={[new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)]} />
        <lineBasicMaterial color={stroke} transparent opacity={0.85} />
      </lineSegments>
      {showValue && (
        <Html distanceFactor={5.5} style={{ pointerEvents: "none" }} center>
          <div
            className="rounded px-1 py-0.5 font-mono text-[9px] font-semibold tabular-nums"
            style={{
              background: darkMode ? "rgba(2,6,23,0.88)" : "rgba(255,255,255,0.92)",
              color: darkMode ? fill : stroke,
              border: `1px solid ${fill}`,
            }}
          >
            {Number.isInteger(value) ? value : value.toFixed(2)}
          </div>
        </Html>
      )}
    </animated.group>
  );
}

function Scene({
  data,
  runId,
  run,
  darkMode,
}: {
  data: VisualizeResponse | null;
  runId: number;
  run: boolean;
  darkMode: boolean;
}) {
  const blocks = useMemo(() => (data ? mergeBlocks(data) : []), [data]);
  const offset = useMemo(() => centerOffset(blocks), [blocks]);
  const shape = data?.after.shape ?? [];

  if (!data || blocks.length === 0) {
    return (
      <mesh>
        <boxGeometry args={[0.05, 0.05, 0.05]} />
        <meshBasicMaterial transparent opacity={0} />
      </mesh>
    );
  }

  return (
    <>
      <AxisTicks shape={shape} blocks={blocks} offset={offset} darkMode={darkMode} />
      <group>
        {blocks.map((b) => (
          <AnimatedCube
            key={`${b.id}-${runId}`}
            from={b.from}
            to={b.to}
            value={b.value}
            offset={offset}
            run={run}
            darkMode={darkMode}
            fill={getTensorAccentColor(b.baseIndex, darkMode)}
          />
        ))}
      </group>
    </>
  );
}

export function TensorCanvas({
  data,
  darkMode = true,
  resetNonce = 0,
}: {
  data: VisualizeResponse | null;
  darkMode?: boolean;
  resetNonce?: number;
}) {
  const [runId, setRunId] = useState(0);
  const [run, setRun] = useState(false);
  const sceneBg = darkMode ? "#0b0f12" : "#f1f5f9";
  const cameraView = useMemo(
    () => cameraViewForPositions(data?.after.elements.map((e) => e.position as Vec3) ?? []),
    [data],
  );

  useEffect(() => {
    if (!data) {
      setRun(false);
      return;
    }
    setRun(false);
    setRunId((k) => k + 1);
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setRun(true));
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, [data]);

  return (
    <div className="h-full min-h-[420px] w-full rounded-lg border border-border bg-muted/30">
      <Canvas dpr={[1, 2]} gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}>
        <CameraRig view={cameraView} resetNonce={resetNonce} />
        <ambientLight intensity={1} />
        <color attach="background" args={[sceneBg]} />
        <Suspense fallback={null}>
          <Scene data={data} runId={runId} run={run} darkMode={darkMode} />
        </Suspense>
      </Canvas>
    </div>
  );
}
