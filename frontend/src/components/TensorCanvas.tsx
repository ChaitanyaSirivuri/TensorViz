import { Html, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { animated, useSpring } from "@react-spring/three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type { VisualizeResponse } from "@/lib/api";
import { getTensorAccentColor } from "@/lib/tensorVizPalette";
import { VOXEL_SIZE, type Vec3 } from "@/lib/tensorViz3d";

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

function avg(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

/** Axis tick: dimension index d, label k, world position for Html */
function buildAxisTicks(shape: number[], centered: { multiIndex: number[]; pos: Vec3 }[]): { key: string; text: string; pos: Vec3 }[] {
  const rank = shape.length;
  const margin = 0.62;
  const out: { key: string; text: string; pos: Vec3 }[] = [];
  if (!centered.length) return out;

  const { min } = bboxFromPositions(centered.map((c) => c.pos));

  for (let d = 0; d < rank; d++) {
    const size = shape[d];
    for (let k = 0; k < size; k++) {
      const slice = centered.filter((c) => c.multiIndex[d] === k);
      if (!slice.length) continue;
      const cx = avg(slice.map((s) => s.pos[0]));
      const cy = avg(slice.map((s) => s.pos[1]));
      const cz = avg(slice.map((s) => s.pos[2]));
      const c: Vec3 = [cx, cy, cz];

      let pos: Vec3;
      if (rank === 1) {
        pos = [c[0], min[1] - margin, 0];
      } else if (rank === 2) {
        if (d === 0) pos = [c[0], min[1] - margin, c[2]];
        else pos = [min[0] - margin, c[1], c[2]];
      } else if (rank === 3) {
        if (d === 0) pos = [c[0], min[1] - margin, c[2]];
        else if (d === 1) pos = [min[0] - margin, c[1], c[2]];
        else pos = [c[0], c[1], min[2] - margin];
      } else {
        const dirs: Vec3[] = [
          [0, -1, 0],
          [-1, 0, 0],
          [0, 0, -1],
          [1, 0, 0],
          [0, 1, 0],
        ];
        const dir = dirs[d % dirs.length];
        const bump = margin + (d * 0.04 + k * 0.02) * 0.15;
        pos = [c[0] + dir[0] * bump, c[1] + dir[1] * bump, c[2] + dir[2] * bump];
      }
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
}: {
  data: VisualizeResponse | null;
  darkMode?: boolean;
}) {
  const [runId, setRunId] = useState(0);
  const [run, setRun] = useState(false);
  const sceneBg = darkMode ? "#0b0f12" : "#f1f5f9";

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
        <PerspectiveCamera makeDefault position={[8, 6, 10]} fov={48} />
        <OrbitControls enableDamping dampingFactor={0.08} />
        <ambientLight intensity={1} />
        <color attach="background" args={[sceneBg]} />
        <Suspense fallback={null}>
          <Scene data={data} runId={runId} run={run} darkMode={darkMode} />
        </Suspense>
      </Canvas>
    </div>
  );
}
