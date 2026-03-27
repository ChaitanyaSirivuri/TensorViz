import type { TensorElement } from "@/lib/api";

function formatNum(n: number): string {
  if (Number.isInteger(n) && Math.abs(n) < 1e15) return `${n}.`;
  return `${n}`;
}

/** PyTorch-style terminal print from row-major flat layout. */
export function tensorTerminalString(shape: number[], elements: TensorElement[]): string {
  const n = shape.reduce((a, b) => a * b, 1);
  const flat = new Array<number>(n);
  for (const e of elements) {
    if (e.linear >= 0 && e.linear < n) flat[e.linear] = e.value;
  }

  function nest(offset: number, dims: number[], depth: number): string {
    if (dims.length === 0) return formatNum(flat[offset] ?? 0);
    const [d0, ...rest] = dims;
    const block = rest.reduce((a, b) => a * b, 1) || 1;
    const rows: string[] = [];
    for (let i = 0; i < d0; i++) {
      rows.push(nest(offset + i * block, rest, depth + 1));
    }
    if (rest.length === 0) {
      return `[${rows.join(", ")}]`;
    }
    const pad = " ".repeat(4 + depth * 4);
    const inner = rows.join(`,\n${pad}`);
    return `[\n${pad}${inner}\n${" ".repeat(depth * 4)}  ]`;
  }

  const inner = nest(0, shape, 0);
  return `tensor(${inner})`;
}
