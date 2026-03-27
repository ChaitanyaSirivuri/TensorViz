const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export type TensorElement = {
  id: number;
  linear: number;
  multi_index: number[];
  value: number;
  position: [number, number, number];
};

export type TensorState = {
  shape: number[];
  elements: TensorElement[];
};

export type VisualizeResponse = {
  ok: boolean;
  operation: string;
  before: TensorState;
  after: TensorState;
  bases: TensorState[];
  error?: string | null;
};

export type VisualizePayload =
  | {
      mode: "gui";
      initial_shape: number[];
      operation: string;
      op_kwargs: Record<string, unknown>;
      /** Each entry: row-major flat list, or null for default 0..N−1. */
      tensors?: (number[] | null)[];
      values?: number[];
      values_2?: number[];
    }
  | {
      mode: "code";
      initial_shape: number[];
      code: string;
      tensors?: (number[] | null)[];
      values?: number[];
      values_2?: number[];
    };

export async function visualize(payload: VisualizePayload): Promise<VisualizeResponse> {
  const res = await fetch(`${API}/api/visualize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = typeof err?.detail === "string" ? err.detail : res.statusText;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}
