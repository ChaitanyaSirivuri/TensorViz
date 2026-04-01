import { Minus, Plus } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { visualize, type VisualizeResponse } from "@/lib/api";
import { parseTensorPayload } from "@/lib/tensorBracket";

const SHAPE_PRESETS: { label: string; value: string }[] = [
  { label: "[6] (1D)", value: "[6]" },
  { label: "[2, 3] (2D)", value: "[2, 3]" },
  { label: "[2, 2, 2] (3D)", value: "[2, 2, 2]" },
  { label: "[2, 2, 2, 2] (4D)", value: "[2, 2, 2, 2]" },
  { label: "[2, 2, 2, 2, 2] (5D)", value: "[2, 2, 2, 2, 2]" },
];

const OPS = [
  { id: "reshape", label: "Reshape" },
  { id: "transpose", label: "Transpose" },
  { id: "squeeze", label: "Squeeze" },
  { id: "unsqueeze", label: "Unsqueeze" },
  { id: "slice", label: "Slice" },
  { id: "index_select", label: "Index (select)" },
  { id: "stack", label: "Stack" },
] as const;

type OpId = (typeof OPS)[number]["id"];

function parseShape(s: string): number[] {
  const m = s.trim().match(/\[(.*?)\]/);
  const inner = m ? m[1] : s;
  return inner
    .split(/[, ]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => parseInt(x, 10));
}

function exampleNestedForShape(shape: number[]): string {
  let v = 0;
  function nest(dims: number[]): string {
    if (dims.length === 0) return String(v++);
    const [d0, ...rest] = dims;
    return `[${Array.from({ length: d0 }, () => nest(rest)).join(", ")}]`;
  }
  try {
    return nest(shape);
  } catch {
    return "[]";
  }
}

function buildTensorsPayload(
  strs: string[],
  shape: number[],
  mode: "gui-unary" | "gui-stack" | "code",
): (number[] | null)[] | undefined {
  const row = strs.map((s) => {
    const t = s.trim();
    if (!t) return null;
    return parseTensorPayload(t, shape);
  });
  // All empty: omit tensors unless GUI stack — backend stack without `tensors` only builds two defaults (values + values_2).
  if (row.every((x) => x === null)) {
    if (mode === "gui-stack" && row.length >= 2) return row;
    return undefined;
  }

  if (mode === "gui-unary") {
    const only = row[0];
    return only !== null ? [only] : undefined;
  }
  if (mode === "gui-stack") {
    if (row.length < 2) throw new Error("Stack needs at least two tensor inputs");
    return row;
  }
  // code: send all slots; backend errors if count wrong for the expression
  if (row.length === 1) return row[0] !== null ? [row[0]] : undefined;
  return row;
}

export function LeftPane({
  onResult,
  onError,
  busy,
  setBusy,
  onReset,
}: {
  onResult: (r: VisualizeResponse) => void;
  onError: (msg: string | null) => void;
  busy: boolean;
  setBusy: (b: boolean) => void;
  onReset: () => void;
}) {
  const [tab, setTab] = useState<"gui" | "code">("gui");
  const [shapeStr, setShapeStr] = useState("[2, 3]");
  const [op, setOp] = useState<OpId>("transpose");
  const [reshapeTarget, setReshapeTarget] = useState("[3, 2]");
  const [dim0, setDim0] = useState("0");
  const [dim1, setDim1] = useState("1");
  const [squeezeDim, setSqueezeDim] = useState("");
  const [unsqueezeDim, setUnsqueezeDim] = useState("0");
  const [sliceDim, setSliceDim] = useState("0");
  const [sliceStart, setSliceStart] = useState("0");
  const [sliceEnd, setSliceEnd] = useState("2");
  const [indexSelectDim, setIndexSelectDim] = useState("0");
  const [indexSelectIndices, setIndexSelectIndices] = useState("0, 2");
  const [stackDim, setStackDim] = useState("0");
  const [code, setCode] = useState("t.transpose(0, 1)");
  const [tensorStrs, setTensorStrs] = useState<string[]>([""]);

  const initialShape = useMemo(() => parseShape(shapeStr), [shapeStr]);
  const examplePlaceholder = useMemo(() => exampleNestedForShape(initialShape), [initialShape]);

  const allowAddRemove = tab === "code" || (tab === "gui" && op === "stack");

  useEffect(() => {
    if (tab !== "gui") return;
    if (op === "stack") {
      setTensorStrs((s) => (s.length < 2 ? [...s, ...Array(2 - s.length).fill("")] : s));
    } else {
      setTensorStrs((s) => [s[0] ?? ""]);
    }
  }, [op, tab]);

  function addTensorSlot() {
    setTensorStrs((s) => [...s, ""]);
  }

  function removeTensorSlot() {
    setTensorStrs((s) => {
      const min = tab === "gui" && op === "stack" ? 2 : 1;
      if (s.length <= min) return s;
      return s.slice(0, -1);
    });
  }

  async function run() {
    onError(null);
    setBusy(true);
    try {
      const mode =
        tab === "code" ? "code" : op === "stack" ? "gui-stack" : "gui-unary";
      let tensors: (number[] | null)[] | undefined;
      try {
        tensors = buildTensorsPayload(tensorStrs, initialShape, mode);
      } catch (e) {
        onError(e instanceof Error ? e.message : "Invalid tensor input");
        return;
      }

      const tensorPayload =
        tensors !== undefined && tensors.length > 0 ? { tensors } : {};

      if (tab === "code") {
        const r = await visualize({
          mode: "code",
          initial_shape: initialShape,
          code: code.trim(),
          ...tensorPayload,
        });
        onResult(r);
        return;
      }

      const kwargs: Record<string, unknown> = {};
      if (op === "reshape") kwargs.new_shape = parseShape(reshapeTarget);
      if (op === "transpose") {
        kwargs.dim0 = parseInt(dim0, 10);
        kwargs.dim1 = parseInt(dim1, 10);
      }
      if (op === "squeeze") {
        if (squeezeDim.trim() !== "") kwargs.dim = parseInt(squeezeDim, 10);
      }
      if (op === "unsqueeze") kwargs.dim = parseInt(unsqueezeDim, 10);
      if (op === "slice") {
        kwargs.dim = parseInt(sliceDim, 10);
        kwargs.start = parseInt(sliceStart, 10);
        kwargs.end = parseInt(sliceEnd, 10);
      }
      if (op === "index_select") {
        kwargs.dim = parseInt(indexSelectDim, 10);
        kwargs.index = indexSelectIndices.split(/[, ]+/).map((x) => parseInt(x.trim(), 10));
      }
      if (op === "stack") kwargs.dim = parseInt(stackDim, 10);

      const r = await visualize({
        mode: "gui",
        initial_shape: initialShape,
        operation: op,
        op_kwargs: kwargs,
        ...tensorPayload,
      });
      onResult(r);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col gap-4 border-r border-border bg-background p-4">
      <div>
        <h1 className="text-lg font-semibold tracking-tight text-foreground">Tensor Operations Visualizer</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          Data-shape transforms only — not layers, weights, or computation graphs.
        </p>
      </div>

      <div className="space-y-2">
        <Label>Initial shape</Label>
        <Select
          value={SHAPE_PRESETS.find((p) => p.value === shapeStr)?.value ?? "custom"}
          onValueChange={(v) => {
            if (v !== "custom") setShapeStr(v);
          }}
        >
          <SelectTrigger>
            <SelectValue placeholder="Pick shape" />
          </SelectTrigger>
          <SelectContent>
            {SHAPE_PRESETS.map((p) => (
              <SelectItem key={p.value} value={p.value}>
                {p.label}
              </SelectItem>
            ))}
            <SelectItem value="custom">Custom (edit below)</SelectItem>
          </SelectContent>
        </Select>
        <input
          className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
          value={shapeStr}
          onChange={(e) => setShapeStr(e.target.value)}
          aria-label="Tensor shape"
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <Label>Input tensors (optional)</Label>
          {allowAddRemove && (
            <div className="flex gap-1">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 px-2"
                onClick={addTensorSlot}
                aria-label="Add tensor input"
              >
                <Plus className="h-3.5 w-3.5" />
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 px-2"
                onClick={removeTensorSlot}
                aria-label="Remove last tensor input"
              >
                <Minus className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </div>
        <p className="text-[10px] leading-snug text-muted-foreground">
          JSON-style nested brackets matching the shape above. Leave empty for default values{" "}
          <code className="text-primary">0 … N−1</code>. Example for current shape:{" "}
          <code className="break-all text-primary">{examplePlaceholder}</code>
        </p>
        {tensorStrs.map((txt, i) => (
          <div key={i} className="space-y-1">
            <span className="text-[10px] text-muted-foreground">
              Tensor {i + 1}
              {tab === "gui" && op === "stack" ? " (stack)" : ""}
            </span>
            <Textarea
              value={txt}
              onChange={(e) => {
                const next = [...tensorStrs];
                next[i] = e.target.value;
                setTensorStrs(next);
              }}
              spellCheck={false}
              className="min-h-[64px] font-mono text-[11px]"
              placeholder={`e.g. ${examplePlaceholder}`}
            />
          </div>
        ))}
        {tab === "gui" && op === "stack" && (
          <p className="text-[10px] text-muted-foreground">
            Stack needs ≥2 tensors (same shape). Use <span className="font-medium">+</span> to add more.
          </p>
        )}
        {tab === "code" && (
          <p className="text-[10px] text-muted-foreground">
            Unary code uses tensor 1 only. <code className="text-primary">torch.stack</code> uses every listed tensor.
          </p>
        )}
      </div>

      <Tabs value={tab} onValueChange={(v) => setTab(v as "gui" | "code")}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="gui">GUI</TabsTrigger>
          <TabsTrigger value="code">Code</TabsTrigger>
        </TabsList>

        <TabsContent value="gui" className="space-y-4">
          <div className="space-y-2">
            <Label>Operation</Label>
            <Select value={op} onValueChange={(v) => setOp(v as OpId)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {OPS.map((o) => (
                  <SelectItem key={o.id} value={o.id}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {op === "reshape" && (
            <div className="space-y-2">
              <Label>New shape</Label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={reshapeTarget}
                onChange={(e) => setReshapeTarget(e.target.value)}
              />
            </div>
          )}

          {op === "transpose" && (
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-2">
                <Label>dim0</Label>
                <input
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={dim0}
                  onChange={(e) => setDim0(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>dim1</Label>
                <input
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={dim1}
                  onChange={(e) => setDim1(e.target.value)}
                />
              </div>
            </div>
          )}

          {op === "squeeze" && (
            <div className="space-y-2">
              <Label>dim (optional)</Label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                placeholder="omit to squeeze all 1s"
                value={squeezeDim}
                onChange={(e) => setSqueezeDim(e.target.value)}
              />
            </div>
          )}

          {op === "unsqueeze" && (
            <div className="space-y-2">
              <Label>dim</Label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={unsqueezeDim}
                onChange={(e) => setUnsqueezeDim(e.target.value)}
              />
            </div>
          )}

          {op === "slice" && (
            <div className="grid grid-cols-3 gap-2">
              <div className="space-y-2">
                <Label>dim</Label>
                <input
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={sliceDim}
                  onChange={(e) => setSliceDim(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>start</Label>
                <input
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={sliceStart}
                  onChange={(e) => setSliceStart(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>end</Label>
                <input
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={sliceEnd}
                  onChange={(e) => setSliceEnd(e.target.value)}
                />
              </div>
            </div>
          )}

          {op === "index_select" && (
            <div className="space-y-2">
              <Label>dim</Label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={indexSelectDim}
                onChange={(e) => setIndexSelectDim(e.target.value)}
              />
              <Label>indices (comma-separated)</Label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={indexSelectIndices}
                onChange={(e) => setIndexSelectIndices(e.target.value)}
              />
            </div>
          )}

          {op === "stack" && (
            <div className="space-y-2">
              <Label>dim</Label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={stackDim}
                onChange={(e) => setStackDim(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Same as <code className="text-primary">torch.stack([…], dim=…)</code> with your input tensors.
              </p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="code" className="space-y-2">
          <Label>Pseudo PyTorch (variable is always t)</Label>
          <Textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck={false}
            className="font-mono text-xs"
            placeholder="t.reshape(3, 2)"
          />
          <p className="text-xs text-muted-foreground">
            Supported: <code className="text-primary">reshape</code>, <code className="text-primary">transpose</code>,{" "}
            <code className="text-primary">squeeze</code>, <code className="text-primary">unsqueeze</code>, and{" "}
            <code className="text-primary">torch.stack([t, t, …], dim=0)</code> (match number of <code>t</code> with
            tensor slots when using custom data).
          </p>
        </TabsContent>
      </Tabs>

      <Button className="w-full" onClick={() => void run()} disabled={busy}>
        {busy ? "Running…" : "Visualize output"}
      </Button>
      <Button type="button" variant="outline" className="w-full" onClick={onReset}>
        Reset canvas
      </Button>
    </div>
  );
}
