import type { TensorState } from "@/lib/api";
import { tensorTerminalString } from "@/lib/tensorPrint";

export function TensorTerminalOutput({
  label,
  state,
  darkMode,
}: {
  label: string;
  state: TensorState;
  darkMode: boolean;
}) {
  const text = tensorTerminalString(state.shape, state.elements);
  return (
    <div className="space-y-1">
      <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <pre
        className="max-h-48 overflow-auto rounded-md border border-border p-2 font-mono text-[11px] leading-snug"
        style={{
          background: darkMode ? "rgba(2,6,23,0.92)" : "rgba(248,250,252,0.98)",
          color: darkMode ? "#e2e8f0" : "#0f172a",
        }}
      >
        {text}
      </pre>
    </div>
  );
}
