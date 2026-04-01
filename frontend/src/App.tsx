import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { BaseTensorPreviews } from "@/components/BaseTensorPreviews";
import { LeftPane } from "@/components/LeftPane";
import { TensorCanvas } from "@/components/TensorCanvas";
import { TensorTerminalOutput } from "@/components/TensorTerminalOutput";
import { Button } from "@/components/ui/button";
import type { VisualizeResponse } from "@/lib/api";

export default function App() {
  const [dark, setDark] = useState(true);
  const [data, setData] = useState<VisualizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [canvasResetNonce, setCanvasResetNonce] = useState(0);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  function clearCanvas() {
    setData(null);
    setError(null);
  }

  function resetCanvasView() {
    setCanvasResetNonce((n) => n + 1);
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex items-center justify-between border-b border-border px-4 py-2">
        <span className="text-xs text-muted-foreground">Tensor spatial layout</span>
        <Button variant="ghost" size="sm" onClick={() => setDark((d) => !d)} aria-label="Toggle theme">
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </header>
      <div className="grid flex-1 grid-cols-1 gap-0 lg:grid-cols-[minmax(280px,380px)_1fr]">
        <LeftPane
          onResult={setData}
          onError={setError}
          busy={busy}
          setBusy={setBusy}
          onClearCanvas={clearCanvas}
          onResetCanvas={resetCanvasView}
        />
        <main className="flex min-h-0 flex-1 flex-col gap-3 p-4">
          {error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}
          {data && (
            <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
              <div className="min-w-0 flex-1 space-y-3">
                <div className="text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">{data.operation}</span>
                  {" · "}
                  before {JSON.stringify(data.before.shape)} → after {JSON.stringify(data.after.shape)}
                </div>
                <div className="grid gap-3 lg:grid-cols-2">
                  <TensorTerminalOutput label="Before (input)" state={data.before} darkMode={dark} />
                  <TensorTerminalOutput label="After (output)" state={data.after} darkMode={dark} />
                </div>
              </div>
              <BaseTensorPreviews bases={data?.bases} darkMode={dark} />
            </div>
          )}
          <div className="min-h-[420px] flex-1">
            <TensorCanvas data={data} darkMode={dark} resetNonce={canvasResetNonce} />
          </div>
        </main>
      </div>
    </div>
  );
}
