/** Parse nested lists like [[1,2],[3,4]] (JSON-compatible). */
export function parseBracketTensor(source: string): unknown {
  const t = source.trim();
  if (!t) throw new Error("Tensor text is empty");
  try {
    return JSON.parse(t) as unknown;
  } catch {
    throw new Error("Invalid bracket tensor: use JSON-style nested arrays, e.g. [[1,2],[3,4]]");
  }
}

export function nestedTensorShape(a: unknown): number[] {
  if (typeof a === "number") {
    if (!Number.isFinite(a)) throw new Error("Tensor contains non-finite number");
    return [];
  }
  if (!Array.isArray(a)) throw new Error("Tensor must be nested numbers in [...]");
  if (a.length === 0) throw new Error("Tensor has an empty dimension (use at least one element per axis)");
  const first = a[0];
  if (typeof first === "number") {
    if (!a.every((x) => typeof x === "number" && Number.isFinite(x))) {
      throw new Error("Tensor row must be all numbers");
    }
    return [a.length];
  }
  const inner = nestedTensorShape(first);
  for (let i = 1; i < a.length; i++) {
    const sh = nestedTensorShape(a[i]);
    if (sh.length !== inner.length || sh.some((v, j) => v !== inner[j])) {
      throw new Error("Ragged nested list: rows must have the same shape");
    }
  }
  return [a.length, ...inner];
}

export function shapeEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/** Row-major (C) / PyTorch default: last index varies fastest. */
export function flattenRowMajor(a: unknown): number[] {
  if (typeof a === "number") return [a];
  if (!Array.isArray(a)) throw new Error("Expected nested array");
  return a.flatMap(flattenRowMajor);
}

export function parseTensorPayload(
  text: string,
  expectedShape: number[],
): number[] | null {
  const t = text.trim();
  if (!t) return null;
  const parsed = parseBracketTensor(t);
  const sh = nestedTensorShape(parsed);
  if (!shapeEqual(sh, expectedShape)) {
    throw new Error(
      `Tensor shape [${sh.join(", ")}] does not match selected shape [${expectedShape.join(", ")}]`,
    );
  }
  return flattenRowMajor(parsed);
}
