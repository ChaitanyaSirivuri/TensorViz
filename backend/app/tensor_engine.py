"""Apply PyTorch ops and build before/after element layouts (no NN / graph logic)."""

from __future__ import annotations

import ast
import re
from typing import Any

import torch

from app.layout import tensor_with_identity


class TensorVizError(ValueError):
    pass


def _prod(t: tuple[int, ...]) -> int:
    p = 1
    for x in t:
        p *= x
    return p


def _tensor_from_values(shape: tuple[int, ...], values: Any | None) -> torch.Tensor:
    if values is None:
        return tensor_with_identity(shape)
    if not isinstance(values, list):
        raise TensorVizError("values must be a list of numbers")
    n = _prod(shape)
    if len(values) != n:
        raise TensorVizError(f"values length {len(values)} must equal product of shape ({n})")
    return torch.tensor([float(x) for x in values], dtype=torch.float32).reshape(shape)


def _arange_ids(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    n = _prod(shape)
    return torch.arange(n, dtype=torch.int64, device=device).reshape(shape)


def _ensure_shape(shape: list[int]) -> tuple[int, ...]:
    if not shape or any(d < 1 for d in shape):
        raise TensorVizError("Shape must be non-empty with positive dimensions.")
    if len(shape) > 5:
        raise TensorVizError("Visualization supports tensors up to rank 5.")
    return tuple(shape)


def _pair_elements(values: torch.Tensor, ids: torch.Tensor, shape: tuple[int, ...]) -> list[dict]:
    flat_v = values.reshape(-1)
    flat_i = ids.reshape(-1)
    from app.layout import multi_index_to_position, unravel_index

    out: list[dict] = []
    for linear in range(flat_v.numel()):
        midx = unravel_index(linear, shape)
        pos = multi_index_to_position(midx, shape)
        out.append(
            {
                "id": int(flat_i[linear].item()),
                "linear": linear,
                "multi_index": list(midx),
                "value": float(flat_v[linear].item()),
                "position": list(pos),
            }
        )
    return out


def apply_gui(
    initial_shape: list[int],
    operation: str,
    kwargs: dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    list[tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]],
]:
    """
    Returns (v_before, id_before, shape_before, v_after, id_after, shape_after, bases).
    bases is a list of (values, ids, shape) for each input tensor (two entries for stack).
    """
    shape_b = _ensure_shape(initial_shape)
    op = operation.strip().lower()

    if op == "stack":
        raw = kwargs.get("tensors")
        if raw is not None:
            if not isinstance(raw, list) or len(raw) < 2:
                raise TensorVizError("stack: pass `tensors` with at least two entries (each row-major or null for 0..N-1)")
            vs = [_tensor_from_values(shape_b, t) for t in raw]
        else:
            v1 = _tensor_from_values(shape_b, kwargs.get("values"))
            v2 = _tensor_from_values(shape_b, kwargs.get("values_2"))
            vs = [v1, v2]
        n0 = vs[0].numel()
        device = vs[0].device
        id_list: list[torch.Tensor] = []
        off = 0
        for v in vs:
            if v.numel() != n0:
                raise TensorVizError("stack: all tensors must have the same shape")
            id_list.append(torch.arange(off, off + n0, dtype=torch.int64, device=device).reshape(shape_b))
            off += n0
        stack_dim = int(kwargs.get("dim", 0))
        if stack_dim < 0:
            stack_dim += len(shape_b) + 1
        if stack_dim < 0 or stack_dim > len(shape_b):
            raise TensorVizError("stack: dim out of range")
        v_after = torch.stack(vs, dim=stack_dim)
        id_after = torch.stack(id_list, dim=stack_dim)
        bases = list(zip(vs, id_list, [shape_b] * len(vs)))
        return vs[0], id_list[0], shape_b, v_after, id_after, tuple(v_after.shape), bases

    tl = kwargs.get("tensors")
    if tl is not None:
        if not isinstance(tl, list) or len(tl) != 1:
            raise TensorVizError(f"{op} uses one input tensor; send tensors: [ row_major_or_null ]")
        values = _tensor_from_values(shape_b, tl[0])
    else:
        values = _tensor_from_values(shape_b, kwargs.get("values"))
    ids = _arange_ids(shape_b, values.device)
    bases_one = [(values, ids, shape_b)]

    if op == "reshape":
        new_shape = kwargs.get("new_shape")
        if not new_shape or not isinstance(new_shape, list):
            raise TensorVizError("reshape requires new_shape: list of positive ints")
        new_shape_t = _ensure_shape([int(x) for x in new_shape])
        if values.numel() != _prod(new_shape_t):
            raise TensorVizError("reshape: element count must match")
        v_after = values.reshape(new_shape_t)
        id_after = ids.reshape(new_shape_t)
        return values, ids, shape_b, v_after, id_after, new_shape_t, bases_one

    if op == "transpose":
        d0 = int(kwargs.get("dim0", 0))
        d1 = int(kwargs.get("dim1", 1))
        if d0 == d1 or d0 < 0 or d1 < 0 or d0 >= len(shape_b) or d1 >= len(shape_b):
            raise TensorVizError("transpose: invalid dim0/dim1 for tensor rank")
        v_after = values.transpose(d0, d1)
        id_after = ids.transpose(d0, d1)
        return values, ids, shape_b, v_after, id_after, tuple(v_after.shape), bases_one

    if op == "squeeze":
        dim = kwargs.get("dim")
        if dim is None:
            v_after = values.squeeze()
            id_after = ids.squeeze()
        else:
            d = int(dim)
            if d < -len(shape_b) or d >= len(shape_b):
                raise TensorVizError("squeeze: dim out of range")
            v_after = values.squeeze(d)
            id_after = ids.squeeze(d)
        if v_after.shape == values.shape:
            raise TensorVizError("squeeze: chosen dim is not size 1")
        return values, ids, shape_b, v_after, id_after, tuple(v_after.shape), bases_one

    if op == "unsqueeze":
        dim = int(kwargs["dim"])
        if dim < -(len(shape_b) + 1) or dim > len(shape_b):
            raise TensorVizError("unsqueeze: dim out of range")
        v_after = values.unsqueeze(dim)
        id_after = ids.unsqueeze(dim)
        return values, ids, shape_b, v_after, id_after, tuple(v_after.shape), bases_one

    if op == "slice":
        dim = int(kwargs.get("dim", 0))
        start = int(kwargs.get("start", 0))
        end = int(kwargs.get("end", 1))
        if dim < 0 or dim >= len(shape_b):
            raise TensorVizError("slice: dim out of range")
        if start < 0 or end > shape_b[dim] or start >= end:
            raise TensorVizError("slice: invalid start/end for this shape")
        sl = [slice(None)] * len(shape_b)
        sl[dim] = slice(start, end)
        v_after = values[tuple(sl)]
        id_after = ids[tuple(sl)]
        return values, ids, shape_b, v_after, id_after, tuple(v_after.shape), bases_one

    if op == "index_select":
        dim = int(kwargs.get("dim", 0))
        index = kwargs.get("index")
        if not isinstance(index, list) or not index:
            raise TensorVizError("index_select requires non-empty index: list[int]")
        idx_t = torch.tensor([int(i) for i in index], dtype=torch.long)
        if dim < 0 or dim >= len(shape_b):
            raise TensorVizError("index_select: dim out of range")
        if (idx_t < 0).any() or (idx_t >= shape_b[dim]).any():
            raise TensorVizError("index_select: index out of bounds")
        v_after = values.index_select(dim, idx_t)
        id_after = ids.index_select(dim, idx_t)
        return values, ids, shape_b, v_after, id_after, tuple(v_after.shape), bases_one

    raise TensorVizError(f"Unknown operation: {operation}")


# --- Minimal safe code parser (no eval of arbitrary Python) ---

_ALLOWED_CALLS = frozenset({"reshape", "transpose", "squeeze", "unsqueeze", "stack"})


def apply_code(
    code: str,
    initial_shape: list[int],
    value_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], torch.Tensor, torch.Tensor, tuple[int, ...], str, list[tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]]]:
    vk = dict(value_kwargs or {})
    shape_b = _ensure_shape(initial_shape)
    src = code.strip()
    if not src:
        raise TensorVizError("Empty code")

    # torch.stack([t, t, ...], dim=k) — at least two t
    m_stack = re.match(
        r"^torch\.stack\s*\(\s*\[(.*?)\]\s*,\s*dim\s*=\s*(\d+)\s*\)\s*$",
        src,
        re.I,
    )
    if m_stack:
        inner = m_stack.group(1)
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if len(parts) < 2 or any(p != "t" for p in parts):
            raise TensorVizError("stack code: use torch.stack([t, t, ...], dim=k) with at least two t")
        n_need = len(parts)
        raw_tl = vk.get("tensors")
        if raw_tl is None:
            if n_need != 2:
                raise TensorVizError(
                    f"stack code lists {n_need} tensors: add {n_need} bracket tensors in the UI or use exactly [t, t] with defaults"
                )
        elif not isinstance(raw_tl, list) or len(raw_tl) != n_need:
            raise TensorVizError(f"stack code expects {n_need} tensors in `tensors`; got {len(raw_tl) if isinstance(raw_tl, list) else 0}")
        stack_dim = int(m_stack.group(2))
        v_b, id_b, s_b, v_a, id_a, s_a, bases = apply_gui(list(shape_b), "stack", {**vk, "dim": stack_dim})
        return v_b, id_b, s_b, v_a, id_a, s_a, "stack", bases

    tree = ast.parse(src, mode="eval")

    def err(msg: str) -> TensorVizError:
        return TensorVizError(msg)

    if not isinstance(tree.body, ast.Call):
        raise err("Only a single call like t.reshape(...) is allowed")

    call = tree.body
    if not isinstance(call.func, ast.Attribute):
        raise err("Call must be on t, e.g. t.reshape(...)")

    if not isinstance(call.func.value, ast.Name) or call.func.value.id != "t":
        raise err("Use tensor name t for the initial tensor")

    method = call.func.attr
    if method not in _ALLOWED_CALLS:
        raise err(f"Unsupported method: {method}")

    if method == "reshape":
        new_shape = _parse_reshape_args(call)
        v_b, id_b, s_b, v_a, id_a, s_a, bases = apply_gui(
            list(shape_b), "reshape", {**vk, "new_shape": list(new_shape)}
        )
        return v_b, id_b, s_b, v_a, id_a, s_a, "reshape", bases

    if method == "transpose":
        d0, d1 = _parse_two_ints(call)
        v_b, id_b, s_b, v_a, id_a, s_a, bases = apply_gui(
            list(shape_b), "transpose", {**vk, "dim0": d0, "dim1": d1}
        )
        return v_b, id_b, s_b, v_a, id_a, s_a, "transpose", bases

    if method == "squeeze":
        dim_kw = _parse_optional_dim(call)
        kw = {**vk, **({} if dim_kw is None else {"dim": dim_kw})}
        v_b, id_b, s_b, v_a, id_a, s_a, bases = apply_gui(list(shape_b), "squeeze", kw)
        return v_b, id_b, s_b, v_a, id_a, s_a, "squeeze", bases

    if method == "unsqueeze":
        dim = _parse_dim_kw(call)
        v_b, id_b, s_b, v_a, id_a, s_a, bases = apply_gui(list(shape_b), "unsqueeze", {**vk, "dim": dim})
        return v_b, id_b, s_b, v_a, id_a, s_a, "unsqueeze", bases

    if method == "stack":
        raise err("For stack use: torch.stack([t, t], dim=0)")

    raise err("Unsupported expression")


def _parse_reshape_args(call: ast.Call) -> tuple[int, ...]:
    if len(call.args) == 1 and isinstance(call.args[0], (ast.Tuple, ast.List)):
        elts = call.args[0].elts
        out: list[int] = []
        for e in elts:
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant):
                raise TensorVizError("Negative dimensions not allowed")
            if not isinstance(e, ast.Constant) or not isinstance(e.value, int):
                raise TensorVizError("reshape: only integer literals allowed")
            out.append(int(e.value))
        return tuple(out)
    if call.args and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in call.args):
        return tuple(int(a.value) for a in call.args)
    raise TensorVizError("reshape expects (d1, d2, ...) or a tuple/list of ints")


def _parse_two_ints(call: ast.Call) -> tuple[int, int]:
    d0 = d1 = None
    for kw in call.keywords:
        if kw.arg == "dim0":
            d0 = _const_int(kw.value)
        elif kw.arg == "dim1":
            d1 = _const_int(kw.value)
    if len(call.args) >= 2:
        d0 = _const_int(call.args[0])
        d1 = _const_int(call.args[1])
    if d0 is None or d1 is None:
        raise TensorVizError("transpose needs dim0 and dim1 (positional or keywords)")
    return d0, d1


def _parse_optional_dim(call: ast.Call) -> int | None:
    if call.args:
        return _const_int(call.args[0])
    for kw in call.keywords:
        if kw.arg == "dim":
            return _const_int(kw.value)
    return None


def _parse_dim_kw(call: ast.Call) -> int:
    for kw in call.keywords:
        if kw.arg == "dim":
            return _const_int(kw.value)
    if call.args:
        return _const_int(call.args[0])
    raise TensorVizError("unsqueeze requires dim=")


def _const_int(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    raise TensorVizError("Expected integer literal")
