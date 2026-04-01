"""Map multi-indices to 3D scene positions for 1D–5D tensors."""

from __future__ import annotations

from typing import Sequence

# Unit spacing between adjacent cells along an axis
CELL = 1.0
# Gap between stacked 3D volumes (4D+) or groups (5D)
VOLUME_GAP = 2.5


def multi_index_to_position(
    indices: Sequence[int],
    shape: Sequence[int],
    *,
    cell: float = CELL,
    volume_gap: float = VOLUME_GAP,
) -> tuple[float, float, float]:
    """Return (x, y, z) for one element. Higher dims become spatial separation.

    Invariant:
    - The last two dims are always shown as a matrix: rows on +Y (row 0 at top),
      columns on +X (col 0 at left).
    - The third-from-last dim becomes front-to-back slices on +Z, with index 0 in front.
    - Rank 4 groups those 3D volumes along +X.
    - Rank 5 groups those 4D blocks along +Y, with index 0 at the visual top.
    """
    if len(shape) == 0:
        return (0.0, 0.0, 0.0)
    if len(shape) == 1:
        i0 = indices[0]
        return (i0 * cell, 0.0, 0.0)
    if len(shape) == 2:
        s0, s1 = shape[0], shape[1]
        i0, i1 = indices[0], indices[1]
        x = i1 * cell
        y = (s0 - 1 - i0) * cell
        return (x, y, 0.0)
    if len(shape) == 3:
        s0, s1, s2 = shape
        i0, i1, i2 = indices[0], indices[1], indices[2]
        x = i2 * cell
        y = (s1 - 1 - i1) * cell
        z = (s0 - 1 - i0) * cell
        return (x, y, z)
    if len(shape) == 4:
        s0, s1, s2, s3 = shape
        i0, i1, i2, i3 = indices[0], indices[1], indices[2], indices[3]
        inner_x = i3 * cell
        inner_y = (s2 - 1 - i2) * cell
        inner_z = (s1 - 1 - i1) * cell
        block_span_x = max(1, s3) * cell + volume_gap
        return (i0 * block_span_x + inner_x, inner_y, inner_z)

    # 5D: stack 4D blocks vertically. The inner 4D layout keeps the same invariant.
    s0 = shape[0]
    i0 = indices[0]
    inner = multi_index_to_position(indices[1:], shape[1:], cell=cell, volume_gap=volume_gap)
    block_span_y = max(1, shape[3]) * cell + volume_gap
    return (inner[0], (s0 - 1 - i0) * block_span_y + inner[1], inner[2])


def unravel_index(linear: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    import torch

    idx = []
    rest = linear
    for dim in reversed(shape):
        idx.append(rest % dim)
        rest //= dim
    return tuple(reversed(idx))


def tensor_with_identity(shape: Sequence[int]):
    """Values 0..N-1 so elements stay identifiable after transforms."""
    import torch

    n = 1
    for d in shape:
        n *= int(d)
    return torch.arange(n, dtype=torch.float32).reshape(tuple(shape))
