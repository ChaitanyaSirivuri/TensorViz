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

    2D/3D: match terminal/PyTorch print layout.
    - dim0 = rows, drawn along +Y with row 0 at the visual top
    - dim1 = columns, drawn along +X with col 0 at the visual left
    - dim2 = depth, drawn along +Z
    4D/5D keep the same embedded orientation while separating outer dims into groups.
    """
    if len(shape) == 0:
        return (0.0, 0.0, 0.0)
    if len(shape) == 1:
        s0 = shape[0]
        i0 = indices[0]
        x = i0 * cell
        return (x, 0.0, 0.0)
    if len(shape) == 2:
        s0, s1 = shape[0], shape[1]
        i0, i1 = indices[0], indices[1]
        x = i1 * cell
        y = (s0 - 1 - i0) * cell
        return (x, y, 0.0)
    if len(shape) == 3:
        s0, s1 = shape[0], shape[1]
        i0, i1, i2 = indices[0], indices[1], indices[2]
        x = i1 * cell
        y = (s0 - 1 - i0) * cell
        z = i2 * cell
        return (x, y, z)
    if len(shape) == 4:
        # d0 volumes along +X; inside each: dim1=rows→Y, dim2=cols→X, dim3→Z.
        s0, s1, s2, s3 = shape
        i0, i1, i2, i3 = indices[0], indices[1], indices[2], indices[3]
        span_x = max(1, s2) * cell
        origin_x = i0 * (span_x + volume_gap)
        return (origin_x + i2 * cell, (s1 - 1 - i1) * cell, i3 * cell)
    # 5D: grid of 4D blocks along +Y (slice0), then 4D layout inside.
    s0, s1, s2, s3, s4 = shape
    i0, i1, i2, i3, i4 = indices[0], indices[1], indices[2], indices[3], indices[4]
    span_x = max(1, s3) * cell
    span_y_block = max(1, s1) * (max(1, s2) * cell + volume_gap * 0.5)
    origin_y = (s0 - 1 - i0) * (span_y_block + volume_gap)
    origin_x = i1 * (span_x + volume_gap)
    return (origin_x + i3 * cell, origin_y + (s2 - 1 - i2) * cell, i4 * cell)


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
