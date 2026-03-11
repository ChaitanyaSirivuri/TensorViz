"""FastAPI entry: CORS for static frontend (e.g. GitHub Pages) + visualize endpoint."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.tensor_engine import TensorVizError, _pair_elements, apply_code, apply_gui

DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app = FastAPI(title="Tensor Viz API", version="0.1.0")

_origins_env = os.getenv("TENSOR_VIZ_CORS_ORIGINS", "")
_extra = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS + _extra,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VisualizeRequest(BaseModel):
    mode: Literal["gui", "code"] = "gui"
    initial_shape: list[int]
    operation: str | None = None
    op_kwargs: dict[str, Any] = Field(default_factory=dict)
    code: str | None = None
    values: list[float] | None = Field(
        default=None,
        description="Row-major flattened values; omit to use 0..N-1.",
    )
    values_2: list[float] | None = Field(
        default=None,
        description="Second tensor for stack; same element count as the first tensor.",
    )
    tensors: list[list[float] | None] | None = Field(
        default=None,
        description="One nested-bracket tensor per input: row-major flat list, or null for default 0..N-1. Stack needs ≥2 entries.",
    )


class TensorState(BaseModel):
    shape: list[int]
    elements: list[dict[str, Any]]


class VisualizeResponse(BaseModel):
    ok: bool = True
    operation: str
    before: TensorState
    after: TensorState
    bases: list[TensorState] = Field(default_factory=list)
    error: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


def _value_kw(body: VisualizeRequest) -> dict[str, Any]:
    pass

@app.post("/api/visualize", response_model=VisualizeResponse)
def visualize(body: VisualizeRequest):
    pass