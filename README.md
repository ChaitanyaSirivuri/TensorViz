# TensorViz

TensorViz is a small full-stack app for **exploring PyTorch tensor operations visually**. A **FastAPI** backend runs tensor code (GUI-selected ops or user Python snippets) with **PyTorch**, and a **React + Vite** frontend renders the results with **React Three Fiber** so you can see shapes and values change after each operation.

## Repository layout

- **`backend/`** — FastAPI API (`/api/visualize`, `/health`), powered by `uv` and `pyproject.toml` / `uv.lock`.
- **`frontend/`** — Vite + React + TypeScript UI; dependencies are managed with **Bun** (`package.json`, `bun.lock`).

The browser talks to the API using `VITE_API_URL` (see `frontend/src/lib/api.ts`), which defaults to `http://127.0.0.1:8000` for local development.

## Run locally (without Docker)

### Backend

From the repo root:

```powershell
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

### Frontend

In another terminal:

```powershell
cd frontend
bun install
bun run dev
```

Open **http://localhost:5173**. Ensure the backend is running on port **8000** (or set `VITE_API_URL` in `frontend/.env`).

### Production-style frontend build

```powershell
cd frontend
bun install
bun run build
bun run preview
```

## Run with Docker

Build and start both services from the **repository root**:

```powershell
docker compose up --build
```

Then open **http://localhost:5173** for the UI. The API is on **http://localhost:8000** (health check: **http://localhost:8000/health**).

The frontend image bakes in `VITE_API_URL=http://127.0.0.1:8000` so the browser (on your machine) calls the published API port. If you deploy behind another host or port, rebuild the frontend with a build argument, for example:

```powershell
docker compose build frontend --build-arg VITE_API_URL=https://api.example.com
```

### Implementation notes

- **Backend image** uses the **Astral** [`ghcr.io/astral-sh/uv`](https://ghcr.io/astral-sh/uv) Python image, runs `uv sync --frozen`, and starts **uvicorn** via `uv run`. (Python/uv is only used for the API; the UI is built with Bun.) **PyTorch** is resolved from the official **CPU** wheel index (`pyproject.toml` / `uv.lock`) so Docker does not pull CUDA/NVIDIA packages.
- **Frontend image** uses [**oven/bun**](https://hub.docker.com/r/oven/bun): dependency layers copy only `package.json` and `bun.lock` before `bun install`; the app is built with `bun run build`, then static files are served with the `serve` CLI on port 5173.
