# https://hub.docker.com/r/oven/bun
# Cache-friendly: install deps only when package.json / bun.lock change.
FROM oven/bun:1 AS builder
WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .

# Vite inlines this at build time — set via docker compose build.args (not runtime `environment:`).
ARG VITE_API_URL=http://127.0.0.1:8000
ENV VITE_API_URL=${VITE_API_URL}

RUN bun run build

FROM oven/bun:1-alpine AS runner
WORKDIR /srv

COPY --from=builder /app/dist ./dist

RUN printf '%s\n' '{"private":true}' > package.json \
  && bun add serve@14.2.4

EXPOSE 5173

CMD ["./node_modules/.bin/serve", "dist", "--listen", "tcp://0.0.0.0:5173", "--single"]
