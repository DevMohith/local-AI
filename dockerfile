# =============================================================================
# DocAI Dockerfile — 3-stage build
# Stage 1: React frontend → static files
# Stage 2: llama.cpp compiled from source
# Stage 3: final image with Python + llama binary + React build
#
# You never install llama.cpp manually — Docker does everything.
# =============================================================================

# ── Stage 1: React frontend ───────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build


# ── Stage 2: llama.cpp compiled from source ───────────────────────────────────
FROM ubuntu:24.04 AS llamacpp-builder
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 --branch b3400 https://github.com/ggerganov/llama.cpp.git

WORKDIR /build/llama.cpp
RUN cmake -B build \
        -DLLAMA_NATIVE=OFF \
        -DLLAMA_AVX=ON \
        -DLLAMA_AVX2=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc) --target llama-server


# ── Stage 3: Final runtime image ──────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    wget ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# llama-server binary
COPY --from=llamacpp-builder \
     /build/llama.cpp/build/bin/llama-server \
     /usr/local/bin/llama-server
RUN chmod +x /usr/local/bin/llama-server

# Python packages
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY backend/ ./backend/

# React build
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Models volume mount point
RUN mkdir -p /models

# Entrypoint
COPY scripts/entrypoint.sh /entrypoint.sh

# WINDOWS FIX: strip \r from line endings so Linux can run the script
# Safe on all operating systems
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["/entrypoint.sh"]