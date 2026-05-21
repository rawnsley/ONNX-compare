#!/usr/bin/env bash
# Build an x86_64 Linux image (under emulation if the host isn't amd64) and
# run onnxruntime-node inside it. The project directory is mounted so the
# container reads shared/input.f32 + cache/model.onnx from the host.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd .. && pwd)"
TAG="onnx-compare-linux-x64"

docker build --platform linux/amd64 -t "$TAG" . >&2

# Mount shared/ and cache/ only — mounting the whole project would shadow
# the container's pre-installed node_modules.
mkdir -p "$PROJECT_ROOT/cache"
docker run --rm \
    --platform linux/amd64 \
    -v "$PROJECT_ROOT/shared:/onnx-compare/shared:ro" \
    -v "$PROJECT_ROOT/cache:/onnx-compare/cache" \
    "$TAG"
