#!/usr/bin/env bash
# Runs onnxruntime-web under Node — same WASM bytecode a browser would use.
set -euo pipefail
cd "$(dirname "$0")"
npm install --silent
node run.mjs
