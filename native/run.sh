#!/usr/bin/env bash
# Runs onnxruntime-node directly on the host — whichever platform that is.
set -euo pipefail
cd "$(dirname "$0")"
npm install --silent
node run.mjs
