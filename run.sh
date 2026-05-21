#!/usr/bin/env bash
# Runs every platform runner in turn and writes a comparison report to stdout.
# Add a folder here when adding a new platform (must contain package.json + run.mjs
# that follows the same JSON output contract as shared/pipeline.mjs#report).

set -euo pipefail
cd "$(dirname "$0")"

PLATFORMS=(nodejs wasm)

mkdir -p output

for p in "${PLATFORMS[@]}"; do
    echo "=== $p ===" >&2
    (cd "$p" && npm install --silent && node run.mjs > "../output/$p.json")
done

echo >&2
echo "=== compare ===" >&2
node compare.mjs "output/${PLATFORMS[0]}.json" "output/${PLATFORMS[1]}.json" | tee "output/comparison.txt"
