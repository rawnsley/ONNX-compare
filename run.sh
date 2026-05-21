#!/usr/bin/env bash
# Runs every platform under ./<name>/ and writes a comparison report. Each
# platform folder is self-contained: it owns its dependencies and how to run
# them (local node, docker, …). Add a folder + add its name here.

set -euo pipefail
cd "$(dirname "$0")"

PLATFORMS=(native wasm linux-x64 linux-arm64)

mkdir -p output

for p in "${PLATFORMS[@]}"; do
    echo "=== $p ===" >&2
    "./$p/run.sh" > "output/$p.json"
done

echo >&2
echo "=== compare ===" >&2
reports=()
for p in "${PLATFORMS[@]}"; do reports+=("output/$p.json"); done
node compare.mjs "${reports[@]}" | tee "output/comparison.txt"
