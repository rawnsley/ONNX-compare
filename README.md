# ONNX-compare

Reproducer for cross-runtime divergence on the same int8-quantised ONNX model.

The same model bytes and the same Float32 input tensor (both verified by
SHA-256) produce slightly different logits across four ONNX Runtime targets
on the same ORT release:

- **native** — `onnxruntime-node` loaded against whatever native binary the
  host platform happens to provide (e.g. `darwin/arm64` on Apple Silicon).
- **wasm** — `onnxruntime-web` (WASM EP), run under Node so it's reproducible
  from CLI; bytecode is host-independent.
- **linux-x64** — `onnxruntime-node` inside a `--platform linux/amd64` Docker
  container, so it always loads the `linux/x64` prebuilt binary.
- **linux-arm64** — `onnxruntime-node` inside a `--platform linux/arm64`
  Docker container, so it always loads the `linux/arm64` prebuilt binary.

On a small number of frames the divergence exceeds the top-1/top-2 logit gap
and the argmax flips, which for an ASR/CTC pipeline turns into a different
output token. Different platforms disagree on different frames.

The input is a 211 KB pre-normalised Float32 PCM buffer ([`shared/input.f32`](shared/input.f32))
baked into the repo, so there's nothing the reviewer has to compute before
ORT sees it. The model is ~350 MB so it's fetched and cached on first run.

## Run

```sh
./run.sh
```

Requires Node 20+, Docker (for the `linux-*` platforms; needs `buildx` and
qemu for cross-arch emulation), and roughly 5 GB free disk for the model
cache and Docker layers. The first run downloads the model (~350 MB,
cached) and builds two Docker images; later runs reuse both.

Output looks like:

```
# ONNX-compare: native / wasm / linux-x64 / linux-arm64

## Per-platform summary

  platform     package                  host            logits hash
  ----------------------------------------------------------------------------
  native       onnxruntime-node 1.26.0  darwin/arm64    sha256:b1c17f85…
  wasm         onnxruntime-web 1.26.0   darwin/arm64    sha256:da32a8f4…
  linux-x64    onnxruntime-node 1.26.0  linux/x64       sha256:db1e2912…
  linux-arm64  onnxruntime-node 1.26.0  linux/arm64     sha256:6197c15a…

Model bytes: ✓ identical across runs
Input bytes: ✓ identical across runs

## Pairwise argmax agreement

  Each cell shows frames where the row's top-1 disagrees with the column's,
  out of 164 CTC frames.

                    native    wasm   linux-x64  linux-arm64
  native                 ·       1           0            1
  wasm                   1       ·           1            2
  linux-x64              0       1           ·            1
  linux-arm64            1       2           1            ·
```

## Adding a platform

Create `<platform>/` with at least a `run.sh` that prints the standard JSON
report to stdout. For an `onnxruntime-node`-based runner the boilerplate is
~10 lines (see [`native/run.mjs`](native/run.mjs) or
[`linux-x64/run.mjs`](linux-x64/run.mjs)). Then add the folder name to
`PLATFORMS` in [`run.sh`](run.sh).

## Replacing the fixture

`shared/input.f32` is a flat little-endian Float32 array — write any tensor
matching `[1, N]` in that shape. `MODEL_URL` env var overrides the model URL;
anything with `input_values: [B, T]` in and `logits: [B, T', V]` out works.

## Caveats

- The `linux-x64` test on an Apple Silicon host runs the x86_64 binary under
  Docker's qemu emulation. The MLAS kernels still execute via emulation, but
  CPU feature detection may surface a different feature set than a real x86
  host would, which could itself influence kernel dispatch. To verify behavior
  on a real Linux host, run `./linux-x64/run.sh` on an x86_64 Linux machine.
- The `linux-arm64` test on an Apple Silicon host is **not** emulated —
  Docker on macOS runs `linux/arm64` containers natively against the host
  ARM64 CPU.
