# ONNX-compare

Reproducer for cross-runtime divergence on the same int8-quantised ONNX model.

The same model bytes and the same Float32 input tensor (both verified by
SHA-256) produce slightly different logits in `onnxruntime-node` (native CPU
EP) vs `onnxruntime-web` (WASM EP). On a small number of frames the
divergence exceeds the top-1/top-2 logit gap, so the argmax flips. For an
ASR/CTC pipeline that means a different output token.

The input is a 211 KB pre-normalised Float32 PCM buffer ([`shared/input.f32`](shared/input.f32)) baked
into the repo, so there's nothing the reviewer has to compute before ORT sees
it. The model is ~350 MB so it's fetched and cached on first run.

## Run

```sh
./run.sh
```

That runs every platform under [`./<platform>/`](.) in turn, writes their JSON
reports to `output/`, and prints the diff:

```
# ONNX-compare: nodejs vs wasm

Package:     onnxruntime-node 1.26.0   vs   onnxruntime-web 1.26.0
Model:       sha256:b65c4e4ac2e3...
Input:       52800 f32 samples

## Stage-by-stage hash comparison
  ✓ Model bytes          (match)
  ✓ Input bytes          (match)
  ✗ Logits               (differ)

## Per-frame argmax agreement
  163 / 164 frames agree (99.39%)

## Frames where argmax differs (1)

  frame 19:
    nodejs   id=71 v=8.6393  id=49 v=8.5710  ...
    wasm     id=49 v=8.6061  id=71 v=8.4967  ...
    top-1/top-2 margin:  nodejs 0.0683   wasm 0.1094
```

## Adding a platform

Create `<platform>/` with `package.json` and `run.mjs`. The runner imports the
shared pipeline and yields control to ORT for inference:

```js
import * as ort from "<your-ort-package>"
import { report } from "../shared/pipeline.mjs"
await report("<platform>", "<package>", "<version>", async (input, modelPath) => {
    // ...create session, run, return { logits, dims }
})
```

Then add `<platform>` to `PLATFORMS` in [`run.sh`](run.sh).

## Replacing the fixture

`shared/input.f32` is a flat little-endian Float32 array — write any tensor
matching `[1, N]` in that shape. `MODEL_URL` env var overrides the model URL;
anything with `input_values: [B, T]` in and `logits: [B, T', V]` out works.
