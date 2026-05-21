// Platform runner: onnxruntime-web (WASM EP, running under Node).
// Shares decode/normalise/hash with ../nodejs via ../shared/pipeline.mjs.

import * as ort from "onnxruntime-web/wasm"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, resolve } from "node:path"
import { report } from "../shared/pipeline.mjs"

// `require("onnxruntime-web/package.json")` is blocked by the package's
// exports map, so read it directly off disk instead.
const here = dirname(fileURLToPath(import.meta.url))
const distDir = resolve(here, "node_modules/onnxruntime-web/dist")
const ortVersion = JSON.parse(readFileSync(resolve(here, "node_modules/onnxruntime-web/package.json"), "utf-8")).version

// Default WASM init does `fetch(<wasmUrl>)` which fails under Node. Point the
// loader at the local files instead.
ort.env.wasm.wasmPaths = distDir + "/"

await report("wasm", "onnxruntime-web", ortVersion, async (normalised, modelPath) => {
    // Read the model bytes ourselves — onnxruntime-web's session-from-path
    // helper is browser-only, so under Node we pass it the buffer directly.
    const modelBytes = readFileSync(modelPath)
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] })
    const input = new ort.Tensor("float32", normalised, [1, normalised.length])
    const result = await session.run({ input_values: input })
    const t = result["logits"]
    return { logits: t.data, dims: t.dims }
})
