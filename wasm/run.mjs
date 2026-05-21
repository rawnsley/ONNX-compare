// Platform runner: onnxruntime-web (WASM EP, running under Node).
// WASM bytecode is host-independent, so the output should match what a
// browser running the same package version would produce.

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

await report("wasm", "onnxruntime-web", ortVersion, async (input, modelPath) => {
    // Read the model bytes ourselves — onnxruntime-web's session-from-path
    // helper is browser-only, so under Node we pass it the buffer directly.
    const modelBytes = readFileSync(modelPath)
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] })
    const tensor = new ort.Tensor("float32", input, [1, input.length])
    const result = await session.run({ input_values: tensor })
    const t = result["logits"]
    return { logits: t.data, dims: t.dims }
})
