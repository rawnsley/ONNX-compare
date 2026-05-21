// Platform runner: onnxruntime-node (native CPU EP).
// Shares decode/normalise/hash with ../wasm via ../shared/pipeline.mjs.

import * as ort from "onnxruntime-node"
import { createRequire } from "node:module"
import { report } from "../shared/pipeline.mjs"

const require = createRequire(import.meta.url)
const ortVersion = require("onnxruntime-node/package.json").version

await report("nodejs", "onnxruntime-node", ortVersion, async (normalised, modelPath) => {
    const session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] })
    const input = new ort.Tensor("float32", normalised, [1, normalised.length])
    const result = await session.run({ input_values: input })
    const t = result["logits"]
    return { logits: t.data, dims: t.dims }
})
