// Platform runner: onnxruntime-node running inside a Docker container built
// --platform linux/arm64, so the package loads its linux/arm64 prebuilt binary
// regardless of the host machine's architecture.

import * as ort from "onnxruntime-node"
import { createRequire } from "node:module"
import { report } from "../shared/pipeline.mjs"

const require = createRequire(import.meta.url)
const ortVersion = require("onnxruntime-node/package.json").version

await report("linux-arm64", "onnxruntime-node", ortVersion, async (input, modelPath) => {
    const session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] })
    const tensor = new ort.Tensor("float32", input, [1, input.length])
    const result = await session.run({ input_values: tensor })
    const t = result["logits"]
    return { logits: t.data, dims: t.dims }
})
