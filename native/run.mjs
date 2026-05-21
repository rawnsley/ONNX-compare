// Platform runner: onnxruntime-node loaded on the host platform — whatever
// process.platform/arch happens to be at runtime. On a Mac that's darwin/arm64;
// on a Linux server it'd be linux/x64 or linux/arm64 (in which case this run
// should match the corresponding Docker-based linux-*/ runner).

import * as ort from "onnxruntime-node"
import { createRequire } from "node:module"
import { report } from "../shared/pipeline.mjs"

const require = createRequire(import.meta.url)
const ortVersion = require("onnxruntime-node/package.json").version

await report("native", "onnxruntime-node", ortVersion, async (input, modelPath) => {
    const session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] })
    const tensor = new ort.Tensor("float32", input, [1, input.length])
    const result = await session.run({ input_values: tensor })
    const t = result["logits"]
    return { logits: t.data, dims: t.dims }
})
