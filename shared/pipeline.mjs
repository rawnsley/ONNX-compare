// Shared input pipeline: load the pre-normalised PCM buffer that ships with
// the repo, fetch the model (cached), and emit a standard JSON report. The
// two platform runners (nodejs/, wasm/) call into this so the only thing
// that differs between their runs is the ORT package they import.

import { createHash } from "node:crypto"
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))

// Pre-normalised Float32 PCM, baked into the repo so the reviewer has nothing
// to compute before ORT sees the input. Originally derived from a 3.3 s
// 16 kHz mono WAV of an English sentence ("but the more he blew the more
// closely…") processed through the standard wav2vec2 feature extractor
// (zero-mean, unit-variance). Any 16 kHz mono Float32 buffer would do.
const INPUT_PATH = resolve(here, "input.f32")

// Int8-quantised wav2vec2-lv-60-espeak-cv-ft (392-token IPA vocab). Too large
// to commit (~350 MB), so it's fetched and cached on first run. Override with
// MODEL_URL if you want to test a different int8 ONNX model — anything with
// `input_values: [B, T]` in and `logits: [B, T', V]` out will work.
const MODEL_URL = process.env.MODEL_URL
    ?? "https://avnfs.com/tlxOSsLjHCQqCXancqjApUfJNU6j7UYvj24XhCzjwpY?size=356488038&type=application%2Foctet-stream&name=wav2vec2-phonemes-int8.onnx"
const CACHE_DIR = resolve(here, "../cache")

async function loadInputs() {
    mkdirSync(CACHE_DIR, { recursive: true })

    const inputBytes = readFileSync(INPUT_PATH)
    const input = new Float32Array(inputBytes.buffer, inputBytes.byteOffset, inputBytes.byteLength / 4)

    const modelPath = resolve(CACHE_DIR, "model.onnx")
    if (!existsSync(modelPath)) {
        process.stderr.write(`Downloading ${MODEL_URL} → cache/model.onnx ...\n`)
        const r = await fetch(MODEL_URL)
        if (!r.ok) throw new Error(`Model download failed: ${r.status} ${r.statusText}`)
        writeFileSync(modelPath, Buffer.from(await r.arrayBuffer()))
    }

    return {
        modelPath,
        modelHash: sha256(readFileSync(modelPath)),
        input,
        inputHash: sha256(inputBytes),
    }
}

function sha256(buf) {
    return "sha256:" + createHash("sha256").update(buf).digest("hex")
}

function floatBytes(f32) {
    return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength)
}

/** Emit the standard report JSON to stdout. `runInference(input, modelPath)`
 *  returns a flat Float32Array of logits and the [B, T, V] dims. */
export async function report(platform, ortPackage, ortVersion, runInference) {
    const { modelPath, modelHash, input, inputHash } = await loadInputs()
    process.stderr.write(`Running inference on ${platform} (${ortPackage} ${ortVersion}) ...\n`)
    const t0 = performance.now()
    const { logits, dims } = await runInference(input, modelPath)
    const inferenceMs = Math.round(performance.now() - t0)
    const [, numFrames, vocabSize] = dims

    const frames = []
    const buf = new Array(vocabSize)
    for (let f = 0; f < numFrames; f++) {
        const offset = f * vocabSize
        for (let i = 0; i < vocabSize; i++) buf[i] = i
        buf.sort((a, b) => logits[offset + b] - logits[offset + a])
        frames.push({
            f,
            top: buf.slice(0, 5).map(id => ({ id, val: +logits[offset + id].toFixed(4) })),
        })
    }

    const out = {
        platform,
        // Host OS/arch the runner ran on — for native EPs this picks which
        // prebuilt binary the package loaded (e.g. darwin/arm64 vs linux/x64).
        // Doesn't affect WASM EPs, which are runtime-agnostic bytecode.
        host: { os: process.platform, arch: process.arch },
        ortPackage,
        ortVersion,
        modelHash,
        input: {
            samples: input.length,
            hash: inputHash,
        },
        logits: {
            dims,
            hash: sha256(floatBytes(logits)),
        },
        inferenceMs,
        frames,
    }
    process.stdout.write(JSON.stringify(out, null, 2) + "\n")
}
