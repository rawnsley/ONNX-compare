// Reads two platform reports (the JSON written by shared/pipeline.mjs#report)
// and prints a human-readable diff: which hashes match, how often the per-frame
// argmax agrees, and the exact frames where they disagree.

import { readFileSync } from "node:fs"

const [aPath, bPath] = process.argv.slice(2)
if (!aPath || !bPath) {
    process.stderr.write("Usage: node compare.mjs <a.json> <b.json>\n")
    process.exit(2)
}

const a = JSON.parse(readFileSync(aPath, "utf-8"))
const b = JSON.parse(readFileSync(bPath, "utf-8"))

const lines = []
const log = s => lines.push(s)

log(`# ONNX-compare: ${a.platform} vs ${b.platform}\n`)
log(`Package:     ${a.ortPackage} ${a.ortVersion}   vs   ${b.ortPackage} ${b.ortVersion}`)
log(`Host:        ${a.host.os}/${a.host.arch}            vs   ${b.host.os}/${b.host.arch}`)
log(`Model:       ${a.modelHash}`)
log(`Input:       ${a.input.samples} f32 samples`)
log("")

function row(label, av, bv) {
    const ok = av === bv ? "✓" : "✗"
    log(`  ${ok} ${label.padEnd(20)} ${av === bv ? "(match)" : "(differ)"}`)
    log(`      ${a.platform.padEnd(8)} ${av}`)
    log(`      ${b.platform.padEnd(8)} ${bv}`)
}

log("## Stage-by-stage hash comparison")
row("Model bytes", a.modelHash, b.modelHash)
row("Input bytes", a.input.hash, b.input.hash)
row("Logits",      a.logits.hash, b.logits.hash)
log("")

if (a.logits.dims.join(",") !== b.logits.dims.join(",")) {
    log(`!! Logits shape differs: ${a.logits.dims} vs ${b.logits.dims}`)
    process.stdout.write(lines.join("\n") + "\n")
    process.exit(1)
}

const numFrames = a.frames.length
const diffs = []
for (let i = 0; i < numFrames; i++) {
    const aTop = a.frames[i].top[0]
    const bTop = b.frames[i].top[0]
    if (aTop.id !== bTop.id) diffs.push({ f: i, a: a.frames[i].top, b: b.frames[i].top })
}

log(`## Per-frame argmax agreement`)
log(`  ${numFrames - diffs.length} / ${numFrames} frames agree (${(100 * (numFrames - diffs.length) / numFrames).toFixed(2)}%)`)
log("")

if (diffs.length === 0) {
    log("  Every CTC frame's top-1 token matches across the two runtimes.")
} else {
    log(`## Frames where argmax differs (${diffs.length})\n`)
    for (const d of diffs) {
        log(`  frame ${d.f}:`)
        log(`    ${a.platform.padEnd(8)} ` + d.a.map(t => `id=${t.id} v=${t.val}`).join("  "))
        log(`    ${b.platform.padEnd(8)} ` + d.b.map(t => `id=${t.id} v=${t.val}`).join("  "))
        const aTop1 = d.a[0].val, aTop2 = d.a[1].val
        const bTop1 = d.b[0].val, bTop2 = d.b[1].val
        log(`    top-1/top-2 margin:  ${a.platform} ${(aTop1 - aTop2).toFixed(4)}   ${b.platform} ${(bTop1 - bTop2).toFixed(4)}`)
        log("")
    }
}

process.stdout.write(lines.join("\n") + "\n")
