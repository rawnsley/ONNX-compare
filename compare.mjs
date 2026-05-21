// Reads N platform reports and prints:
//   1. A summary table of model/input/logits hashes per platform.
//   2. Pairwise per-frame argmax agreement counts.
//   3. The exact divergent frames, grouped by which pair disagrees.
//
// Reports are the JSON files written by shared/pipeline.mjs#report.

import { readFileSync } from "node:fs"

const paths = process.argv.slice(2)
if (paths.length < 2) {
    process.stderr.write("Usage: node compare.mjs <a.json> <b.json> [<c.json> ...]\n")
    process.exit(2)
}

const reports = paths.map(p => JSON.parse(readFileSync(p, "utf-8")))
const platforms = reports.map(r => r.platform)
const lines = []
const log = s => lines.push(s)

log(`# ONNX-compare: ${platforms.join(" / ")}\n`)

log("## Per-platform summary\n")
const labelWidth = Math.max(...platforms.map(p => p.length))
const pkgWidth = Math.max(...reports.map(r => `${r.ortPackage} ${r.ortVersion}`.length))
log("  " + "platform".padEnd(labelWidth) + "  "
    + "package".padEnd(pkgWidth) + "  "
    + "host".padEnd(14) + "  "
    + "logits hash")
log("  " + "-".repeat(labelWidth + pkgWidth + 14 + 70 + 6))
for (const r of reports) {
    log("  " + r.platform.padEnd(labelWidth) + "  "
        + `${r.ortPackage} ${r.ortVersion}`.padEnd(pkgWidth) + "  "
        + `${r.host.os}/${r.host.arch}`.padEnd(14) + "  "
        + r.logits.hash)
}
log("")

// Sanity check: every report must have read the same model and the same input.
const modelHashes = new Set(reports.map(r => r.modelHash))
const inputHashes = new Set(reports.map(r => r.input.hash))
log(`Model bytes: ${modelHashes.size === 1 ? "✓ identical across runs" : "✗ DIFFER across runs"}`)
log(`Input bytes: ${inputHashes.size === 1 ? "✓ identical across runs" : "✗ DIFFER across runs"}`)
log("")

// Pairwise per-frame argmax agreement.
const numFrames = reports[0].frames.length
log("## Pairwise argmax agreement\n")
log("  Each cell shows frames where the row's top-1 token disagrees with the column's,")
log(`  out of ${numFrames} CTC frames.\n`)
const colWidth = Math.max(labelWidth, 8)
log("  " + " ".repeat(labelWidth + 2) + platforms.map(p => p.padStart(colWidth)).join("  "))
for (let i = 0; i < reports.length; i++) {
    const cells = []
    for (let j = 0; j < reports.length; j++) {
        if (i === j) { cells.push("·".padStart(colWidth)); continue }
        let diffs = 0
        for (let f = 0; f < numFrames; f++) {
            if (reports[i].frames[f].top[0].id !== reports[j].frames[f].top[0].id) diffs++
        }
        cells.push(String(diffs).padStart(colWidth))
    }
    log("  " + platforms[i].padEnd(labelWidth) + "  " + cells.join("  "))
}
log("")

// Frame-by-frame breakdown: for each frame where any platforms disagree, show
// every platform's top-5 so the divergent decision is visible at a glance.
const divergentFrames = []
for (let f = 0; f < numFrames; f++) {
    const top1Ids = reports.map(r => r.frames[f].top[0].id)
    if (new Set(top1Ids).size > 1) divergentFrames.push(f)
}

if (divergentFrames.length === 0) {
    log("All platforms agree on every frame's argmax.")
} else {
    log(`## Frames where argmax differs (${divergentFrames.length})\n`)
    for (const f of divergentFrames) {
        log(`  frame ${f}:`)
        for (const r of reports) {
            const top = r.frames[f].top
            const margin = (top[0].val - top[1].val).toFixed(4)
            log(`    ${r.platform.padEnd(labelWidth)}  ` + top.map(t => `id=${t.id} v=${t.val}`).join("  ")
                + `   (margin ${margin})`)
        }
        log("")
    }
}

process.stdout.write(lines.join("\n") + "\n")
