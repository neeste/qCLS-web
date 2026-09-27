"use strict";

const path = require("node:path");
const readline = require("node:readline");

const moduleArgument = process.argv.indexOf("--module");
if (moduleArgument < 0 || !process.argv[moduleArgument + 1]) {
    throw new Error("Expected --module <qcls_core.js>");
}

let resolveRuntime;
const runtimeReady = new Promise((resolve) => {
    resolveRuntime = resolve;
});
const Module = require(path.resolve(process.argv[moduleArgument + 1]));
Module.onRuntimeInitialized = resolveRuntime;

function readStimulus(selector, minLevel, maxLevel) {
    const freqPointer = Module._malloc(4);
    const levelPointer = Module._malloc(4);
    try {
        selector(minLevel, maxLevel, freqPointer, levelPointer);
        const frequency = new Float32Array(Module.HEAPF32.buffer, freqPointer, 1)[0];
        const level = new Float32Array(Module.HEAPF32.buffer, levelPointer, 1)[0];
        return { frequency, level };
    } finally {
        Module._free(freqPointer);
        Module._free(levelPointer);
    }
}

function runFit(message) {
    const { frequencies, levels, responses } = message;
    if (!Array.isArray(frequencies) || frequencies.length === 0 ||
        levels.length !== frequencies.length || responses.length !== frequencies.length) {
        throw new Error("fit requires equally sized non-empty frequencies, levels, and responses");
    }

    const count = frequencies.length;
    const frequencyPointer = Module._malloc(count * 4);
    const levelPointer = Module._malloc(count * 4);
    const responsePointer = Module._malloc(count * 4);
    const outputPointer = Module._malloc(100 * 4);
    try {
        new Float32Array(Module.HEAPF32.buffer, frequencyPointer, count).set(frequencies);
        new Float32Array(Module.HEAPF32.buffer, levelPointer, count).set(levels);
        new Float32Array(Module.HEAPF32.buffer, responsePointer, count).set(responses);
        const ok = Module._qcls_pca_fit_report(
            frequencyPointer, levelPointer, responsePointer, count, outputPointer
        );
        if (!ok) throw new Error("qcls_pca_fit_report failed");
        return Array.from(new Float32Array(Module.HEAPF32.buffer, outputPointer, 100));
    } finally {
        Module._free(frequencyPointer);
        Module._free(levelPointer);
        Module._free(responsePointer);
        Module._free(outputPointer);
    }
}

function installPcaModel(model) {
    const fields = [
        ["mu", 100],
        ["d", 100],
        ["V", 500],
        ["score_mean", 5],
        ["score_std", 5],
    ];
    if (!model || fields.some(([name, length]) =>
        !Array.isArray(model[name]) || model[name].length !== length)) {
        throw new Error("PCA model must contain arrays sized 100, 100, 500, 5, and 5");
    }

    const pointers = fields.map(([, length]) => Module._malloc(length * 4));
    try {
        fields.forEach(([name], index) => {
            Module.HEAPF32.set(model[name], pointers[index] / 4);
        });
        const ok = Module._qcls_pca_set_model(...pointers);
        if (!ok) throw new Error("qcls_pca_set_model rejected the supplied parameters");
        return { ok: true };
    } finally {
        pointers.forEach((pointer) => Module._free(pointer));
    }
}

async function dispatch(message) {
    await runtimeReady;
    switch (message.cmd) {
        case "init":
            Module._init_bayesian_state();
            return { ok: true };
        case "update":
            Module._qcls_update_trial(message.frequency, message.level, message.response);
            return { ok: true };
        case "bayesian":
            return readStimulus(
                Module._qcls_select_bayesian_next,
                message.min_level,
                message.max_level
            );
        case "isophon":
            return readStimulus(
                Module._qcls_select_isophon_next,
                message.min_level,
                message.max_level
            );
        case "fit":
            return { boundaries: runFit(message) };
        case "set_pca_model":
            return installPcaModel(message.model);
        default:
            throw new Error(`Unknown bridge command: ${message.cmd}`);
    }
}

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let pending = Promise.resolve();
input.on("line", (line) => {
    pending = pending.then(async () => {
        try {
            const result = await dispatch(JSON.parse(line));
            process.stdout.write(JSON.stringify(result) + "\n");
        } catch (error) {
            process.stdout.write(JSON.stringify({ ok: false, error: String(error) }) + "\n");
        }
    });
});