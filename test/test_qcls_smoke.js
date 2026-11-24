"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.resolve(__dirname, "..");
const INDEX_PATH = path.join(ROOT, "index.html");
const CORE_JS_PATH = path.join(ROOT, "qcls_core.js");

function checkIndexHtmlScriptsAndVersion() {
    const html = fs.readFileSync(INDEX_PATH, "utf8");
    const scriptBlocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)];
    assert(scriptBlocks.length >= 2, "Expected at least 2 inline <script> blocks in index.html");
    scriptBlocks.forEach((m, idx) => {
        new vm.Script(m[1], { filename: `index.html#script${idx + 1}` });
    });

    const verMatch = html.match(/const\s+QCLS_VERSION\s*=\s*['"]([^'"]+)['"]/);
    assert(verMatch, "Missing QCLS_VERSION constant in index.html");
    const version = verMatch[1];

    const srcMatch = html.match(/<script\s+src="qcls_core\.js\?v=([^"]+)"/);
    assert(srcMatch, "Missing cache-busted qcls_core.js?v=... script tag in index.html");
    assert.equal(srcMatch[1], version, "qcls_core.js?v=... must match QCLS_VERSION");
    assert(html.includes("locateFile"), "Module.locateFile must be set for qcls_core.wasm cache-busting");
    return { html, scriptBlocks, version };
}

function measureStimulus(Module, stimType, bwOct, centerHz = 1000) {
    const fsHz = 48000;
    const dur = 1.0;
    const n = fsHz * dur;
    const ptr = Module._malloc(n * 4);
    try {
        Module._generate_stimulus(dur, fsHz, stimType, bwOct, centerHz, ptr);
        const buf = new Float32Array(Module.HEAPF32.buffer, ptr, n);
        // Steady-state region excluding 50 ms cosine ramps
        const i0 = Math.round(0.06 * fsHz);
        const i1 = Math.round(0.94 * fsHz);
        let sumSq = 0;
        let peak = 0;
        for (let i = i0; i < i1; i++) {
            const v = Math.abs(buf[i]);
            sumSq += v * v;
            if (v > peak) peak = v;
        }
        const rms = Math.sqrt(sumSq / (i1 - i0));
        return { rms, peak, crestFactor: peak / rms };
    } finally {
        Module._free(ptr);
    }
}

function checkDomExportAndSessionRoundTrip(scriptBlocks, sampleMu) {
    const downloads = [];
    const store = new Map();
    const elements = new Map();

    function makeEl(id, initial = {}) {
        const el = {
            id,
            value: initial.value ?? "",
            innerText: initial.innerText ?? "",
            innerHTML: "",
            hidden: false,
            disabled: false,
            open: false,
            dataset: {},
            style: { setProperty() {} },
            setAttribute() {},
            addEventListener() {},
            focus() {},
            click() { downloads.push({ filename: this.download, href: this.href }); },
            showModal() { this.open = true; },
            close() { this.open = false; },
            reportValidity() { return true; },
            appendChild() {},
            get valueAsNumber() { return Number(this.value); },
            getContext() {
                return {
                    fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
                    save() {}, restore() {}, rect() {}, clip() {}, arc() {}, fill() {},
                    setLineDash() {}, fillText() {}, translate() {}, rotate() {}
                };
            },
            width: 500,
            height: 500
        };
        elements.set(id, el);
        return el;
    }

    [
        ["participant-id", { value: "BT_042" }],
        ["test-ear", { value: "right" }],
        ["stim-type", { value: "3" }],
        ["stim-bw", { value: "oct_0.25" }],
        ["center-freq", { value: "1000" }],
        ["start-level", { value: "50" }],
        ["max-output", { value: "100" }],
        ["adaptive-tracking", { value: "2" }],
        ["total-trials", { value: "100" }],
        ["audio-units", { value: "hl" }],
        ["transducer", { value: "1" }],
        ["cal-frequency", { value: "1000" }],
        ["cal-level", { value: "60" }],
        ["cal-btn", {}],
        ["run-btn", {}],
        ["stop-btn", {}],
        ["menu-btn", {}],
        ["status", {}],
        ["engine-error", {}],
        ["modal-overlay", {}],
        ["pause-modal", {}],
        ["pause-status", {}],
        ["pause-resume-btn", {}],
        ["control-menu", {}],
        ["scale-container", {}],
        ["audiogram-grid", {}],
        ["session-list", {}],
        ["sound-indicator", {}],
        ["progress-container", {}],
        ["test-progress", {}],
        ["prog-text", {}],
        ["map-title", {}],
        ["heatmap", {}],
        ["metrics-container", {}],
        ["val-slope", { innerText: "0.62 CU/dB" }],
        ["dl-btn", {}],
        ["restart-message", {}]
    ].forEach(([id, init]) => makeEl(id, init));
    [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000].forEach(f => makeEl("thr-" + f, { value: "" }));

    const blobs = new Map();
    let blobSeq = 0;

    const sandbox = {
        console,
        Math,
        Number,
        String,
        Array,
        Object,
        JSON,
        Date,
        Float32Array,
        parseInt,
        parseFloat,
        isFinite,
        isNaN,
        setTimeout: () => 1,
        clearTimeout: () => {},
        location: { protocol: "http:" },
        localStorage: {
            getItem: (k) => store.has(k) ? store.get(k) : null,
            setItem: (k, v) => store.set(k, String(v))
        },
        Blob: class FakeBlob {
            constructor(parts) { this.text = parts.join(""); }
        },
        URL: {
            createObjectURL: (b) => {
                const u = `blob:mock/${++blobSeq}`;
                blobs.set(u, b.text);
                return u;
            },
            revokeObjectURL: () => {}
        },
        document: {
            getElementById: (id) => elements.get(id) || makeEl(id),
            createElement: () => makeEl("_tmp_" + (++blobSeq)),
            querySelectorAll: () => []
        },
        window: {}
    };
    vm.createContext(sandbox);
    scriptBlocks.forEach((m) => vm.runInContext(m[1], sandbox));

    // Populate a synthetic completed 5TC-lin session and verify save/reload/export
    vm.runInContext(`
        session.ceiling = outputCeiling();
        session.maxL = 95;
        session.nThresholds = 3;
        session.meta = captureSessionMeta();
        session.history = [
            { p: 'P1', f: 1000, l: 50.0, r: 20 },
            { p: 'P2', f: 4000, l: 65.0, r: 25 }
        ];
        session.report = { mu: ${JSON.stringify(sampleMu)} };
        saveSession();
    `, sandbox);

    // Mutate form controls after completion to verify session.meta preserves run provenance
    elements.get("stim-type").value = "0";
    elements.get("participant-id").value = "OTHER_ID";

    vm.runInContext(`downloadELC(); downloadTBT();`, sandbox);
    assert.equal(downloads.length, 2, "Expected ELC and TBT downloads");

    const elcDl = downloads[0];
    const tbtDl = downloads[1];
    assert.match(elcDl.filename, /^qCLS_ELC_BT_042_R_\d{8}_\d{4}\.csv$/);
    assert.match(tbtDl.filename, /^qCLS_TBT_BT_042_R_\d{8}_\d{4}\.csv$/);

    const elcCsv = blobs.get(elcDl.href);
    const tbtCsv = blobs.get(tbtDl.href);
    const elcLines = elcCsv.trim().split("\n");
    const tbtLines = tbtCsv.trim().split("\n");

    assert.equal(elcLines[0], "Band,Freq_Hz,CU,Level_dB_SPL,Stim_Type,Bandwidth,Tracking_Mode,Ceiling_dB_SPL");
    assert.equal(elcLines.length, 91, "ELC CSV must contain header + 90 rows");
    elcLines.slice(1).forEach((line) => {
        const cols = line.split(",");
        assert.equal(cols.length, 8, "Every ELC row must have exactly 8 comma-separated fields");
        assert.equal(cols[4], "5TC-lin");
        assert.equal(cols[5], "0.25_oct");
        assert.equal(cols[6], "isoPhon");
        assert.equal(cols[7], "100");
    });

    assert.equal(tbtLines[0], "Phase,Trial,Freq,Level,Response,Stim_Type,Bandwidth,Tracking_Mode,Ceiling_dB_SPL");
    assert.equal(tbtLines.length, 3, "TBT CSV must contain header + 2 trial rows");
    assert(tbtLines[1].endsWith(",5TC-lin,0.25_oct,isoPhon,100"));

    // Reload saved session from localStorage and verify controls + timestamp restore
    elements.get("session-list").value = "0";
    vm.runInContext(`loadSession();`, sandbox);
    assert.equal(elements.get("participant-id").value, "BT_042");
    assert.equal(elements.get("stim-type").value, "3");
    assert.equal(elements.get("stim-bw").value, "oct_0.25");

    // Cross-app parser check against qPRO and qLDM if sibling repos are present
    const qproPath = path.resolve(ROOT, "../qPRO/qpro_core.js");
    if (fs.existsSync(qproPath)) {
        const QPro = require(qproPath);
        const parsedQpro = QPro.parseQclsElcCsv(elcCsv, elcDl.filename);
        assert.equal(parsedQpro.participantId, "BT_042");
        assert.equal(parsedQpro.ear, "Right");
        assert(Math.abs(parsedQpro.elc10x10[3][0] - sampleMu[30]) < 0.02);
    }
    const qldmPath = path.resolve(ROOT, "../qLDM/qldm_core.js");
    if (fs.existsSync(qldmPath)) {
        const QLDM = require(qldmPath);
        const parsedQldm = QLDM.parseQclsElcCsv(elcCsv, elcDl.filename);
        assert.equal(parsedQldm.ok, true);
        assert.equal(parsedQldm.participantId, "BT_042");
        assert.equal(parsedQldm.ear, "Right");
        assert(Math.abs(parsedQldm.cu5_1k - sampleMu[30]) < 0.02);
    }
}

async function main() {
    const { scriptBlocks, version } = checkIndexHtmlScriptsAndVersion();

    let resolveRuntime;
    const ready = new Promise((r) => { resolveRuntime = r; });
    const Module = require(CORE_JS_PATH);
    if (Module.HEAPF32 && Module._init_bayesian_state) resolveRuntime();
    else Module.onRuntimeInitialized = resolveRuntime;
    await ready;

    // 1. Stimulus synthesis & crest factors
    const tone = measureStimulus(Module, 0, 0.0, 1000);
    const tcLog = measureStimulus(Module, 2, 0.25, 1000);
    const tcLin = measureStimulus(Module, 3, 0.25, 1000);
    assert(Math.abs(tone.rms - 1.0) < 0.01 && Math.abs(tone.crestFactor - Math.SQRT2) < 0.02);
    assert(Math.abs(tcLog.rms - 1.0) < 0.02 && tcLog.crestFactor > 3.0);
    assert(Math.abs(tcLin.rms - 1.0) < 0.02 && tcLin.crestFactor < 2.0);

    // 2. Unconditioned CUK0 prior parity & transducer RETSPL conditioning
    Module._init_bayesian_state();
    const ptrMuA = Module._malloc(90 * 4);
    const ptrSdA = Module._malloc(90 * 4);
    const ptrMuB = Module._malloc(90 * 4);
    const ptrThr = Module._malloc(10 * 4);
    try {
        Module._qcls_report(ptrMuA, ptrSdA, 0);
        const muUncond = Array.from(new Float32Array(Module.HEAPF32.buffer, ptrMuA, 90));
        const sdUncond = Array.from(new Float32Array(Module.HEAPF32.buffer, ptrSdA, 90));

        new Float32Array(Module.HEAPF32.buffer, ptrThr, 10).fill(NaN);
        Module._qcls_audiogram_prior(ptrThr, 1);
        Module._qcls_report(ptrMuB, 0, 0);
        const muAllNan = Array.from(new Float32Array(Module.HEAPF32.buffer, ptrMuB, 90));

        for (let i = 0; i < 90; i++) {
            assert(Math.abs(muUncond[i] - muAllNan[i]) < 1e-4, "init_bayesian_state must match all-NaN CUK0 prior");
        }
        assert(muUncond[30] < 36.0 && sdUncond[30] > 17.0, "1 kHz CU 5 prior must reflect CUK0 (~33.7 +/- 18.0 dB)");

        // Verify Legacy vs ER-3A RETSPL difference at 250 Hz (30.0 vs 17.5 dB -> 9.72 dB CU5 shift)
        const thr20 = new Float32Array(Module.HEAPF32.buffer, ptrThr, 10);
        thr20.fill(20.0);
        thr20[9] = NaN;
        Module._qcls_set_transducer(0);
        Module._qcls_audiogram_prior(ptrThr, 1);
        Module._qcls_report(ptrMuA, 0, 0);
        const cu5Legacy250 = new Float32Array(Module.HEAPF32.buffer, ptrMuA, 1)[0];

        Module._qcls_set_transducer(1);
        Module._qcls_audiogram_prior(ptrThr, 1);
        Module._qcls_report(ptrMuB, 0, 0);
        const cu5Er3a250 = new Float32Array(Module.HEAPF32.buffer, ptrMuB, 1)[0];
        assert(Math.abs((cu5Legacy250 - cu5Er3a250) - 9.724) < 0.05, "250 Hz Legacy vs ER-3A prior shift mismatch");
    } finally {
        Module._free(ptrMuA);
        Module._free(ptrSdA);
        Module._free(ptrMuB);
        Module._free(ptrThr);
    }

    // 3. Simulated adaptive session + post-hoc PCA fit & GP report monotonicity
    Module._init_bayesian_state();
    const freqs = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000];
    const histF = [], histL = [], histR = [];
    const pOutF = Module._malloc(4), pOutL = Module._malloc(4);
    try {
        for (let t = 0; t < 36; t++) {
            let f, l;
            if (t < 9) {
                f = freqs[t];
                l = 55;
            } else if (t % 3 === 0) {
                Module._qcls_select_isophon_next(0, 100, pOutF, pOutL);
                f = new Float32Array(Module.HEAPF32.buffer, pOutF, 1)[0];
                l = new Float32Array(Module.HEAPF32.buffer, pOutL, 1)[0];
            } else if (t % 3 === 1) {
                Module._qcls_select_bayesian_next(0, 100, pOutF, pOutL);
                f = new Float32Array(Module.HEAPF32.buffer, pOutF, 1)[0];
                l = new Float32Array(Module.HEAPF32.buffer, pOutL, 1)[0];
            } else {
                Module._qcls_select_gp_next(0, 100, pOutF, pOutL);
                f = new Float32Array(Module.HEAPF32.buffer, pOutF, 1)[0];
                l = new Float32Array(Module.HEAPF32.buffer, pOutL, 1)[0];
            }
            const cu = Math.max(0, Math.min(50, Math.round(((l - 15) / 85) * 50 / 5) * 5));
            histF.push(f); histL.push(l); histR.push(cu);
            Module._qcls_update_trial(f, l, cu);
            Module._qcls_gp_update_trial(f, l, cu);
        }
    } finally {
        Module._free(pOutF);
        Module._free(pOutL);
    }

    const nT = histF.length;
    const pF = Module._malloc(nT * 4), pL = Module._malloc(nT * 4), pR = Module._malloc(nT * 4);
    const pPca = Module._malloc(100 * 4), pGp = Module._malloc(100 * 4);
    let pcaMu;
    try {
        new Float32Array(Module.HEAPF32.buffer, pF, nT).set(histF);
        new Float32Array(Module.HEAPF32.buffer, pL, nT).set(histL);
        new Float32Array(Module.HEAPF32.buffer, pR, nT).set(histR);
        assert.equal(Module._qcls_pca_fit_report(pF, pL, pR, nT, pPca), 1);
        assert.equal(Module._qcls_gp_report(pGp, 0), 1);
        pcaMu = Array.from(new Float32Array(Module.HEAPF32.buffer, pPca, 100));
        const gpMu = Array.from(new Float32Array(Module.HEAPF32.buffer, pGp, 100));
        for (let b = 0; b < 9; b++) {
            for (let k = 1; k < 10; k++) {
                assert(Number.isFinite(pcaMu[b * 10 + k]), "Non-finite PCA boundary");
                assert(gpMu[b * 10 + k] > gpMu[b * 10 + k - 1], "GP boundaries must be strictly monotonic");
            }
        }
    } finally {
        [pF, pL, pR, pPca, pGp].forEach((p) => Module._free(p));
    }

    // 4. DOM export provenance, _HHMM filename, and qPRO/qLDM parser compatibility
    checkDomExportAndSessionRoundTrip(scriptBlocks, pcaMu);

    console.log(
        `[PASS qCLS v${version}] Scripts, WASM crest factors (5TC-log=${tcLog.crestFactor.toFixed(3)}, ` +
        `5TC-lin=${tcLin.crestFactor.toFixed(3)}), CUK0 prior parity, trackers, and _HHMM provenance exports verified.`
    );
}

main().catch((err) => {
    console.error("[FAIL qCLS test]", err);
    process.exit(1);
});
