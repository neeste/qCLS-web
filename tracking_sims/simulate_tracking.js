const fs = require('fs');
const qcls = require('../qcls_core.js');

const json_file = process.argv[2] || 'simulated_listeners_v2.json';
const listeners = JSON.parse(fs.readFileSync(json_file, 'utf8'));
console.log(`Running simulation on ${json_file}...`);

// The 10 frequencies
const freqs_hz = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000];

function simulate_response(theta_210, freq_hz, spl) {
    let f_idx = freqs_hz.indexOf(freq_hz);
    if(f_idx < 0) f_idx = 3; // fallback to 1000Hz

    // true boundaries and slopes for this frequency
    let slopes = theta_210.slice(f_idx * 20, f_idx * 20 + 10);
    let intervals = theta_210.slice(f_idx * 20 + 10, f_idx * 20 + 20);
    let far = theta_210[200 + f_idx];

    let cb = new Float32Array(10);
    let dfa = far / 11.0;

    let md = new Float32Array(10);
    let current_md = 0;
    for(let k=0; k<10; k++) {
        current_md += intervals[k];
        md[k] = current_md;
    }

    for(let k=0; k<10; k++) {
        let pf = 1.0 / (1.0 + Math.exp(-slopes[k] * (spl - md[k])));
        cb[k] = pf * (1.0 - far) + (k + 1) * dfa;
    }

    // eliminate cross-overs
    for (let it = 0; it < 20; it++) {
        for (let j = 0; j < 9; j++) {
            let pd = cb[j+1] - cb[j];
            if (pd < (dfa / 2)) {
                cb[j] = cb[j] + pd/2 - dfa/2;
                cb[j+1] = cb[j] + dfa;
            }
        }
    }

    let cbmx = 1.0 - dfa;
    for (let j = 9; j >= 0; j--) {
        if (cb[j] > cbmx) cb[j] = cbmx;
        cbmx = cb[j] - dfa;
    }

    let cbmn = dfa;
    for (let j = 0; j < 10; j++) {
        if (cb[j] < cbmn) cb[j] = cbmn;
        cbmn = cb[j] + dfa;
    }

    let pcat = new Float32Array(11);
    pcat[0] = cb[0];
    for(let k=1; k<10; k++) {
        pcat[k] = cb[k] - cb[k-1];
    }
    pcat[10] = 1.0 - cb[9];

    let sum = 0;
    for(let k=0; k<11; k++) {
        if(pcat[k] < 0.001) pcat[k] = 0.001;
        sum += pcat[k];
    }
    for(let k=0; k<11; k++) pcat[k] /= sum;

    let cdf = 0;
    let rand_val = Math.random();
    for(let k=0; k<11; k++) {
        cdf += pcat[k];
        if(cdf >= rand_val) {
            return k * 5; // CU: 0, 5, 10... 50
        }
    }
    return 50;
}

qcls.onRuntimeInitialized = () => {
    let total_mae = 0;
    let n_test = Math.min(10, listeners.length);

    let ptrF = qcls._malloc(100 * 4);
    let ptrL = qcls._malloc(100 * 4);
    let ptrR = qcls._malloc(100 * 4);
    let ptrOutF = qcls._malloc(4);
    let ptrOutL = qcls._malloc(4);
    let ptrBounds = qcls._malloc(10 * 4);

    for(let i=0; i<n_test; i++) {
        let listener = listeners[i];
        let theta = listener.theta; // 210 values

        qcls._init_bayesian_state();

        let histF = new Float32Array(qcls.HEAPF32.buffer, ptrF, 100);
        let histL = new Float32Array(qcls.HEAPF32.buffer, ptrL, 100);
        let histR = new Float32Array(qcls.HEAPF32.buffer, ptrR, 100);

        for(let trial=0; trial<100; trial++) {
            // Get next stimulus
            qcls._calculate_bap_next(ptrF, ptrL, ptrR, trial, 0, 0, 110, ptrOutF, ptrOutL);
            let nextF = new Float32Array(qcls.HEAPF32.buffer, ptrOutF, 1)[0];
            let nextL = new Float32Array(qcls.HEAPF32.buffer, ptrOutL, 1)[0];

            // Simulate response
            let resp = simulate_response(theta, nextF, nextL);

            // Record
            histF[trial] = nextF;
            histL[trial] = nextL;
            histR[trial] = resp;
        }

        // Run post-hoc MLE fitting
        qcls._estimate_mcpf(ptrF, ptrL, ptrR, 100);

        // Get boundaries and compute MAE against truth
        let listener_mae = 0;
        let count = 0;
        for(let f=0; f<10; f++) {
            qcls._get_loudness_boundaries(freqs_hz[f], ptrBounds);
            let est_bounds = new Float32Array(qcls.HEAPF32.buffer, ptrBounds, 10);
            
            // Reconstruct true md array for this frequency
            let intervals = theta.slice(f * 20 + 10, f * 20 + 20);
            let true_md = new Float32Array(10);
            let current_md = 0;
            for(let k=0; k<10; k++) {
                current_md += intervals[k];
                true_md[k] = current_md;
            }
            
            for(let k=0; k<10; k++) {
                let diff = Math.abs(est_bounds[k] - true_md[k]);
                listener_mae += diff;
                count++;
            }
        }
        listener_mae /= count;
        total_mae += listener_mae;
    }

    console.log(`Average RMSE (MAE) at 100 trials: ${(total_mae / n_test).toFixed(2)} dB`);
    
    qcls._free(ptrF); qcls._free(ptrL); qcls._free(ptrR);
    qcls._free(ptrOutF); qcls._free(ptrOutL); qcls._free(ptrBounds);
};
