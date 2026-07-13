# qCLS Web Application: Architectural Overview & User Guide

## 1. Introduction & Technology Stack
The **qCLS** web application is a fully client-side, browser-based tool for measuring categorical loudness growth functions. To achieve real-time, high-performance execution of the Bayesian Adaptive Procedure (BAP) without relying on a remote server, the application utilizes a hybrid architecture:

- **Front-End (HTML/JavaScript):** Handles the user interface, state management, and real-time audio rendering via the Web Audio API.
- **Back-End (C / WebAssembly):** The heavy mathematical lifting—including stimulus generation, psychoacoustic interpolation, and the Extended Kalman Filter—has been translated from objective MATLAB into C. This C code is compiled into WebAssembly (Wasm) using Emscripten, allowing it to run at near-native speeds directly in the user's browser.
- **Security & Privacy:** Because the entire application runs locally in the browser, no participant data (PHI) or trial history is ever transmitted over the internet.

## 2. Translating the Bayesian Tracker (MATLAB to C)
To replicate the MATLAB BAP algorithm in a web environment, the core logic was translated into standard C. Here is how the specific mathematical dependencies were handled:

### A. State Management & Structs
The dynamic qCLS MATLAB class was flattened into static C structs (`qCLS_Config` and `qCLS_State`). Memory footprints are allocated upfront (e.g., matrices for the 9 frequency bands and 11 categories) to ensure memory safety in the browser.

### B. Replicating pchip (Monotonic Interpolation)
MATLAB relies on `pchip` to interpolate categorical boundaries smoothly without overshooting. Because a standard cubic spline would allow boundaries to cross (violating monotonicity), a custom C function (`interpolate_pchip`) was written. This function implements the exact **Fritsch-Carlson shape-preserving interpolant** algorithm used by MATLAB, ensuring the psychometric boundary curves are mathematically identical.

### C. Linear Algebra Engine
The Kalman Filter update (`Kalman_update`) relies heavily on matrix multiplication, transposition, and inversion (e.g., calculating the Kalman gain $K$ and updating the covariance matrix $P$). A custom, lightweight linear algebra engine was built into the C code. Most notably, matrix inversion is handled via **Gauss-Jordan elimination with partial pivoting**, which is highly stable for the 10x10 and 12x12 matrices used in this procedure.

### D. Single-Model Optimization
In the original MATLAB script, the estimator simultaneously tracks 21 different hypothetical models (combinations of anchor frequencies). However, the tracking algorithm (stimulus selection) ultimately selected random frequencies and levels. To drastically reduce the computational load on the browser, the WebAssembly engine currently only executes the Kalman update for the **primary model** (using anchor frequencies 1, 3, 6, and 9). The random selection behavior remains mathematically identical.

## 3. Application Flow & Phase Logic

### Phase 1: Dynamic Range Bounding
Before the BAP takes over, Phase 1 establishes the listener's dynamic range using a rule-based tracking procedure:
1. **First Presentation:** Plays the specified Start Level (default 50 dB SPL) and records the initial response.
2. **Descending:** The level decreases in 10 dB steps until the listener responds "Can't Hear" or the level reaches 0 dB SPL. This locks in the absolute minimum level.
3. **Ascending:** The test jumps back to the initial Start Level. If the very first response was "Loud" or greater, the ascending step size is set to **5 dB**. Otherwise, it uses **10 dB** steps. It ascends until the listener responds "Too Loud" or the level reaches 110 dB SPL. This locks in the absolute maximum level.

### Phase 2: Bayesian Adaptive Tracking
Once bounded, Phase 2 begins. On every trial:
1. JavaScript allocates a block of shared WebAssembly memory (`Module.HEAPF32`).
2. The entire trial history (frequencies, levels, and responses) is copied into this shared memory.
3. JavaScript calls the compiled C function `_calculate_bap_next`.
4. The C engine resets the state, loops through the history, and runs the `Kalman_update` to build the current posterior model of the listener.
5. The C engine selects the next frequency and level (currently using the randomized selection bounded by the Phase 1 limits) and passes the next target back to JavaScript.

## 4. Audio Generation & Routing
Stimulus generation (Pure Tones, SAM-Tones, and Five-Tone Complexes) is calculated sample-by-sample in the C engine to ensure phase accuracy and precise RMS scaling.
When JavaScript requests a stimulus, the C engine writes the raw float data into shared memory. JavaScript reads this data, scales it to the target dB SPL, and feeds it into an `AudioBufferSourceNode`. The audio is then routed through a `StereoPannerNode` (directed by the UI's "Test Ear" dropdown) before reaching the hardware destination.

## 5. The Loudness Map (Heatmap Rendering)
Rather than using a naive IDW (Inverse Distance Weighting) blur to draw the final results, the heatmap queries the Bayesian model directly.
When the test completes, the JavaScript canvas iterates across the frequency spectrum. For every frequency slice, it asks the C engine (`_get_loudness_boundaries`) to evaluate the current state of the phi parameters through the `pchip` interpolator. The resulting visualization paints the exact, mathematically monotonic categorical bands modeled by the Kalman filter.

## 6. Global MCPF Estimation (MLE)
The application includes a highly advanced Maximum Likelihood Estimator (MLE) built directly into the WebAssembly layer. Upon completion of the test, the algorithm fits the sparse trial data against a 210-parameter generative model of human hearing (MCPF). 

1. **PCA Latent Space:** To prevent overfitting on sparse data, the C engine loads a pre-computed Principal Component Analysis (PCA) model of human hearing (`pca_model.h`). It optimizes only 10 PCA weights, mathematically projecting them back out to a full 210-parameter physiological model.
2. **Nelder-Mead Optimization:** A custom downhill simplex algorithm searches the 10-dimensional latent space to minimize the Negative Log-Likelihood (NLL) of the patient's exact responses.
3. **Clinical Metrics:** Once converged, the optimized model effortlessly separates the patient's threshold, loudness slopes, and False-Alarm Rate (FAR). The **Average Slope** and **Average FAR** are instantly extracted from the C memory layer and displayed as actionable clinical metrics.

## 7. How to Use the App
1. **Load the App:** Open `index.html` in any modern web browser. (The app requires a local server environment, like Python's `http.server`, to allow WebAssembly to load securely).
2. **Configure:** Enter the Participant ID, Test Ear, Stimulus Type/Bandwidth, and Start Level.
3. **Calibrate:** Click **Calibrate** to play a continuous 1 kHz tone at a reference level of 60 dB SPL to calibrate your hardware equipment. Click again to stop.
4. **Run Test:** Click **Run Test** to begin Phase 1.
5. **Export Data:** Upon completion (or when 100 Phase 2 trials are reached), the Heatmap will render. Click **Download CSV Data** to export the trial history. The filename is automatically formatted as `qCLS_[ParticipantID]_[Ear]_[YYYYMMDD].csv`.
