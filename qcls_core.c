#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "pca_model.h"

#define PI 3.14159265358979323846f

// --- BAP CONSTANTS ---
#define N_CATEGORIES 11
#define N_FREQS 9
#define K_FREQS 4
#define PHI_LEN (K_FREQS * 3)
// Anchor subsets: the two endpoints are fixed at frequency 1 and N_FREQS and
// the middle K_FREQS-2 are chosen from 2..N_FREQS-1, so C(7,2) = 21 models.
// Mirrors nchoosek(2:Nfreqs-1, kfreqs-2) in qCLS.m.
#define N_MODELS 21

// Prior for phi conditioned on the audiogram, generated from
// qCLS_core/cls_prior_hyper.mat. Rows are the 9 BAP frequency bands, columns
// the three boundaries calc_alpha anchors on (indices 1, 5 and 10).
// CUK is {intercept, slope, residual_sd} regressed on threshold in dB SPL.
static const float CUK[9][3][3] = {
  { {9.541069f,0.777958f,9.954793f}, {82.747685f,0.152409f,8.663963f}, {113.014599f,0.022964f,10.365659f} },
  { {7.175458f,0.832067f,9.907452f}, {77.696519f,0.219623f,9.269154f}, {110.872889f,0.035455f,10.986171f} },
  { {8.062010f,0.830853f,9.963816f}, {78.301655f,0.200816f,10.672021f}, {108.747053f,0.059845f,12.102284f} },
  { {12.292245f,0.793625f,10.238809f}, {77.214433f,0.221603f,9.790290f}, {108.213324f,0.041339f,11.448002f} },
  { {11.761756f,0.858843f,10.220027f}, {77.688104f,0.246026f,11.642567f}, {106.333359f,0.081863f,11.000892f} },
  { {6.863046f,0.796562f,10.058991f}, {73.841899f,0.247670f,9.456480f}, {103.948009f,0.097027f,11.129556f} },
  { {6.677511f,0.786996f,10.432455f}, {72.930442f,0.275449f,10.853000f}, {102.422606f,0.146522f,11.883761f} },
  { {9.535355f,0.832504f,9.874957f}, {74.715981f,0.286727f,10.149155f}, {103.563136f,0.149755f,11.280537f} },
  { {15.266472f,0.853497f,11.330088f}, {78.947260f,0.274759f,10.030553f}, {111.246051f,0.042939f,11.072435f} },
};
// CUK0 is {mean, sd} over the cohort, used where a threshold is missing: the
// same population with the audiogram integrated out rather than guessed.
static const float CUK0[9][3][2] = {
  { {43.944701f,15.272769f}, {89.487677f,8.956191f}, {114.030154f,10.371296f} },
  { {36.484583f,17.987326f}, {85.432625f,10.080657f}, {112.121787f,11.004780f} },
  { {29.008255f,18.090650f}, {83.364343f,11.278790f}, {110.255790f,12.151055f} },
  { {35.666137f,19.516764f}, {83.741102f,10.833961f}, {109.430848f,11.480671f} },
  { {30.860785f,18.301520f}, {83.159245f,12.428355f}, {108.153834f,11.095665f} },
  { {36.142093f,20.555059f}, {82.945452f,10.976743f}, {107.514392f,11.341715f} },
  { {32.429770f,19.958819f}, {81.943746f,12.379568f}, {107.217130f,12.298748f} },
  { {43.811535f,23.613264f}, {86.521230f,12.553086f}, {109.728893f,11.922152f} },
  { {45.121980f,21.822886f}, {88.558375f,11.690277f}, {112.748055f,11.112123f} },
};
// Reference equivalent threshold SPL, used to convert dB HL to dB SPL.
//
// RETSPL depends on the transducer, and the difference is not small: insert
// earphones sit roughly 10 dB below supra-aural at 250 Hz, which is where the
// regression of the CU5 boundary on threshold has its steepest slope, 0.778.
// Passing dB HL measured on inserts through a supra-aural table therefore
// shifts the low-frequency prior by more than the residual scatter it is
// built on. It would not fail loudly.
//
// Index with a RetsplSet value. RETSPL_LEGACY reproduces the single table
// that was hardcoded here, and is the default, so nothing changes unless a
// transducer is chosen deliberately.
//
// VERIFY BEFORE CLINICAL USE. The non-legacy rows are the nominal published
// values for those transducers and are provided so the mechanism is usable,
// not as a substitute for the calibration in force where the test runs. Check
// them against ANSI S3.6 and against your own coupler measurements, and
// correct them here and in qCLS_audiogram_prior.m together: the two must
// agree or the MATLAB and WebAssembly paths will disagree about the prior.
typedef enum {
    RETSPL_LEGACY = 0,   // the table previously hardcoded, provenance unrecorded
    RETSPL_TDH39,        // supra-aural, NBS 9A coupler
    RETSPL_TDH49,        // supra-aural, NBS 9A coupler
    RETSPL_ER3A,         // insert, HA-2 with rigid tube
    RETSPL_HDA200,       // circumaural
    RETSPL_NSETS
} RetsplSet;

// Columns: 250 500 750 1000 1500 2000 3000 4000 6000 8000 Hz.
static const float RETSPL_TABLE[RETSPL_NSETS][10] = {
    {30.0f, 19.0f, 12.0f, 10.0f,  9.0f, 15.0f, 15.5f, 13.0f, 13.0f, 14.0f},
    {25.5f, 11.5f,  8.0f,  7.0f,  6.5f,  9.0f, 10.0f,  9.5f, 15.5f, 13.0f},
    {26.5f, 13.5f,  8.5f,  7.5f,  7.5f, 11.0f,  9.5f, 10.5f, 13.5f, 13.0f},
    {14.0f,  5.5f,  2.0f,  0.0f,  2.0f,  3.0f,  3.5f,  5.5f,  2.0f,  0.0f},
    {30.5f, 18.0f, 17.0f, 16.5f, 16.0f, 16.0f, 14.0f, 16.0f, 21.0f, 15.5f}
};

// Which set qcls_audiogram_prior uses. Set through qcls_set_transducer.
static int retspl_set = RETSPL_LEGACY;

// Select the transducer whose RETSPL converts dB HL to dB SPL. Out-of-range
// values leave the current selection alone and return the active one, so a
// caller cannot silently land on a wrong table.
int qcls_set_transducer(int which) {
    if (which >= 0 && which < RETSPL_NSETS) retspl_set = which;
    return retspl_set;
}

// Band center frequencies. Shared by the Hz-to-index mapping and by stimulus
// selection, so a chosen band is presented at a frequency the model has a
// band for.
static const float BAND_HZ[N_FREQS] = {250,500,750,1000,1500,2000,3000,4000,6000};

typedef enum { STIM_TONE = 0, STIM_SAM_TONE, STIM_FIVE_TONE } StimulusType;

// --- STRUCTS ---
typedef struct {
    int Ncategories;
    int Nfreqs;
    int kfreqs;
    float beta;
    float lambda;
    float phi_prior_mu[N_FREQS][3];
    // Per frequency, so a band whose prior is better informed can be given a
    // tighter one. Uniform rows reproduce the previous single [3] behavior.
    float phi_prior_std[N_FREQS][3];
    float x_lim[2][2]; 
    float Lclearance;
    float fclearance;
    float likelihood_exp;
    float diffusion;
} qCLS_Config;

typedef struct {
    qCLS_Config par;
    float x_current[2];  
    float x_next[2];     
    int trial_n;
    // A discrete posterior over anchor-frequency subsets with a Gaussian over
    // phi inside each, rather than one fixed subset. Reading the most likely
    // model alone discards the disagreement between them; by the law of total
    // variance the posterior variance is
    //   V = sum_m w_m*Var_m + sum_m w_m*(mu_m - mu)^2
    // and only the first term is available from a single model.
    float phi[N_MODELS][PHI_LEN];
    float P[N_MODELS][PHI_LEN * PHI_LEN];
    float mkfreqs[N_MODELS][K_FREQS];
    float Lmodels[N_MODELS];
} qCLS_State;

// Global instance of the test state
qCLS_State global_qcls_state;

// Set once the configuration is populated. calculate_bap_next reseeds the
// posterior on every call but must not overwrite a prior the caller has
// already conditioned on an audiogram.
static int qcls_par_ready = 0;

// --- FORWARD DECLARATIONS ---
void qcls_seed_models(void);
void qcls_audiogram_prior(float* thr, int units_are_hl);
int  qcls_set_transducer(int which);
void qcls_report(float* out_mu, float* out_sd, float* out_muMAP);
float Kalman_update(qCLS_State* qcls, float* phi, float* P, float* kfreqs, int phi_len, float freq, float lev, int* r_bool);
void CLS_psycfun(qCLS_State* qcls, float freq, float lev, float* kfreqs, float* phi, float* p_out);
void CLS_jacobian(qCLS_State* qcls, float freq, float lev, float* kfreqs, float* phi, int phi_len, float* H_out);
void calc_alpha(qCLS_State* qcls, float freq, float* kfreqs, float* phi, float* alpha_out);
float interpolate_pchip(float* x, float* y, int n, float xq);

// --- INITIALIZATION ---
void init_bayesian_state() {
    global_qcls_state.trial_n = 0;
    global_qcls_state.par.Ncategories = N_CATEGORIES;
    global_qcls_state.par.Nfreqs = N_FREQS;
    global_qcls_state.par.kfreqs = K_FREQS;
    global_qcls_state.par.beta = 0.5f;
    global_qcls_state.par.lambda = 0.1f;
    
    for (int i = 0; i < N_FREQS; i++) {
        global_qcls_state.par.phi_prior_mu[i][0] = 40.0f;
        global_qcls_state.par.phi_prior_mu[i][1] = 90.0f;
        global_qcls_state.par.phi_prior_mu[i][2] = 110.0f;
    }
    
    for (int i = 0; i < N_FREQS; i++) {
        global_qcls_state.par.phi_prior_std[i][0] = 10.0f;
        global_qcls_state.par.phi_prior_std[i][1] = 10.0f;
        global_qcls_state.par.phi_prior_std[i][2] = 10.0f;
    }

    global_qcls_state.par.x_lim[0][0] = 1.0f;
    global_qcls_state.par.x_lim[0][1] = 0.0f;
    global_qcls_state.par.x_lim[1][0] = (float)N_FREQS;
    global_qcls_state.par.x_lim[1][1] = 110.0f;
    
    global_qcls_state.par.Lclearance = 0.1f;
    global_qcls_state.par.fclearance = 0.1f;
    global_qcls_state.par.likelihood_exp = 0.95f;
    global_qcls_state.par.diffusion = 0.01f;

    qcls_par_ready = 1;
    qcls_seed_models();
}

// Enumerate the anchor subsets and seed each model from the current prior.
// Separate from init_bayesian_state so that conditioning the prior on an
// audiogram can reseed without disturbing the rest of the configuration.
//
// This also supplies the starting phi and P. Previously neither was ever
// written: the state is a file-scope global, so phi and P began as all zeros,
// and because the diffusion step is multiplicative (P += diffusion*P) a zero
// covariance stayed zero. The Kalman gain was therefore identically zero and
// phi never moved off zero for the whole run.
void qcls_seed_models(void) {
    int m = 0;
    for (int i = 2; i <= N_FREQS - 1; i++) {
        for (int j = i + 1; j <= N_FREQS - 1; j++) {
            global_qcls_state.mkfreqs[m][0] = 1.0f;
            global_qcls_state.mkfreqs[m][1] = (float)i;
            global_qcls_state.mkfreqs[m][2] = (float)j;
            global_qcls_state.mkfreqs[m][3] = (float)N_FREQS;
            m++;
        }
    }

    for (m = 0; m < N_MODELS; m++) {
        global_qcls_state.Lmodels[m] = 0.0f;
        for (int a = 0; a < K_FREQS; a++) {
            int f = (int)global_qcls_state.mkfreqs[m][a] - 1;   // 0-based band
            for (int j = 0; j < 3; j++) {
                int k = a * 3 + j;
                global_qcls_state.phi[m][k] = global_qcls_state.par.phi_prior_mu[f][j];
                for (int c = 0; c < PHI_LEN; c++) {
                    global_qcls_state.P[m][k * PHI_LEN + c] = 0.0f;
                }
                float sd = global_qcls_state.par.phi_prior_std[f][j];
                global_qcls_state.P[m][k * PHI_LEN + k] = sd * sd;
            }
        }
    }
    global_qcls_state.trial_n = 0;
}

// --- MATRIX MATH HELPERS ---
void mat_transpose(float* A, int a_rows, int a_cols, float* Out) {
    for (int r = 0; r < a_rows; r++) {
        for (int c = 0; c < a_cols; c++) {
            Out[c * a_rows + r] = A[r * a_cols + c];
        }
    }
}

void mat_mult(float* A, int a_rows, int a_cols, float* B, int b_rows, int b_cols, float* Out) {
    for (int r = 0; r < a_rows; r++) {
        for (int c = 0; c < b_cols; c++) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; k++) {
                sum += A[r * a_cols + k] * B[k * b_cols + c];
            }
            Out[r * b_cols + c] = sum;
        }
    }
}

void mat_add(float* A, float* B, int rows, int cols, float* Out) {
    int total_elements = rows * cols;
    for (int i = 0; i < total_elements; i++) {
        Out[i] = A[i] + B[i];
    }
}

int mat_invert(float* A, int n, float* Out) {
    float aug[20][40] = {0}; 
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++) {
            aug[r][c] = A[r * n + c];
            aug[r][c + n] = (r == c) ? 1.0f : 0.0f; 
        }
    }
    for(int r = 0; r < n; r++) {
        int pivot = r;
        for(int i = r + 1; i < n; i++) {
            if(fabsf(aug[i][r]) > fabsf(aug[pivot][r])) pivot = i;
        }
        if (pivot != r) {
            for(int c = 0; c < 2 * n; c++) {
                float tmp = aug[r][c];
                aug[r][c] = aug[pivot][c];
                aug[pivot][c] = tmp;
            }
        }
        if(fabsf(aug[r][r]) < 1e-6f) return 0; 
        
        float div = aug[r][r];
        for(int c = 0; c < 2 * n; c++) aug[r][c] /= div;
        
        for(int i = 0; i < n; i++) {
            if(i != r) {
                float factor = aug[i][r];
                for(int c = 0; c < 2 * n; c++) aug[i][c] -= factor * aug[r][c];
            }
        }
    }
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++) Out[r * n + c] = aug[r][c + n];
    }
    return 1; 
}

// --- PCHIP INTERPOLATION ---
float interpolate_pchip(float* x, float* y, int n, float xq) {
    if (n == 1) return y[0];
    float h[20] = {0}, delta[20] = {0}, d[20] = {0};
    for (int i = 0; i < n - 1; i++) {
        h[i] = x[i+1] - x[i];
        delta[i] = (y[i+1] - y[i]) / h[i];
    }
    if (n == 2) {
        d[0] = d[1] = delta[0];
    } else {
        for (int i = 1; i < n - 1; i++) {
            if (delta[i-1] * delta[i] <= 0.0f) d[i] = 0.0f;
            else {
                float w1 = 2.0f * h[i] + h[i-1];
                float w2 = h[i] + 2.0f * h[i-1];
                d[i] = (w1 + w2) / ((w1 / delta[i-1]) + (w2 / delta[i]));
            }
        }
        d[0] = ((2.0f * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
        if (d[0] * delta[0] <= 0.0f) d[0] = 0.0f;
        else if (delta[0] * delta[1] <= 0.0f && fabsf(d[0]) > 3.0f * fabsf(delta[0])) d[0] = 3.0f * delta[0];

        int last = n - 1;
        d[last] = ((2.0f * h[last-1] + h[last-2]) * delta[last-1] - h[last-1] * delta[last-2]) / (h[last-1] + h[last-2]);
        if (d[last] * delta[last-1] <= 0.0f) d[last] = 0.0f;
        else if (delta[last-1] * delta[last-2] <= 0.0f && fabsf(d[last]) > 3.0f * fabsf(delta[last-1])) d[last] = 3.0f * delta[last-1];
    }
    int k = 0;
    if (xq >= x[n-1]) k = n - 2; 
    else if (xq <= x[0]) k = 0;     
    else {
        for (int i = 0; i < n - 1; i++) {
            if (xq >= x[i] && xq <= x[i+1]) { k = i; break; }
        }
    }
    float s = xq - x[k], t = s / h[k], t2 = t * t, t3 = t2 * t;
    float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f, h10 = t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2, h11 = t3 - t2;
    return y[k] * h00 + h[k] * d[k] * h10 + y[k+1] * h01 + h[k] * d[k+1] * h11;
}

// --- PSYCHOACOUSTIC BAYESIAN FUNCTIONS ---
void calc_alpha(qCLS_State* qcls, float freq, float* kfreqs, float* phi, float* alpha_out) {
    int num_bounds = qcls->par.Ncategories - 1; 
    int num_anchors = qcls->par.kfreqs;         
    float alpha_tmp[10][4]; 
    float x_nodes[3] = {1.0f, (float)qcls->par.Ncategories / 2.0f, (float)num_bounds};

    for (int ianchor = 0; ianchor < num_anchors; ianchor++) {
        float y_nodes[3] = { phi[ianchor * 3 + 0], phi[ianchor * 3 + 1], phi[ianchor * 3 + 2] };
        for (int p = 0; p < num_bounds; p++) {
            float eval_x = (float)(p + 1); 
            alpha_tmp[p][ianchor] = interpolate_pchip(x_nodes, y_nodes, 3, eval_x);
        }
    }
    for (int p = 0; p < num_bounds; p++) {
        float y_freq_nodes[4];
        for (int ianchor = 0; ianchor < num_anchors; ianchor++) y_freq_nodes[ianchor] = alpha_tmp[p][ianchor];
        alpha_out[p] = interpolate_pchip(kfreqs, y_freq_nodes, num_anchors, freq);
    }
}

void CLS_psycfun(qCLS_State* qcls, float freq, float lev, float* kfreqs, float* phi, float* p_out) {
    float alpha[N_CATEGORIES - 1];
    calc_alpha(qcls, freq, kfreqs, phi, alpha);
    for (int i = 0; i < qcls->par.Ncategories - 1; i++) {
        float p_chance = (float)(qcls->par.Ncategories - 1 - i) / (float)qcls->par.Ncategories;
        float exp_val = expf(-qcls->par.beta * (lev - alpha[i]));
        p_out[i] = (qcls->par.lambda * p_chance) + ((1.0f - qcls->par.lambda) / (1.0f + exp_val));
    }
}

void CLS_jacobian(qCLS_State* qcls, float freq, float lev, float* kfreqs, float* phi, int phi_len, float* H_out) {
    float h = 1e-3f; 
    float ref[N_CATEGORIES - 1];
    CLS_psycfun(qcls, freq, lev, kfreqs, phi, ref);

    for (int iphi = 0; iphi < phi_len; iphi++) {
        float original_phi_val = phi[iphi];
        phi[iphi] = original_phi_val + h;
        float p_perturbed[N_CATEGORIES - 1];
        CLS_psycfun(qcls, freq, lev, kfreqs, phi, p_perturbed);
        phi[iphi] = original_phi_val;
        for (int i = 0; i < qcls->par.Ncategories - 1; i++) {
            int flat_index = (i * phi_len) + iphi;
            H_out[flat_index] = (p_perturbed[i] - ref[i]) / h;
        }
    }
}

// Returns sum(log l), the log-likelihood of the observed response pattern
// under this model, which the caller accumulates into Lmodels.
float Kalman_update(qCLS_State* qcls, float* phi, float* P, float* kfreqs, int phi_len, float freq, float lev, int* r_bool) {
    int n_bounds = qcls->par.Ncategories - 1;
    for (int i = 0; i < phi_len; i++) {
        int diag_idx = i * phi_len + i;
        P[diag_idx] += qcls->par.diffusion * P[diag_idx];
    }
    float mu[10];
    CLS_psycfun(qcls, freq, lev, kfreqs, phi, mu);
    
    float var[100] = {0}; 
    for (int i = 0; i < n_bounds; i++) var[i * n_bounds + i] = mu[i] * (1.0f - mu[i]);

    float H1[120]; 
    CLS_jacobian(qcls, freq, lev, kfreqs, phi, phi_len, H1);
    
    float H1_T[120];
    mat_transpose(H1, n_bounds, phi_len, H1_T);
    
    float Numerator[120]; 
    mat_mult(P, phi_len, phi_len, H1_T, phi_len, n_bounds, Numerator);
    
    float Denom_part[120]; 
    mat_mult(H1, n_bounds, phi_len, P, phi_len, phi_len, Denom_part);
    
    float Denom_full[100]; 
    mat_mult(Denom_part, n_bounds, phi_len, H1_T, phi_len, n_bounds, Denom_full);
    mat_add(Denom_full, var, n_bounds, n_bounds, Denom_full); 
    
    float Denom_inv[100];
    if (!mat_invert(Denom_full, n_bounds, Denom_inv)) return 0.0f; // Singular failsafe

    float K[120]; 
    mat_mult(Numerator, phi_len, n_bounds, Denom_inv, n_bounds, n_bounds, K);

    float r_minus_mu[10];
    for (int i = 0; i < n_bounds; i++) r_minus_mu[i] = (float)r_bool[i] - mu[i];
    
    float K_times_diff[12]; 
    mat_mult(K, phi_len, n_bounds, r_minus_mu, n_bounds, 1, K_times_diff);
    for (int i = 0; i < phi_len; i++) phi[i] += K_times_diff[i];

    float K_H1[144]; 
    mat_mult(K, phi_len, n_bounds, H1, n_bounds, phi_len, K_H1);
    
    float K_H1_P[144];
    mat_mult(K_H1, phi_len, phi_len, P, phi_len, phi_len, K_H1_P);
    for (int i = 0; i < phi_len * phi_len; i++) P[i] -= K_H1_P[i];

    // Likelihood of the observed pattern: mu where the listener responded
    // above the boundary, 1-mu where below. Evaluated at the pre-update mu,
    // matching qCLS.m.
    float loglik = 0.0f;
    for (int i = 0; i < n_bounds; i++) {
        float l = r_bool[i] ? mu[i] : (1.0f - mu[i]);
        if (l < 1e-6f) l = 1e-6f;
        loglik += logf(l);
    }
    return loglik;
}

// Prior for phi conditioned on the audiogram.
//
// thr is 10 thresholds at 250 500 750 1000 1500 2000 3000 4000 6000 8000 Hz,
// NaN where not measured. units_are_hl selects dB HL, otherwise dB SPL.
//
// UNITS ARE NOT OPTIONAL. CUK was fitted against thresholds in dB SPL. Passing
// dB HL unconverted shifts the prior by the RETSPL, 30 dB at 250 Hz, which is
// larger than the residual scatter the prior is built on. It would not fail
// loudly, it would quietly bias the low frequencies.
//
// The stock prior is [40 90 110] +/- 10 dB at every frequency, the same for a
// listener with normal hearing and one with a 70 dB loss. On 148 CLS2023
// listeners replaying recorded responses, conditioning on the audiogram is
// worth 0.396 dB, SE 0.116, t = 3.4, better in 89 of 148.
void qcls_audiogram_prior(float* thr, int units_are_hl) {
    if (!qcls_par_ready) init_bayesian_state();

    for (int f = 0; f < N_FREQS; f++) {
        float t = thr ? thr[f] : NAN;
        if (units_are_hl && !isnan(t)) t += RETSPL_TABLE[retspl_set][f];

        for (int j = 0; j < 3; j++) {
            float mu, sd;
            if (!isnan(t)) {
                mu = CUK[f][j][0] + CUK[f][j][1] * t;
                sd = CUK[f][j][2];
            } else {
                // Fall back per frequency to that boundary's cohort
                // distribution: the audiogram integrated out, not guessed.
                mu = CUK0[f][j][0];
                sd = CUK0[f][j][1];
            }
            if (mu < -5.0f) mu = -5.0f;
            if (mu > 120.0f) mu = 120.0f;
            global_qcls_state.par.phi_prior_mu[f][j]  = mu;
            global_qcls_state.par.phi_prior_std[f][j] = sd;
        }
        // Keep the three anchors ordered and separated.
        float* pm = global_qcls_state.par.phi_prior_mu[f];
        if (pm[1] < pm[0] + 5.0f) pm[1] = pm[0] + 5.0f;
        if (pm[2] < pm[1] + 5.0f) pm[2] = pm[1] + 5.0f;
    }

    qcls_seed_models();
}

// Map a stimulus frequency in Hz to a continuous BAP band index in [1,N_FREQS].
// calc_alpha works in band index, but the JavaScript layer passes frequencies
// in Hz, so the two have to be reconciled before any Kalman update. The band
// centers are the first N_FREQS audiometric frequencies, the same set CUK,
// get_loudness_boundaries and the MATLAB catalog use. calc_alpha interpolates
// continuously, so a fractional index is meaningful and no rounding is needed.
static float bap_freq_index(float f_hz) {
    if (f_hz <= BAND_HZ[0]) return 1.0f;
    if (f_hz >= BAND_HZ[N_FREQS-1]) return (float)N_FREQS;
    for (int i = 0; i < N_FREQS - 1; i++) {
        if (f_hz <= BAND_HZ[i+1]) {
            float t = (log10f(f_hz) - log10f(BAND_HZ[i])) /
                      (log10f(BAND_HZ[i+1]) - log10f(BAND_HZ[i]));
            return (float)(i + 1) + t;
        }
    }
    return (float)N_FREQS;
}

// alpha for every band under a given phi, laid out freq-major to match
// reshape(.., Ncategories-1, Nfreqs) in qCLS.m.
static void model_alpha_all(qCLS_State* q, int m, float* phi, float* out) {
    for (int f = 0; f < N_FREQS; f++) {
        float a[10];
        calc_alpha(q, (float)(f + 1), q->mkfreqs[m], phi, a);
        for (int b = 0; b < 10; b++) out[f * 10 + b] = a[b];
    }
}

// The boundaries this procedure reports and their standard deviation,
// averaged over the candidate models rather than taken from the single most
// likely one. out_mu, out_sd and out_muMAP are each N_FREQS*10 floats;
// out_sd and out_muMAP may be null.
//
// On 148 listeners the between-model standard deviation is 2.18 dB and the
// mixture mean is 0.22 to 0.25 dB more accurate than the mode, t = 7.2 with
// the shipped prior and 9.1 with an audiogram prior.
void qcls_report(float* out_mu, float* out_sd, float* out_muMAP) {
    static float A[N_MODELS][N_FREQS * 10];
    static float Vm[N_MODELS][N_FREQS * 10];
    static float J[N_FREQS * 10][PHI_LEN];
    float w[N_MODELS];
    qCLS_State* q = &global_qcls_state;
    const int NB = N_FREQS * 10;
    const float h = 1e-3f;

    float Lmax = q->Lmodels[0];
    for (int m = 1; m < N_MODELS; m++) if (q->Lmodels[m] > Lmax) Lmax = q->Lmodels[m];
    float wsum = 0.0f;
    for (int m = 0; m < N_MODELS; m++) { w[m] = expf(q->Lmodels[m] - Lmax); wsum += w[m]; }
    for (int m = 0; m < N_MODELS; m++) w[m] /= wsum;

    for (int m = 0; m < N_MODELS; m++) {
        model_alpha_all(q, m, q->phi[m], A[m]);
        for (int j = 0; j < PHI_LEN; j++) {
            float p2[PHI_LEN], a2[N_FREQS * 10];
            for (int k = 0; k < PHI_LEN; k++) p2[k] = q->phi[m][k];
            p2[j] += h;
            model_alpha_all(q, m, p2, a2);
            for (int i = 0; i < NB; i++) J[i][j] = (a2[i] - A[m][i]) / h;
        }
        for (int i = 0; i < NB; i++) {
            float s = 0.0f;
            for (int a = 0; a < PHI_LEN; a++) {
                float jp = 0.0f;
                for (int b = 0; b < PHI_LEN; b++) jp += J[i][b] * q->P[m][b * PHI_LEN + a];
                s += jp * J[i][a];
            }
            Vm[m][i] = s;
        }
    }

    int best = 0;
    for (int m = 1; m < N_MODELS; m++) if (q->Lmodels[m] > q->Lmodels[best]) best = m;

    for (int i = 0; i < NB; i++) {
        float mu = 0.0f;
        for (int m = 0; m < N_MODELS; m++) mu += w[m] * A[m][i];
        out_mu[i] = mu;
        if (out_sd) {
            float V = 0.0f;
            for (int m = 0; m < N_MODELS; m++) {
                float d = A[m][i] - mu;
                V += w[m] * Vm[m][i] + w[m] * d * d;   // within + between
            }
            out_sd[i] = sqrtf(V > 0.0f ? V : 0.0f);
        }
        if (out_muMAP) out_muMAP[i] = A[best][i];
    }
}

// --- JAVASCRIPT BRIDGES ---
void calculate_bap_next(
    float* history_f, float* history_l, float* history_r, 
    int num_trials, int p1_trials, float min_L, float max_L, 
    float* out_f, float* out_l
) {
    // Reseed the posterior for this replay, but keep whatever prior the
    // caller has set. Calling init_bayesian_state() here would silently
    // discard an audiogram-conditioned prior on every trial.
    if (!qcls_par_ready) init_bayesian_state();
    qcls_seed_models();

    int r_bool[10];

    for (int i = 0; i < num_trials; i++) {
        for (int b = 0; b < 10; b++) {
            float boundary_val = (float)(b + 1) * 5.0f;
            r_bool[b] = (history_r[i] >= boundary_val) ? 1 : 0;
        }
        float fidx = bap_freq_index(history_f[i]);

        for (int m = 0; m < N_MODELS; m++) {
            float ll = Kalman_update(
                &global_qcls_state,
                global_qcls_state.phi[m],
                global_qcls_state.P[m],
                global_qcls_state.mkfreqs[m],
                PHI_LEN,
                fidx,
                history_l[i],
                r_bool
            );
            global_qcls_state.Lmodels[m] =
                global_qcls_state.par.likelihood_exp * global_qcls_state.Lmodels[m] + ll;
        }
        global_qcls_state.trial_n++;
    }

    // Choose the next stimulus from the posterior instead of at random:
    // present the band and level where the mixture is least certain, which
    // is where a trial has the most to tell us. Testing at the boundary's
    // own mean puts the stimulus where that boundary's response is closest
    // to a coin flip. Once a trial lands there its SD falls and the next
    // maximum moves elsewhere, so the rule spreads itself over the run.
    //
    // Previously this returned a uniformly random frequency and level and
    // never consulted the posterior at all, which made the interface's
    // "Bayesian" adaptive-tracking option identical to its "Random" one.
    //
    // On the CLS2023 catalog, posterior-driven selection has not beaten
    // random overall: entropy MEI was neutral and a variance-targeted rule
    // measured 0.696 dB worse, t = -5.7, while being markedly better at the
    // CU5 end, 11.7 dB down to 6.8. Selection stays behind the interface
    // switch so the two arms remain comparable on real listeners.
    {
        static float rmu[N_FREQS * 10], rsd[N_FREQS * 10];
        qcls_report(rmu, rsd, NULL);

        int best = 0;
        for (int i = 1; i < N_FREQS * 10; i++) if (rsd[i] > rsd[best]) best = i;

        *out_f = BAND_HZ[best / 10];

        // A little jitter around the mean, so a run does not keep landing on
        // exactly one level and leave the slope between boundaries unsampled.
        float lev = rmu[best] + (((float)(rand() % 101) / 100.0f) - 0.5f) * 5.0f;
        if (lev < min_L) lev = min_L;
        if (lev > max_L) lev = max_L;
        *out_l = 5.0f * floorf(lev / 5.0f + 0.5f);
    }
}

void generate_stimulus(float duration, float samplingRate, int stimulusType, float bandwidthOctaves, float centerFreq, float* outputBuffer) {
    int num_samples = (int)(duration * samplingRate);
    float ramp_duration = 0.05f;
    int ramp_samples = (int)(ramp_duration * samplingRate);

    float fm = 0.0f;
    if (stimulusType == STIM_SAM_TONE && bandwidthOctaves > 0.0f) {
        float K_sam = powf(2.0f, bandwidthOctaves); 
        fm = centerFreq * (K_sam - 1.0f) / (K_sam + 1.0f);
    }

    float f1 = centerFreq, f2 = centerFreq, f4 = centerFreq, f5 = centerFreq;
    if (stimulusType == STIM_FIVE_TONE && bandwidthOctaves > 0.0f) {
        float k = powf(2.0f, bandwidthOctaves / 4.0f); 
        f1 = centerFreq / (k * k);
        f2 = centerFreq / k;
        f4 = centerFreq * k;
        f5 = centerFreq * (k * k);
    }

    float pure_tone_amp = sqrtf(2.0f); 
    float sam_tone_amp = 1.0f / sqrtf(0.75f); 
    float five_tone_amp = sqrtf(2.0f / 5.0f);

    for (int i = 0; i < num_samples; i++) {
        float t = (float)i / samplingRate;
        float window = 1.0f;

        if (i < ramp_samples) {
            window = 0.5f * (1.0f - cosf(PI * i / ramp_samples));
        } else if (i > num_samples - ramp_samples) {
            window = 0.5f * (1.0f - cosf(PI * (num_samples - i) / ramp_samples));
        }

        float val = 0.0f;
        float carrier = sinf(2.0f * PI * centerFreq * t);

        if (stimulusType == STIM_TONE) {
            val = pure_tone_amp * carrier;
        } 
        else if (stimulusType == STIM_SAM_TONE) {
            float modulator = 1.0f + sinf(2.0f * PI * fm * t);
            val = sam_tone_amp * modulator * carrier;
        }
        else if (stimulusType == STIM_FIVE_TONE) {
            float sum = sinf(2.0f * PI * f1 * t) +
                        sinf(2.0f * PI * f2 * t) +
                        carrier + 
                        sinf(2.0f * PI * f4 * t) +
                        sinf(2.0f * PI * f5 * t);
            val = five_tone_amp * sum;
        }

        outputBuffer[i] = val * window;
    }
}

// --- MLE MCPF ESTIMATION ---
static float mle_history_f[1000];
static float mle_history_l[1000];
static int mle_history_r[1000];
static int mle_num_trials = 0;
float global_estimated_theta[PCA_PARAMS];

void reconstruct_theta(float* w, float* theta_out) {
    for (int i = 0; i < PCA_PARAMS; i++) {
        theta_out[i] = pca_mu[i];
        for (int c = 0; c < PCA_COMPONENTS; c++) {
            theta_out[i] += pca_V[i * PCA_COMPONENTS + c] * w[c];
        }
    }
    // Clamp FAR parameters (200 to 209)
    for (int i = 200; i < 210; i++) {
        if (theta_out[i] < 0.001f) theta_out[i] = 0.001f;
        if (theta_out[i] > 0.999f) theta_out[i] = 0.999f;
    }
}

int freq_hz_to_idx(float f_hz) {
    float frq_list[10] = {250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000};
    int best_idx = 0;
    float min_diff = 100000.0f;
    for (int i=0; i<10; i++) {
        float diff = fabsf(frq_list[i] - f_hz);
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

float compute_mcpf_nll(float* w) {
    float theta[PCA_PARAMS];
    reconstruct_theta(w, theta);
    
    float nll = 0.0f;
    
    for (int t = 0; t < mle_num_trials; t++) {
        int f_idx = freq_hz_to_idx(mle_history_f[t]);
        float spl = mle_history_l[t];
        int r_idx = (mle_history_r[t] / 5) + 1; // 0->1, 50->11
        if (r_idx < 1) r_idx = 1;
        if (r_idx > 11) r_idx = 11;
        
        float far = theta[200 + f_idx];
        float* pa = &theta[f_idx * 20]; 
        
        float cb[10] = {0};
        float md[10] = {0};
        float sum_md = 0;
        for (int k = 0; k < 10; k++) {
            sum_md += pa[10 + k];
            md[k] = sum_md;
        }
        
        float dfa = far / 11.0f;
        for (int k = 0; k < 10; k++) {
            float cf1 = -md[k] * pa[k];
            float cf2 = pa[k];
            float pf = 1.0f / (1.0f + expf(-(cf1 + cf2 * spl)));
            cb[k] = pf * (1.0f - far) + (float)(k + 1) * dfa;
        }
        
        // Eliminate cross-overs
        for (int it = 0; it < 20; it++) {
            for (int j = 0; j < 9; j++) {
                float pd = cb[j+1] - cb[j];
                if (pd < (dfa / 2.0f)) {
                    cb[j] = cb[j] + pd/2.0f - dfa/2.0f;
                    cb[j+1] = cb[j] + dfa;
                }
            }
        }
        
        // Top-down
        float cbmx = 1.0f - dfa;
        for (int j = 9; j >= 0; j--) {
            if (cb[j] > cbmx) cb[j] = cbmx;
            cbmx = cb[j] - dfa;
        }
        
        // Bottom-up
        float cbmn = dfa;
        for (int j = 0; j < 10; j++) {
            if (cb[j] < cbmn) cb[j] = cbmn;
            cbmn = cb[j] + dfa;
        }
        
        // Convert to category probabilities
        float pcat[11];
        pcat[0] = cb[0]; 
        for (int k=1; k<10; k++) {
            pcat[k] = cb[k] - cb[k-1];
        }
        pcat[10] = 1.0f - cb[9];
        
        float sum_pcat = 0;
        for (int k=0; k<11; k++) {
            if (pcat[k] < 1e-6f) pcat[k] = 1e-6f;
            sum_pcat += pcat[k];
        }
        
        float prob_response = pcat[r_idx - 1] / sum_pcat;
        nll -= logf(prob_response);
    }
    return nll;
}

void run_nelder_mead(float* w_opt) {
    int N = PCA_COMPONENTS;
    float simplex[11][10];
    float f_val[11];
    
    // Initialize simplex
    for(int i=0; i<N+1; i++) {
        for(int j=0; j<N; j++) {
            if (i == 0) simplex[i][j] = 0.0f; // origin
            else {
                simplex[i][j] = (i - 1 == j) ? 0.5f : 0.0f; // step size 0.5
            }
        }
        f_val[i] = compute_mcpf_nll(simplex[i]);
    }
    
    int max_iters = 1000;
    float alpha = 1.0f, gamma = 2.0f, rho = 0.5f, sigma = 0.5f;
    
    for(int iter = 0; iter < max_iters; iter++) {
        int best = 0, worst = 0, second_worst = 0;
        for(int i=1; i<N+1; i++) {
            if(f_val[i] < f_val[best]) best = i;
            if(f_val[i] > f_val[worst]) worst = i;
        }
        for(int i=0; i<N+1; i++) {
            if(i != worst && (i == best || f_val[i] > f_val[second_worst])) {
                second_worst = i;
            }
        }
        
        float centroid[10] = {0};
        for(int i=0; i<N+1; i++) {
            if(i != worst) {
                for(int j=0; j<N; j++) centroid[j] += simplex[i][j];
            }
        }
        for(int j=0; j<N; j++) centroid[j] /= N;
        
        float xr[10];
        for(int j=0; j<N; j++) xr[j] = centroid[j] + alpha * (centroid[j] - simplex[worst][j]);
        float fr = compute_mcpf_nll(xr);
        
        if (fr >= f_val[best] && fr < f_val[second_worst]) {
            for(int j=0; j<N; j++) simplex[worst][j] = xr[j];
            f_val[worst] = fr;
        } else if (fr < f_val[best]) {
            float xe[10];
            for(int j=0; j<N; j++) xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
            float fe = compute_mcpf_nll(xe);
            if (fe < fr) {
                for(int j=0; j<N; j++) simplex[worst][j] = xe[j];
                f_val[worst] = fe;
            } else {
                for(int j=0; j<N; j++) simplex[worst][j] = xr[j];
                f_val[worst] = fr;
            }
        } else {
            float xc[10];
            for(int j=0; j<N; j++) xc[j] = centroid[j] + rho * (simplex[worst][j] - centroid[j]);
            float fc = compute_mcpf_nll(xc);
            if (fc < f_val[worst]) {
                for(int j=0; j<N; j++) simplex[worst][j] = xc[j];
                f_val[worst] = fc;
            } else {
                for(int i=0; i<N+1; i++) {
                    if (i != best) {
                        for(int j=0; j<N; j++) {
                            simplex[i][j] = simplex[best][j] + sigma * (simplex[i][j] - simplex[best][j]);
                        }
                        f_val[i] = compute_mcpf_nll(simplex[i]);
                    }
                }
            }
        }
        
        if (fabsf(f_val[worst] - f_val[best]) < 1e-4f) break;
    }
    
    int best = 0;
    for(int i=1; i<N+1; i++) {
        if(f_val[i] < f_val[best]) best = i;
    }
    for(int j=0; j<N; j++) w_opt[j] = simplex[best][j];
}

void estimate_mcpf(float* history_f, float* history_l, float* history_r, int num_trials) {
    mle_num_trials = num_trials;
    for (int i=0; i<num_trials; i++) {
        mle_history_f[i] = history_f[i];
        mle_history_l[i] = history_l[i];
        mle_history_r[i] = (int)history_r[i];
    }
    
    float w_opt[10] = {0};
    run_nelder_mead(w_opt);
    
    reconstruct_theta(w_opt, global_estimated_theta);
}

// Mean of the 100 psychometric slopes, in 1/dB. Negative by convention.
//
// The stored values are the slopes themselves, not their logs. mcpf_bnd.m
// returns slp = -exp(logs5 + (logs50-logs5)*u), and the catalog stores that
// result directly; checked against coeff_tenfrq_v3, the stored coefficients
// match mcpf_bnd's slopes to 0.00000. They are negative because the
// psychometric function is written 1/(1+exp(-slp*(lev-md))), which with a
// negative slope is high below the boundary and low above it.
//
// An earlier version of this function exponentiated the mean, on the
// mistaken reading that a negative value had to be a log. That turned a
// correct slope into a meaningless positive number. Do not reintroduce it.
//
// This is the steepness of a category boundary, not the rate at which
// loudness grows, so it is not in CU/dB. The interface reports loudness
// growth from the CU5 and CU50 contours instead, which is a different
// quantity and genuinely in CU/dB.
float get_average_slope() {
    float sum_slope = 0;
    for (int i=0; i<10; i++) {
        for (int k=0; k<10; k++) {
            sum_slope += global_estimated_theta[i*20 + k];
        }
    }
    return sum_slope / 100.0f;
}

// Mean of the fitted false-alarm rates.
//
// This cannot serve as a per-listener measure and should not be displayed as
// one. The rate is read off the PCA projection rather than the data, and all
// ten of its parameters carry the identical loading of 0.00131, so a
// two-sigma excursion on every component moves the result only from 0.086 to
// 0.096. Listeners in the CLS2023 catalog span 0.005 to 0.270 with a
// between-listener SD of 0.037, and the parameter was fit as a single value
// per listener, invariant across frequency, which is why the ten slots move
// together. The interface measures the rate from the trials instead.
float get_average_far() {
    float sum_far = 0;
    for (int i=200; i<210; i++) {
        sum_far += global_estimated_theta[i];
    }
    return sum_far / 10.0f;
}

void get_loudness_boundaries(float freq, float* out_boundaries) {
    float frq_list[10] = {250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000};
    int lower_idx = 0, upper_idx = 0;
    
    if (freq <= frq_list[0]) { lower_idx = 0; upper_idx = 0; }
    else if (freq >= frq_list[9]) { lower_idx = 9; upper_idx = 9; }
    else {
        for (int i=0; i<9; i++) {
            if (freq >= frq_list[i] && freq <= frq_list[i+1]) {
                lower_idx = i;
                upper_idx = i+1;
                break;
            }
        }
    }
    
    float t = 0.0f;
    if (upper_idx != lower_idx) {
        t = (log10f(freq) - log10f(frq_list[lower_idx])) / (log10f(frq_list[upper_idx]) - log10f(frq_list[lower_idx]));
    }
    
    float* pa_lower = &global_estimated_theta[lower_idx * 20];
    float* pa_upper = &global_estimated_theta[upper_idx * 20];
    
    float sum_lower = 0, sum_upper = 0;
    for (int k = 0; k < 10; k++) {
        sum_lower += pa_lower[10 + k];
        sum_upper += pa_upper[10 + k];
        out_boundaries[k] = sum_lower + t * (sum_upper - sum_lower);
    }
}

