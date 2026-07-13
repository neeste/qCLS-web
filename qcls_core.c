#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "pca_model.h"

#define PI 3.14159265358979323846f

// --- BAP CONSTANTS ---
#define N_CATEGORIES 11
#define N_FREQS 9
#define K_FREQS 4

typedef enum { STIM_TONE = 0, STIM_SAM_TONE, STIM_FIVE_TONE } StimulusType;

// --- STRUCTS ---
typedef struct {
    int Ncategories;
    int Nfreqs;
    int kfreqs;
    float beta;
    float lambda;
    float phi_prior_mu[N_FREQS][3]; 
    float phi_prior_std[3];
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
    float phi[N_FREQS * 3]; 
    float P[(N_FREQS * 3) * (N_FREQS * 3)]; 
} qCLS_State;

// Global instance of the test state
qCLS_State global_qcls_state;

// --- FORWARD DECLARATIONS ---
void Kalman_update(qCLS_State* qcls, float* phi, float* P, float* kfreqs, int phi_len, float freq, float lev, int* r_bool);
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
    
    global_qcls_state.par.phi_prior_std[0] = 10.0f;
    global_qcls_state.par.phi_prior_std[1] = 10.0f;
    global_qcls_state.par.phi_prior_std[2] = 10.0f;
    
    global_qcls_state.par.x_lim[0][0] = 1.0f;
    global_qcls_state.par.x_lim[0][1] = 0.0f;
    global_qcls_state.par.x_lim[1][0] = (float)N_FREQS;
    global_qcls_state.par.x_lim[1][1] = 110.0f;
    
    global_qcls_state.par.Lclearance = 0.1f;
    global_qcls_state.par.fclearance = 0.1f;
    global_qcls_state.par.likelihood_exp = 0.95f;
    global_qcls_state.par.diffusion = 0.01f;
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

void Kalman_update(qCLS_State* qcls, float* phi, float* P, float* kfreqs, int phi_len, float freq, float lev, int* r_bool) {
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
    if (!mat_invert(Denom_full, n_bounds, Denom_inv)) return; // Singular failsafe
    
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
}

// --- JAVASCRIPT BRIDGES ---
void calculate_bap_next(
    float* history_f, float* history_l, float* history_r, 
    int num_trials, int p1_trials, float min_L, float max_L, 
    float* out_f, float* out_l
) {
    init_bayesian_state();
    float current_kfreqs[4] = {1.0f, 3.0f, 6.0f, 9.0f}; 
    int r_bool[10];

    for (int i = 0; i < num_trials; i++) {
        for (int b = 0; b < 10; b++) {
            float boundary_val = (float)(b + 1) * 5.0f; 
            r_bool[b] = (history_r[i] >= boundary_val) ? 1 : 0;
        }

        Kalman_update(
            &global_qcls_state, 
            global_qcls_state.phi, 
            global_qcls_state.P, 
            current_kfreqs, 
            12, 
            history_f[i], 
            history_l[i], 
            r_bool
        );
    }

    int freq_candidate_idx = (rand() % N_FREQS) + 1;
    float logMin = log10f(250.0f);
    float logMax = log10f(6000.0f);
    float freq_fraction = (float)(freq_candidate_idx - 1) / (float)(N_FREQS - 1);
    *out_f = powf(10.0f, logMin + freq_fraction * (logMax - logMin));

    int min_step = (int)(min_L / 5.0f);
    int max_step = (int)(max_L / 5.0f);
    
    if (max_step <= min_step) {
        *out_l = min_L; 
    } else {
        int random_step = min_step + (rand() % (max_step - min_step + 1));
        *out_l = (float)(random_step * 5);
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

float get_average_slope() {
    float sum_slope = 0;
    for (int i=0; i<10; i++) {
        for (int k=0; k<10; k++) {
            sum_slope += global_estimated_theta[i*20 + k];
        }
    }
    return sum_slope / 100.0f;
}

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

