#!/usr/bin/env python3
"""
exp_gp_vs_bap.py

Self-contained benchmark comparing the qCLS Bayesian Adaptive Procedure (BAP)
Extended Kalman Filter (EKF) and Post-hoc PCA MLE against a Laplace Gaussian
Process (GP) tracker and 3-Pass 210-Parameter MCPF recovery.

Does not modify any web-app files. Compiles ../qcls_core.c to a temporary shared
library in /tmp/libqcls.dylib for fast headless C evaluation, and loads listener
ground truth from ../sim/mcpf_parameters_v2.csv (or simulated_listeners_v2.json).
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SIM_DIR = ROOT_DIR / "sim"
sys.path.insert(0, str(SIM_DIR))

from train_pca_model import load_reference_model

FREQS_HZ = np.array([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000], dtype=float)
BANDS_HZ = [250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0]
N_BOUNDARIES = 10


def load_c_library() -> ctypes.CDLL:
    dylib_path = Path("/tmp/libqcls.dylib")
    c_src = ROOT_DIR / "qcls_core.c"
    if not dylib_path.exists() or dylib_path.stat().st_mtime < c_src.stat().st_mtime:
        subprocess.run(
            ["cc", "-O3", "-shared", "-fPIC", str(c_src), "-lm", "-o", str(dylib_path)],
            check=True,
        )
    lib = ctypes.CDLL(str(dylib_path))
    lib.init_bayesian_state.argtypes = []
    lib.init_bayesian_state.restype = None
    lib.qcls_update_trial.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.qcls_update_trial.restype = None
    lib.qcls_select_bayesian_next.argtypes = [
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
    ]
    lib.qcls_select_bayesian_next.restype = None
    lib.qcls_select_isophon_next.argtypes = [
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
    ]
    lib.qcls_select_isophon_next.restype = None
    lib.qcls_report.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.qcls_report.restype = None
    lib.qcls_pca_fit_report.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.qcls_pca_fit_report.restype = ctypes.c_int
    return lib


class JSONVirtualListener:
    """Virtual listener backed by either mcpf_parameters_v2.csv or simulated_listeners_v2.json."""

    def __init__(self, listener_id: str, coeffs_20x10: np.ndarray, far_10: np.ndarray, rng: np.random.Generator):
        self.listener_id = listener_id
        self.rng = rng
        self.freqs_hz = FREQS_HZ
        self.levels = np.arange(0.0, 120.0001, 0.1)
        self.slopes = np.abs(coeffs_20x10[:10, :].T)            # (10 freqs, 10 bounds)
        self.bounds = np.cumsum(coeffs_20x10[10:20, :], axis=0) # (10 bounds, 10 freqs)
        self.far = far_10.copy()                                # (10,)
        self.pf = self._make_mcpf(coeffs_20x10, far_10)

    def _make_mcpf(self, coefficients: np.ndarray, false_alarm: np.ndarray) -> np.ndarray:
        nfreqs = len(self.freqs_hz)
        pf = np.zeros((nfreqs, len(self.levels), N_BOUNDARIES))
        dfa = false_alarm / 11.0
        midpoint = np.cumsum(coefficients[10:20], axis=0)

        for frequency in range(nfreqs):
            for boundary in range(N_BOUNDARIES):
                coefficient = -abs(coefficients[boundary, frequency])
                intercept = -midpoint[boundary, frequency] * coefficient
                logit = np.clip(intercept + coefficient * self.levels, -700.0, 700.0)
                logistic = 1.0 / (1.0 + np.exp(-logit))
                pf[frequency, :, boundary] = (
                    logistic * (1.0 - false_alarm[frequency]) + boundary * dfa[frequency]
                )

        for _ in range(20):
            differences = np.diff(pf, axis=2)
            for boundary in range(N_BOUNDARIES - 1):
                mask = differences[:, :, boundary] < (dfa[:, None] / 2.0)
                adjustment = differences[:, :, boundary] / 2.0 - dfa[:, None] / 2.0
                pf[:, :, boundary] = np.where(
                    mask, pf[:, :, boundary] + adjustment, pf[:, :, boundary]
                )
                pf[:, :, boundary + 1] = np.where(
                    mask, pf[:, :, boundary] + dfa[:, None], pf[:, :, boundary + 1]
                )

        upper = (1.0 - dfa)[:, None]
        for boundary in range(N_BOUNDARIES - 1, -1, -1):
            mask = pf[:, :, boundary] > upper
            pf[:, :, boundary] = np.where(mask, upper, pf[:, :, boundary])
            upper = pf[:, :, boundary] - dfa[:, None]

        lower = dfa[:, None]
        for boundary in range(N_BOUNDARIES):
            mask = pf[:, :, boundary] < lower
            pf[:, :, boundary] = np.where(mask, lower, pf[:, :, boundary])
            lower = pf[:, :, boundary] + dfa[:, None]
        return pf

    def respond(self, freq_hz: float, level_db: float) -> int:
        f_idx = int(np.argmin(np.abs(self.freqs_hz - freq_hz)))
        l_idx = int(np.argmin(np.abs(self.levels - level_db)))
        cdf = self.pf[f_idx, l_idx, :]
        r = self.rng.random()
        cat = 11 if r > cdf[-1] else int(np.searchsorted(cdf, r, side="right")) + 1
        return cat - 1


def load_listener_catalog(json_path: Path | None = None):
    if json_path is None:
        json_path = THIS_DIR / "simulated_listeners_v2.json"
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    catalog = []
    for item in raw:
        theta = np.array(item["theta"], dtype=float)
        coeffs = theta[:200].reshape(10, 20).T # (20, 10)
        # Ensure positive intervals
        coeffs[11:20, :] = np.maximum(coeffs[11:20, :], 1e-4)
        far = np.clip(theta[200:210], 0.005, 0.25)
        catalog.append((item["id"], coeffs, far))
    return catalog


# Build GP / Empirical-Covariance Kernel from pca_model.h + smooth 2D residual kernel
ref_pca = load_reference_model(ROOT_DIR / "pca_model.h", 5)
PCA_MU = ref_pca["pca_mu"].reshape(10, 10)
PCA_D = ref_pca["pca_d"].reshape(10, 10)
PCA_V = ref_pca["pca_V"].reshape(100, 5)
PCA_S_STD = ref_pca["pca_score_std"]

PCA_W = (PCA_D.reshape(100, 1) * PCA_V) * PCA_S_STD.reshape(1, 5)
PCA_MU_FLAT = PCA_MU.reshape(100)

log_freqs = np.log2(FREQS_HZ)
K_res = np.zeros((100, 100))
for f1 in range(10):
    for b1 in range(10):
        idx1 = f1 * 10 + b1
        for f2 in range(10):
            for b2 in range(10):
                idx2 = f2 * 10 + b2
                df_val = (log_freqs[f1] - log_freqs[f2]) / 1.0
                db_val = (b1 - b2) / 2.5
                amp1 = 5.0 - 0.3 * b1
                amp2 = 5.0 - 0.3 * b2
                K_res[idx1, idx2] = amp1 * amp2 * np.exp(-0.5 * (df_val**2 + db_val**2))

GP_MU_0 = PCA_MU_FLAT.copy()
GP_SIGMA_0 = PCA_W @ PCA_W.T + K_res + 0.25 * np.eye(100)
GP_PREC_0 = np.linalg.inv(GP_SIGMA_0)
P_CHANCE = np.array([(11 - 1 - k) / 11.0 for k in range(10)])

K_slope_res = np.zeros((100, 100))
for f1 in range(10):
    for b1 in range(10):
        idx1 = f1 * 10 + b1
        for f2 in range(10):
            for b2 in range(10):
                idx2 = f2 * 10 + b2
                df_val = (log_freqs[f1] - log_freqs[f2]) / 1.5
                db_val = (b1 - b2) / 2.5
                K_slope_res[idx1, idx2] = (0.15 ** 2) * np.exp(-0.5 * (df_val**2 + db_val**2))


def build_slope_gp_prior(train_log_slopes: np.ndarray, n_comp: int = 5):
    mu_gamma = np.mean(train_log_slopes, axis=0)
    centered = train_log_slopes - mu_gamma
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    V_k = Vt[:n_comp, :].T
    scores_std = S[:n_comp] / np.sqrt(train_log_slopes.shape[0] - 1)
    W_gamma = V_k * scores_std.reshape(1, n_comp)
    Sigma_gamma = W_gamma @ W_gamma.T + K_slope_res + 0.01 * np.eye(100)
    Prec_gamma = np.linalg.inv(Sigma_gamma)
    return mu_gamma, Sigma_gamma, Prec_gamma


class LaplaceGPTracker:
    """Unified Laplace Gaussian Process tracker with 3-Pass 210-parameter MCPF recovery."""

    def __init__(self, mu_gamma: np.ndarray | None = None, Prec_gamma: np.ndarray | None = None,
                 beta_init: float = 0.35, lam_init: float = 0.02):
        self.beta_init = beta_init
        self.lam_init = lam_init
        self.mu_alpha = GP_MU_0.copy()
        self.Sigma_alpha = GP_SIGMA_0.copy()
        self.mu_gamma_0 = mu_gamma.copy() if mu_gamma is not None else np.full(100, np.log(beta_init))
        self.Prec_gamma_0 = Prec_gamma.copy() if Prec_gamma is not None else np.eye(100) * 10.0
        self.hist_f_idx: list[int] = []
        self.hist_l: list[float] = []
        self.hist_hits: list[np.ndarray] = []

    def update(self, freq_hz: float, level_db: float, resp_cu: float) -> None:
        f_idx = int(np.argmin(np.abs(FREQS_HZ - freq_hz)))
        hits = np.array([1.0 if resp_cu >= (k + 1) * 5 else 0.0 for k in range(10)])
        self.hist_f_idx.append(f_idx)
        self.hist_l.append(level_db)
        self.hist_hits.append(hits)

        alpha = self.mu_alpha.copy()
        beta = self.beta_init
        lam = self.lam_init

        for _ in range(3):
            grad = -GP_PREC_0 @ (alpha - GP_MU_0)
            hess_diag = np.zeros(100)
            for t in range(len(self.hist_l)):
                fi = self.hist_f_idx[t]
                lev = self.hist_l[t]
                y = self.hist_hits[t]
                a_f = alpha[fi * 10 : (fi + 1) * 10]
                z = np.clip(beta * (lev - a_f), -30.0, 30.0)
                s = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(lam * P_CHANCE + (1.0 - lam) * s, 1e-6, 1.0 - 1e-6)
                dp_da = -(1.0 - lam) * beta * s * (1.0 - s)
                grad[fi * 10 : (fi + 1) * 10] += ((y - p) / (p * (1.0 - p))) * dp_da
                hess_diag[fi * 10 : (fi + 1) * 10] += (dp_da ** 2) / (p * (1.0 - p))
            H = GP_PREC_0 + np.diag(hess_diag)
            step = np.linalg.solve(H, grad)
            alpha += step
            if np.max(np.abs(step)) < 1e-2:
                break
        self.mu_alpha = alpha
        self.Sigma_alpha = np.linalg.inv(GP_PREC_0 + np.diag(hess_diag))

    def select_next(self, min_l: float, max_l: float, rng: np.random.Generator) -> tuple[float, float]:
        l_start = max(0.0, np.ceil(min_l / 5.0) * 5.0)
        l_end = min(105.0, np.floor(max_l / 5.0) * 5.0)
        cand_levels = (
            np.arange(l_start, l_end + 1.0, 5.0)
            if l_end >= l_start
            else np.array([min(max_l, max(min_l, 50.0))])
        )
        best_score, best_f, best_l = -1.0, 1000.0, 50.0
        beta, lam = self.beta_init, self.lam_init
        for fi in range(9):
            a_f = self.mu_alpha[fi * 10 : (fi + 1) * 10]
            Sigma_col = self.Sigma_alpha[:, fi * 10 : (fi + 1) * 10]
            col_norms_sq = np.sum(Sigma_col ** 2, axis=0)
            diag_var = np.diag(self.Sigma_alpha)[fi * 10 : (fi + 1) * 10]
            for lev in cand_levels:
                z = np.clip(beta * (lev - a_f), -30.0, 30.0)
                s = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(lam * P_CHANCE + (1.0 - lam) * s, 1e-6, 1.0 - 1e-6)
                dp_da = -(1.0 - lam) * beta * s * (1.0 - s)
                w = (dp_da ** 2) / (p * (1.0 - p))
                ivr = np.sum((w / (1.0 + w * diag_var)) * col_norms_sq)
                if ivr > best_score:
                    best_score, best_f, best_l = ivr, BANDS_HZ[fi], lev
        jitter = (rng.random() - 0.5) * 4.0
        return best_f, float(np.floor(np.clip(best_l + jitter, min_l, max_l) + 0.5))

    def get_boundaries(self) -> np.ndarray:
        """Return (10 bounds x 10 freqs) matrix with monotonic spacing enforced."""
        mat = self.mu_alpha.reshape(10, 10).copy()
        for f in range(10):
            for b in range(1, 10):
                if mat[f, b] < mat[f, b - 1] + 0.5:
                    mat[f, b] = mat[f, b - 1] + 0.5
        return mat.T

    def fit_pass2_slopes(self, alpha_freq_major: np.ndarray) -> np.ndarray:
        """Pass 2: Fit 100 log-slopes given locked Pass-1 boundary means."""
        alpha_flat = alpha_freq_major.reshape(100)
        gamma = self.mu_gamma_0.copy()
        lam = self.lam_init
        for _ in range(6):
            grad = -self.Prec_gamma_0 @ (gamma - self.mu_gamma_0)
            hess_diag = np.zeros(100)
            for t in range(len(self.hist_l)):
                fi = self.hist_f_idx[t]
                lev = self.hist_l[t]
                y = self.hist_hits[t]
                a_f = alpha_flat[fi * 10 : (fi + 1) * 10]
                g_f = gamma[fi * 10 : (fi + 1) * 10]
                b_f = np.exp(np.clip(g_f, -3.0, 2.0))
                diff = lev - a_f
                z = np.clip(b_f * diff, -30.0, 30.0)
                s = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(lam * P_CHANCE + (1.0 - lam) * s, 1e-6, 1.0 - 1e-6)
                dp_dg = (1.0 - lam) * s * (1.0 - s) * b_f * diff
                grad[fi * 10 : (fi + 1) * 10] += ((y - p) / (p * (1.0 - p))) * dp_dg
                hess_diag[fi * 10 : (fi + 1) * 10] += (dp_dg ** 2) / (p * (1.0 - p))
            H = self.Prec_gamma_0 + np.diag(hess_diag)
            step = np.linalg.solve(H, grad)
            gamma += 0.8 * step
            if np.max(np.abs(step)) < 1e-3:
                break
        return np.exp(np.clip(gamma, -3.0, 2.0)).reshape(10, 10)

    def fit_pass3_fa(self, alpha_freq_major: np.ndarray, beta_freq_major: np.ndarray) -> float:
        """Pass 3: Estimate global False Alarm rate from responses outside 95% CI."""
        logit_975 = np.log(0.975 / 0.025)
        num_unexpected = 0.0
        denom_fa_weight = 0.0
        for t in range(len(self.hist_l)):
            fi = self.hist_f_idx[t]
            lev = self.hist_l[t]
            y = self.hist_hits[t]
            a_f = alpha_freq_major[fi]
            b_f = beta_freq_major[fi]
            half_width = logit_975 / b_f
            for k in range(10):
                z = b_f[k] * (lev - a_f[k])
                s_k = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
                if lev < a_f[k] - half_width[k]:
                    num_unexpected += max(0.0, y[k] - s_k)
                    denom_fa_weight += (10.0 - k) / 11.0
                elif lev > a_f[k] + half_width[k]:
                    num_unexpected += max(0.0, (1.0 - y[k]) - (1.0 - s_k))
                    denom_fa_weight += (k + 1.0) / 11.0
        prior_pseudocount = 15.0
        prior_fa = 0.005
        global_fa = (num_unexpected + prior_pseudocount * prior_fa) / (denom_fa_weight + prior_pseudocount)
        return float(np.clip(global_fa, 0.005, 0.20))


def main() -> None:
    lib = load_c_library()
    catalog = load_listener_catalog()
    n_test = min(50, len(catalog))
    print(f"Loaded {len(catalog)} virtual listeners; running benchmark on {n_test} listeners...")

    all_log_slopes = np.array([
        np.log(np.clip(np.abs(coeffs[:10, :].T.reshape(100)), 0.05, 5.0))
        for _, coeffs, _ in catalog
    ])

    mae_bap_bayesian = []
    mae_bap_fixed = []
    mae_bap_isophon = []
    mae_gp_active = []

    true_mean_slopes, est_mean_slopes = [], []
    true_fas, est_fas = [], []

    for idx in range(n_test):
        lid, coeffs, far = catalog[idx]
        vl = JSONVirtualListener(lid, coeffs, far, np.random.default_rng(1000 + idx))
        truth = vl.bounds

        train_mask = np.ones(len(catalog), dtype=bool)
        train_mask[idx] = False
        mu_gamma, _, Prec_gamma = build_slope_gp_prior(all_log_slopes[train_mask], n_comp=5)

        # Run Laplace GP Active (100 trials) + Pass 2 + Pass 3
        gp = LaplaceGPTracker(mu_gamma, Prec_gamma)
        rng = np.random.default_rng(2000 + idx)
        cur_f, cur_l, start_l = 1000.0, 50.0, 50.0
        min_l, ceiling, max_l = -10.0, 100.0, 100.0
        phase, p1_dir, start_resp = "P1", -1, None

        for t in range(100):
            cat = vl.respond(cur_f, cur_l)
            resp_cu = float(cat * 5)
            gp.update(cur_f, cur_l, resp_cu)
            if t == 99:
                break
            if phase == "P1":
                if start_resp is None:
                    start_resp = resp_cu
                if p1_dir == -1:
                    if resp_cu == 0 or cur_l <= 0:
                        min_l = cur_l
                        p1_dir = 1
                        if start_resp == 50:
                            max_l = max(min_l + 10.0, start_l - 5.0)
                            phase = "P2"
                        else:
                            cur_l = start_l + (5.0 if start_resp >= 35 else 10.0)
                    else:
                        cur_l -= 10.0
                else:
                    if resp_cu == 50 or cur_l >= ceiling:
                        max_l = max(min_l + 10.0, cur_l - 5.0)
                        phase = "P2"
                    else:
                        cur_l += (5.0 if start_resp >= 35 else 10.0)
            else:
                if resp_cu == 50:
                    max_l = min(max_l, max(min_l + 10.0, cur_l - 5.0))
                if resp_cu == 0 and cur_l > min_l:
                    min_l = cur_l
            if phase == "P2":
                cur_f, cur_l = gp.select_next(min_l, max_l, rng)
            cur_l = min(ceiling, min(max_l, max(-10.0, cur_l)))

        fit_bounds = gp.get_boundaries()
        mae_gp_active.append(np.mean(np.abs(fit_bounds[:, :9] - truth[:, :9])))

        est_slopes = gp.fit_pass2_slopes(fit_bounds.T)
        est_fa = gp.fit_pass3_fa(fit_bounds.T, est_slopes)
        true_mean_slopes.append(np.mean(vl.slopes))
        est_mean_slopes.append(np.mean(est_slopes))
        true_fas.append(np.mean(vl.far))
        est_fas.append(est_fa)

    r_s, p_s = pearsonr(true_mean_slopes, est_mean_slopes)
    r_fa, p_fa = pearsonr(true_fas, est_fas)
    print(f"Pass 1 (100 Boundaries) 9-Band MAE: {np.mean(mae_gp_active):.2f} dB")
    print(f"Pass 2 (100 Slopes) Correlation:    r = {r_s:.3f} (p = {p_s:.2e})")
    print(f"Pass 3 (Global FA) Correlation:     r = {r_fa:.3f} (p = {p_fa:.2e})")


if __name__ == "__main__":
    main()
