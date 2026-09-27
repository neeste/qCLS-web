"""Evaluate qCLS with participant-level leave-one-out PCA training.

Each fold trains its PCA model from all but one listener in the MCPF CSV,
installs that model in the WASM post-hoc fitter, and simulates sessions for
the held-out listener. The held-out listener is used only for response
generation and evaluation, never for PCA training.

The WASM module must be rebuilt after adding the runtime PCA setter:
    make

Example:
    python sim/qcls_wasm_sim_LOOCV.py --mode isophon --n_trials 60 --n_reps 20
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .mcpf_simulator import FREQS_HZ, N_BOUNDARIES, load_ground_truth
    from .qcls_wasm_accuracy_sim import (
        MODE_CODES,
        WasmAdapter,
        mad_between,
        mae,
        plot_profile_comparison,
    )
    from .train_pca_model import load_profiles, train_model
else:
    from mcpf_simulator import FREQS_HZ, N_BOUNDARIES, load_ground_truth
    from qcls_wasm_accuracy_sim import (
        MODE_CODES,
        WasmAdapter,
        mad_between,
        mae,
        plot_profile_comparison,
    )
    from train_pca_model import load_profiles, train_model

N_COMPONENTS = 5


class FoldWasmAdapter(WasmAdapter):
    """WASM session adapter that installs one fold's PCA before each run."""

    def __init__(self, runner: Path, wasm_module: Path, mode: str, model: dict):
        super().__init__(runner, wasm_module, mode)
        self.model = model

    def start_session(self, seed: int, n_trials: int) -> None:
        response = self._send({"cmd": "set_pca_model", "model": self.model})
        if not response.get("ok", False):
            raise RuntimeError(f"WASM rejected the fold PCA model: {response}")
        super().start_session(seed, n_trials)


def model_payload(
    mu: np.ndarray,
    d: np.ndarray,
    components: np.ndarray,
    score_mean: np.ndarray,
    score_std: np.ndarray,
) -> dict[str, list[float]]:
    return {
        "mu": mu.tolist(),
        "d": d.tolist(),
        "V": components.reshape(-1).tolist(),
        "score_mean": score_mean.tolist(),
        "score_std": score_std.tolist(),
    }


def run_loocv(
    csv_path: Path,
    runner: Path,
    wasm_module: Path,
    n_trials: int,
    n_reps: int,
    base_seed: int,
    mode: str,
    plot_repetitions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    listeners = load_ground_truth(csv_path)
    if len(listeners) < 3:
        raise ValueError("LOOCV requires at least three complete listener profiles")

    accuracy_rows = []
    reliability_rows = []
    bias_rows = []
    for fold_index, (listener_id, truth) in enumerate(listeners.items()):
        training_profiles = load_profiles(csv_path, exclude_listener_id=listener_id)
        if training_profiles.shape[0] != len(listeners) - 1:
            raise ValueError(
                f"Fold {listener_id} expected {len(listeners) - 1} training listeners, "
                f"found {training_profiles.shape[0]}"
            )

        # Do not use pca_model.h here: it may already include this held-out listener.
        mu, d, components, score_mean, score_std, explained = train_model(
            training_profiles, N_COMPONENTS
        )
        fold_model = model_payload(mu, d, components, score_mean, score_std)
        print(
            f"Fold {listener_id}: trained on {training_profiles.shape[0]} listeners; "
            f"{explained[:N_COMPONENTS].sum():.2%} variance retained",
            flush=True,
        )

        adapter = FoldWasmAdapter(runner, wasm_module, mode, fold_model)
        fits = []
        try:
            for repetition in range(n_reps):
                seed = base_seed + fold_index * n_reps + repetition
                virtual_listener = listeners.new_virtual_listener(
                    listener_id, np.random.default_rng(seed)
                )
                result = adapter.run_session(
                    virtual_listener, n_trials=n_trials, seed=seed
                )
                fit = result.fitted_boundaries
                fits.append(fit)
                accuracy_rows.append({
                    "mode": mode,
                    "listener_id": listener_id,
                    "training_listeners": training_profiles.shape[0],
                    "rep": repetition,
                    "seed": seed,
                    "mae_db": mae(fit, truth),
                })
                if plot_repetitions:
                    plot_profile_comparison(
                        listener_id, repetition + 1, truth, fit
                    )
        finally:
            adapter.close()

        for first, second in combinations(range(n_reps), 2):
            reliability_rows.append({
                "mode": mode,
                "listener_id": listener_id,
                "rep_i": first,
                "rep_j": second,
                "mad_db": mad_between(fits[first], fits[second]),
            })

        mean_signed_error = np.mean(np.stack(fits, axis=0) - truth[None, :, :], axis=0)
        for boundary in range(N_BOUNDARIES):
            for frequency_index, frequency_hz in enumerate(FREQS_HZ):
                bias_rows.append({
                    "mode": mode,
                    "listener_id": listener_id,
                    "boundary_index": boundary + 1,
                    "freq_hz": frequency_hz,
                    "signed_bias_db": mean_signed_error[boundary, frequency_index],
                })

    return (
        pd.DataFrame(accuracy_rows),
        pd.DataFrame(reliability_rows),
        pd.DataFrame(bias_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate qCLS using leave-one-listener-out PCA training."
    )
    parser.add_argument(
        "--ground_truth_csv",
        type=Path,
        default=Path(__file__).resolve().with_name("mcpf_parameters.csv"),
    )
    parser.add_argument(
        "--runner", type=Path,
        default=Path(__file__).resolve().with_name("wasm_runner.py"),
    )
    parser.add_argument(
        "--wasm_module", type=Path,
        default=Path(__file__).resolve().parents[1] / "qcls_core.js",
    )
    parser.add_argument("--mode", choices=MODE_CODES, default="isophon")
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--plot_repetitions", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().with_name("loocv_results"),
    )
    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n_trials must be positive")
    if args.n_reps < 2:
        parser.error("--n_reps must be at least 2 to calculate test-retest reliability")

    try:
        accuracy, reliability, bias = run_loocv(
            args.ground_truth_csv,
            args.runner,
            args.wasm_module,
            args.n_trials,
            args.n_reps,
            args.base_seed,
            args.mode,
            args.plot_repetitions,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accuracy.to_csv(args.out_dir / "accuracy_per_rep.csv", index=False)
    reliability.to_csv(args.out_dir / "reliability_per_pair.csv", index=False)
    bias.to_csv(args.out_dir / "signed_bias_by_cell.csv", index=False)

    summary = {
        "evaluation": "leave-one-listener-out",
        "mode": args.mode,
        "n_listeners": int(accuracy["listener_id"].nunique()),
        "training_listeners_per_fold": int(accuracy["training_listeners"].iloc[0]),
        "n_trials": args.n_trials,
        "n_reps": args.n_reps,
        "accuracy_mean_mae_db": float(accuracy["mae_db"].mean()),
        "accuracy_sd_mae_db": float(accuracy["mae_db"].std()),
        "accuracy_median_mae_db": float(accuracy["mae_db"].median()),
        "reliability_mean_mad_db": float(reliability["mad_db"].mean()),
        "reliability_sd_mad_db": float(reliability["mad_db"].std()),
        "reliability_median_mad_db": float(reliability["mad_db"].median()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved LOOCV results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
