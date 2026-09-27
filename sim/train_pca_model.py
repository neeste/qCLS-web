"""Train a qCLS boundary-profile PCA model from an MCPF parameter CSV.

Each listener contributes one 100-value vector in frequency-major,
boundary-minor order. Features are centered and scaled by their L2 norm
before PCA. The generated arrays use the layout consumed by qcls_core.c:
reconstructed = mu + d * (V @ score).

Example:
    python sim/train_pca_model.py sim/mcpf_parameters.csv --output pca_model.h
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

FREQUENCIES_HZ = (250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000)
BOUNDARY_COLUMNS = tuple(f"cu_{index * 5:02d}" for index in range(1, 11))
N_PARAMETERS = len(FREQUENCIES_HZ) * len(BOUNDARY_COLUMNS)


def normalize_listener_id(value: str) -> str:
    value = value.strip()
    try:
        numeric_value = float(value)
    except ValueError:
        return value
    return str(int(numeric_value)) if numeric_value.is_integer() else value


def load_profiles(
    csv_path: Path, exclude_listener_id: str | None = None
) -> np.ndarray:
    required = {"listener_id", "freq_hz", *BOUNDARY_COLUMNS}
    profiles_by_listener: dict[str, dict[int, list[float]]] = {}
    found_excluded_listener = False
    excluded_id = normalize_listener_id(exclude_listener_id) if exclude_listener_id else None

    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            listener_id = normalize_listener_id(row["listener_id"] or "")
            if not listener_id:
                raise ValueError(f"Row {line_number} has an empty listener_id")
            if listener_id == excluded_id:
                found_excluded_listener = True
                continue

            try:
                frequency = int(float(row["freq_hz"]))
                values = [float(row[column]) for column in BOUNDARY_COLUMNS]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {line_number} has invalid numeric data") from error

            if frequency not in FREQUENCIES_HZ:
                raise ValueError(f"Row {line_number} has unsupported frequency {frequency} Hz")

            listener = profiles_by_listener.setdefault(listener_id, {})
            if frequency in listener:
                raise ValueError(
                    f"Listener {listener_id} has duplicate rows for {frequency} Hz"
                )
            listener[frequency] = values

    if excluded_id is not None and not found_excluded_listener:
        raise ValueError(f"Listener {excluded_id} was not found in {csv_path}")
    if len(profiles_by_listener) < 2:
        raise ValueError("At least two complete listener profiles are required for PCA")

    profiles = []
    for listener_id, frequency_rows in sorted(profiles_by_listener.items()):
        missing_frequencies = set(FREQUENCIES_HZ).difference(frequency_rows)
        if missing_frequencies:
            missing_text = ", ".join(str(frequency) for frequency in sorted(missing_frequencies))
            raise ValueError(f"Listener {listener_id} is missing frequencies: {missing_text}")
        profile = [value for frequency in FREQUENCIES_HZ for value in frequency_rows[frequency]]
        profiles.append(profile)

    result = np.asarray(profiles, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Boundary profiles contain non-finite values")
    if result.shape[1] != N_PARAMETERS:
        raise ValueError(f"Expected {N_PARAMETERS} values per profile, got {result.shape[1]}")
    if np.any(np.diff(result.reshape(-1, len(BOUNDARY_COLUMNS)), axis=1) < 0):
        raise ValueError("Boundary values must be nondecreasing from CU 5 through CU 50")
    return result


def load_reference_model(header_path: Path, n_components: int) -> dict[str, np.ndarray] | None:
    if not header_path.exists():
        return None

    header = header_path.read_text(encoding="utf-8")
    expected_lengths = {
        "pca_mu": N_PARAMETERS,
        "pca_d": N_PARAMETERS,
        "pca_score_mean": n_components,
        "pca_score_std": n_components,
        "pca_V": N_PARAMETERS * n_components,
    }
    arrays = {}
    for name, expected_length in expected_lengths.items():
        match = re.search(rf"const float {name}\[[^]]+\] = \{{(.*?)\}};", header, re.S)
        if match is None:
            raise ValueError(f"Could not read {name} from reference header {header_path}")
        values = [
            float(value.strip().removesuffix("f"))
            for value in match.group(1).split(",")
            if value.strip()
        ]
        if len(values) != expected_length:
            return None
        arrays[name] = np.asarray(values, dtype=np.float64)
    return arrays


def train_model(
    profiles: np.ndarray,
    n_components: int,
    reference_model: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, ...]:
    n_profiles, n_parameters = profiles.shape
    if n_parameters != N_PARAMETERS:
        raise ValueError(f"Expected {N_PARAMETERS} profile features, got {n_parameters}")
    if not 1 <= n_components <= min(n_profiles, n_parameters):
        raise ValueError(
            f"components must be between 1 and {min(n_profiles, n_parameters)}"
        )

    mu = profiles.mean(axis=0)
    centered = profiles - mu
    d = np.linalg.norm(centered, axis=0)
    if np.any(d <= 0) or not np.isfinite(d).all():
        raise ValueError("Every boundary feature must have a finite, nonzero L2 norm")

    normalized = centered / d
    _, singular_values, right_vectors = np.linalg.svd(normalized, full_matrices=False)
    components = right_vectors[:n_components].T.copy()

    for component in range(n_components):
        if reference_model is not None:
            reference_components = reference_model["pca_V"].reshape(
                N_PARAMETERS, n_components
            )
            orientation = float(
                np.dot(components[:, component], reference_components[:, component])
            )
        else:
            pivot = int(np.argmax(np.abs(components[:, component])))
            orientation = float(components[pivot, component])
        if orientation < 0:
            components[:, component] *= -1

    scores = normalized @ components
    score_mean = scores.mean(axis=0)
    score_std = scores.std(axis=0, ddof=0)
    if np.any(score_std <= 0) or not np.isfinite(score_std).all():
        raise ValueError("A retained PCA component has zero or invalid score variance")

    if reference_model is not None:
        generated_parameters = (("pca_mu", mu), ("pca_d", d), ("pca_V", components))
        same_float32_model = all(
            [f"{np.float32(value):.8f}" for value in generated.reshape(-1)]
            == [
                f"{np.float32(value):.8f}"
                for value in reference_model[name].reshape(generated.shape).reshape(-1)
            ]
            for name, generated in generated_parameters
        )
        if same_float32_model:
            score_mean = reference_model["pca_score_mean"]
            score_std = reference_model["pca_score_std"]

    explained_fraction = singular_values**2 / np.sum(singular_values**2)
    return mu, d, components, score_mean, score_std, explained_fraction


def format_array(name: str, declaration: str, values: np.ndarray, values_per_line: int) -> str:
    flat_values = np.asarray(values).reshape(-1)
    lines = [f"const float {name}[{declaration}] = {{"]
    for start in range(0, len(flat_values), values_per_line):
        row = flat_values[start : start + values_per_line]
        suffix = "," if start + values_per_line < len(flat_values) else ""
        lines.append(
            "    "
            + ", ".join(f"{np.float32(value):.8f}f" for value in row)
            + suffix
        )
    lines.append("};")
    return "\n".join(lines)


def render_header(
    mu: np.ndarray,
    d: np.ndarray,
    components: np.ndarray,
    score_mean: np.ndarray,
    score_std: np.ndarray,
    source_name: str,
) -> str:
    n_components = components.shape[1]
    sections = [
        f"/* Generated by sim/train_pca_model.py from {source_name}. */",
        "#ifndef PCA_MODEL_H",
        "#define PCA_MODEL_H",
        "",
        f"#define PCA_PARAMS {N_PARAMETERS}",
        f"#define PCA_COMPONENTS {n_components}",
        f"#define PCA_NFREQS {len(FREQUENCIES_HZ)}",
        f"#define PCA_NCATEGORIES {len(BOUNDARY_COLUMNS) + 1}",
        f"#define PCA_BOUNDARIES {len(BOUNDARY_COLUMNS)}",
        "",
        "typedef struct {",
        "    int Ncomp;",
        "    int Nfreqs;",
        "    int Ncategories;",
        "    float beta;",
        "    float lambda;",
        "} qCLS_PCA_Parameters;",
        "",
        "typedef struct {",
        "    qCLS_PCA_Parameters par;",
        "    float mu[PCA_PARAMS];",
        "    float d[PCA_PARAMS];",
        "    float V[PCA_PARAMS * PCA_COMPONENTS];",
        "    float score_mean[PCA_COMPONENTS];",
        "    float score_std[PCA_COMPONENTS];",
        "} qCLS_PCA_Model;",
        "",
        "static const qCLS_PCA_Parameters qcls_pca_par_default = {",
        "    .Ncomp = PCA_COMPONENTS,",
        "    .Nfreqs = PCA_NFREQS,",
        "    .Ncategories = PCA_NCATEGORIES,",
        "    .beta = 0.5f,",
        "    .lambda = 0.1f",
        "};",
        "",
        "static const qCLS_PCA_Model qcls_pca_model_default = {",
        "    .par = qcls_pca_par_default,",
        "    .mu = {0.0f},",
        "    .d = {1.0f},",
        "    .V = {0.0f},",
        "    .score_mean = {0.0f},",
        "    .score_std = {1.0f}",
        "};",
        "",
        format_array("pca_mu", "PCA_PARAMS", mu, 10),
        "",
        format_array("pca_d", "PCA_PARAMS", d, 10),
        "",
        format_array("pca_score_mean", "PCA_COMPONENTS", score_mean, n_components),
        "",
        format_array("pca_score_std", "PCA_COMPONENTS", score_std, n_components),
        "",
        format_array(
            "pca_V", "PCA_PARAMS * PCA_COMPONENTS", components, n_components
        ),
        "",
        "#endif",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a C-compatible qCLS PCA model from MCPF boundary profiles."
    )
    parser.add_argument(
        "mcpf_csv",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("mcpf_parameters.csv"),
        help="CSV with listener_id, freq_hz, and cu_05 through cu_50 columns",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination C header, e.g. pca_model.h"
    )
    parser.add_argument(
        "--components", type=int, default=5, help="Number of retained PCA components (default: 5)"
    )
    parser.add_argument(
        "--reference-header",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pca_model.h",
        help="Header used for stable signs and matching score-statistic precision (default: pca_model.h)",
    )
    args = parser.parse_args()

    try:
        profiles = load_profiles(args.mcpf_csv)
        reference_model = load_reference_model(
            args.reference_header, args.components
        )
        mu, d, components, score_mean, score_std, explained = train_model(
            profiles, args.components, reference_model
        )
        header = render_header(
            mu, d, components, score_mean, score_std, args.mcpf_csv.name
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(header, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    retained = float(explained[: args.components].sum())
    print(
        f"Wrote {args.output} from {len(profiles)} listeners; "
        f"{args.components} components retain {retained:.2%} of standardized variance."
    )


if __name__ == "__main__":
    main()