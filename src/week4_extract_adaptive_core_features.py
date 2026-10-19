
from pathlib import Path
import ast
import time
import hashlib
import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorly as tl
from tensorly.decomposition import tucker

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"/ "set-a"
RANK_FILE = (
    ROOT / "results" / "week2" / "rank_budget_all_records"
    / "multi_record_summary.csv"
)
OUT_DIR = ROOT / "results" / "week4" / "adaptive_core_features"
OUT_FILE = OUT_DIR / "adaptive_core_summary_features.csv"

FS_EXPECTED = 500
N_FREQUENCIES = 100
FREQ_MIN = 1
FREQ_MAX = 40

OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_rank(value):
    """Convert a stored rank such as '(49, 17, 10)' into a tuple."""
    parsed = ast.literal_eval(str(value))
    rank = tuple(int(v) for v in parsed)

    if len(rank) != 3 or any(v < 1 for v in rank):
        raise ValueError(f"Invalid Tucker rank: {value}")

    return rank



def find_record_path(record_id):
    matches = sorted(DATA_DIR.rglob(f"{record_id}.hea"))

    valid = []

    for header in matches:
        data_file = header.with_suffix(".dat")

        if header.exists() and data_file.exists():
            header_hash = hashlib.sha256(header.read_bytes()).hexdigest()
            data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()

            valid.append((header, header_hash, data_hash))

    if not valid:
        raise FileNotFoundError(
            f"No valid header/data pair found for {record_id}"
        )

    signatures = {
        (header_hash, data_hash)
        for _, header_hash, data_hash in valid
    }

    if len(signatures) > 1:
        paths = [str(header) for header, _, _ in valid]
        raise RuntimeError(
            f"Different files found for {record_id}: {paths}"
        )

    # Multiple paths are safe to resolve when their file contents match.
    return valid[0][0].with_suffix("")



def create_scalogram_tensor(record_id):
    """Reproduce the project's 1–40 Hz Morlet CWT tensor."""
    record_path = find_record_path(record_id)
    signal, fields = wfdb.rdsamp(str(record_path))

    fs = fields["fs"]

    if fs != FS_EXPECTED:
        raise ValueError(
            f"Unexpected sampling frequency for {record_id}: {fs}"
        )

    if signal.ndim != 2 or not np.isfinite(signal).all():
        raise ValueError(f"Invalid signal for {record_id}")

    frequencies = np.linspace(
        FREQ_MIN, FREQ_MAX, N_FREQUENCIES
    )
    central_frequency = pywt.central_frequency("morl")
    scales = central_frequency * fs / frequencies

    scalograms = []

    for lead_index in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead_index],
            scales,
            "morl",
            sampling_period=1 / fs
        )
        scalograms.append(np.abs(coefficients))

    # Shape: time × frequency × leads
    tensor = np.transpose(np.asarray(scalograms), (2, 1, 0))

    if not np.isfinite(tensor).all():
        raise ValueError(f"Non-finite CWT values for {record_id}")

    return tensor


def summarize_core(core):
    """
    Extract compact statistics rather than flattening differently
    shaped cores into supposedly aligned feature columns.
    """
    values = np.asarray(core, dtype=np.float64).ravel()
    absolute = np.abs(values)
    energy = float(np.sum(values ** 2))
    abs_sum = float(np.sum(absolute))

    # Normalize absolute values to form a histogram-based distribution.
    if abs_sum > 0:
        probabilities = absolute / abs_sum
        nonzero = probabilities[probabilities > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
    else:
        entropy = 0.0

    quantiles = np.quantile(
        values, [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    )

    features = {
        "core_size": int(values.size),
        "core_mean": float(np.mean(values)),
        "core_std": float(np.std(values)),
        "core_mean_absolute": float(np.mean(absolute)),
        "core_max_absolute": float(np.max(absolute)),
        "core_energy": energy,
        "core_log_energy": float(np.log1p(energy)),
        "core_absolute_entropy": entropy,
        "core_near_zero_fraction": float(
            np.mean(absolute <= 1e-12)
        ),
    }

    names = [
        "min", "q10", "q25", "median",
        "q75", "q90", "max"
    ]

    for name, value in zip(names, quantiles):
        features[f"core_{name}"] = float(value)

    # Statistics for each tensor mode.
    for mode, mode_name in enumerate(
        ["time", "frequency", "leads"]
    ):
        reduced_axes = tuple(
            axis for axis in range(core.ndim) if axis != mode
        )
        mode_energy = np.sum(core ** 2, axis=reduced_axes)
        mode_energy = mode_energy / max(energy, 1e-30)

        features[f"{mode_name}_energy_mean"] = float(
            np.mean(mode_energy)
        )
        features[f"{mode_name}_energy_std"] = float(
            np.std(mode_energy)
        )
        features[f"{mode_name}_energy_max"] = float(
            np.max(mode_energy)
        )

    return features


def main():
    if not RANK_FILE.exists():
        raise FileNotFoundError(RANK_FILE)

    rank_data = pd.read_csv(RANK_FILE)
    rank_data["record_id"] = rank_data["record_id"].astype(str)

    if rank_data["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in rank summary.")

    required = {"record_id", "label", "selected_rank"}
    if not required.issubset(rank_data.columns):
        raise ValueError(
            f"Missing columns: {sorted(required - set(rank_data.columns))}"
        )

    rows = []
    failures = []

    print("=" * 65)
    print("ADAPTIVE TUCKER CORE FEATURE EXTRACTION")
    print("=" * 65)
    print(f"Records with selected ranks: {len(rank_data)}")

    for index, row in rank_data.iterrows():
        record_id = str(row["record_id"])
        started = time.time()

        try:
            tensor = create_scalogram_tensor(record_id)
            rank = parse_rank(row["selected_rank"])

            if any(r > d for r, d in zip(rank, tensor.shape)):
                raise ValueError(
                    f"Rank {rank} exceeds tensor shape {tensor.shape}"
                )

            core, factors = tucker(
                tl.tensor(tensor, dtype=tl.float64),
                rank=list(rank),
                init="svd"
            )

            core = np.asarray(core)

            if not np.isfinite(core).all():
                raise ValueError("Core contains non-finite values.")

            features = summarize_core(core)

            output_row = {
                "record_id": record_id,
                "label": row["label"],
                "selected_rank": str(rank),
                "core_time_rank": rank[0],
                "core_frequency_rank": rank[1],
                "core_lead_rank": rank[2],
                "tensor_shape": str(tensor.shape),
                "core_shape": str(core.shape),
                **features,
            }

            rows.append(output_row)

        except Exception as exc:
            failures.append({
                "record_id": record_id,
                "error": repr(exc)
            })

        if (index + 1) % 25 == 0 or index + 1 == len(rank_data):
            print(
                f"Processed {index + 1}/{len(rank_data)} | "
                f"Successful: {len(rows)} | Failed: {len(failures)}"
            )

    if rows:
        result = pd.DataFrame(rows)

        if result["record_id"].duplicated().any():
            raise ValueError("Duplicate IDs in extracted features.")

        feature_columns = [
            c for c in result.columns
            if c not in [
                "record_id", "label", "selected_rank",
                "tensor_shape", "core_shape"
            ]
        ]

        if not np.isfinite(
            result[feature_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError("Saved feature table contains non-finite values.")

        result.to_csv(OUT_FILE, index=False)

    failure_file = OUT_DIR / "failed_records.csv"
    pd.DataFrame(
        failures, columns=["record_id", "error"]
    ).to_csv(failure_file, index=False)

    print("\n" + "=" * 65)
    print("EXTRACTION SUMMARY")
    print("=" * 65)
    print(f"Successful records: {len(rows)}")
    print(f"Failed records: {len(failures)}")
    print(f"Feature output: {OUT_FILE}")
    print(f"Failure log: {failure_file}")

    if rows:
        print(f"Feature columns: {len(feature_columns)}")
        print("\nFirst five extracted records:")
        print(
            result[
                ["record_id", "label", "selected_rank", "core_shape"]
            ].head().to_string(index=False)
        )


if __name__ == "__main__":
    main()
