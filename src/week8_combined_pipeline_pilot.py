
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor


# ==================================================
# 1. SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

SELECTOR_FILE = (
    ROOT / "results" / "week7" / "adaptive_selector"
    / "per_record_adaptive_predictions.csv"
)

RANK_FILE = (
    ROOT / "results" / "week2" / "rank_budget_all_records"
    / "multi_record_summary.csv"
)

OUT_DIR = ROOT / "results" / "week8" / "combined_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_RANK = (50, 20, 6)
MAX_RANK = (50, 36, 12)
TOLERANCE = 1e-8
MAX_SEARCH_ITERATIONS = 2

# Two records for a quick technical pilot.
PILOT_IDS = ["1002867", "1002603"]


# ==================================================
# 2. FIND AND LOAD A RECORD
# ==================================================

def find_record_path(record_id):
    matches = list(DATA_DIR.rglob(f"{record_id}.hea"))

    for header in matches:
        if header.with_suffix(".dat").exists():
            return header.with_suffix("")

    raise FileNotFoundError(
        f"Could not find both .hea and .dat for {record_id}"
    )


def load_ecg(record_id):
    record_path = find_record_path(record_id)
    signal, fields = wfdb.rdsamp(str(record_path))

    if signal.shape != (5000, 12):
        raise ValueError(
            f"{record_id}: unexpected ECG shape {signal.shape}"
        )

    if not np.isfinite(signal).all():
        raise ValueError(f"{record_id}: non-finite ECG values")

    return signal, fields["fs"]


# ==================================================
# 3. BUILD CWT TENSOR
# ==================================================

def build_tensor(signal, fs, upper_frequency):
    frequencies = np.linspace(1, upper_frequency, 100)
    scales = pywt.central_frequency("morl") * fs / frequencies

    scalograms = []

    for lead in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            "morl",
            sampling_period=1 / fs
        )
        scalograms.append(np.abs(coefficients).T)

    tensor = np.stack(scalograms, axis=2).astype(np.float32)

    if tensor.shape != (5000, 100, 12):
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    if not np.isfinite(tensor).all():
        raise ValueError("CWT tensor contains non-finite values")

    return tensor


# ==================================================
# 4. TUCKER EVALUATION
# ==================================================

def representation_size(shape, rank):
    return (
        int(np.prod(rank))
        + sum(int(dim) * int(r) for dim, r in zip(shape, rank))
    )


def evaluate(tensor, rank):
    core, factors = tucker(
        tensor,
        rank=list(rank),
        init="svd",
        tol=1e-5
    )

    reconstructed = tucker_to_tensor((core, factors))

    error = (
        np.linalg.norm(tensor - reconstructed)
        / max(np.linalg.norm(tensor), 1e-12)
    )

    return {
        "rank": tuple(int(x) for x in rank),
        "error": float(error),
        "size": representation_size(tensor.shape, rank)
    }


def search_ranks(tensor):
    baseline = evaluate(tensor, FIXED_RANK)
    baseline_size = baseline["size"]

    evaluated = {FIXED_RANK: baseline}
    current = baseline
    history = []

    for iteration in range(1, MAX_SEARCH_ITERATIONS + 1):
        candidates = set()
        rank = current["rank"]
        shape = tensor.shape

        for decrease_mode in range(3):
            for increase_mode in range(3):
                if decrease_mode == increase_mode:
                    continue

                candidate = list(rank)
                candidate[decrease_mode] -= 1
                candidate[increase_mode] += 1

                if any(
                    candidate[i] < 1
                    or candidate[i] > min(shape[i], MAX_RANK[i])
                    for i in range(3)
                ):
                    continue

                candidate = tuple(candidate)

                if candidate in evaluated:
                    continue

                if representation_size(shape, candidate) >= baseline_size:
                    continue

                candidates.add(candidate)

        if not candidates:
            break

        results = []

        for candidate in sorted(candidates):
            result = evaluate(tensor, candidate)
            evaluated[candidate] = result
            results.append(result)

            history.append({
                "iteration": iteration,
                "rank": str(candidate),
                "error": result["error"],
                "representation_size": result["size"]
            })

        better = [
            item for item in results
            if item["error"] < current["error"] - TOLERANCE
        ]

        if not better:
            break

        current = min(
            better,
            key=lambda item: (item["error"], item["size"])
        )

    qualifying = [
        item for item in evaluated.values()
        if item["size"] < baseline_size
        and item["error"] < baseline["error"] - TOLERANCE
    ]

    selected = (
        min(qualifying, key=lambda item: (item["error"], item["size"]))
        if qualifying else baseline
    )

    return baseline, selected, history


# ==================================================
# 5. LOAD EXISTING CWT SELECTIONS
# ==================================================

def load_selections():
    if not SELECTOR_FILE.exists():
        raise FileNotFoundError(
            f"Missing existing selector output: {SELECTOR_FILE}"
        )

    df = pd.read_csv(SELECTOR_FILE)
    required = {"record_id", "selected_configuration"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Selector file must contain {sorted(required)}"
        )

    df["record_id"] = df["record_id"].astype(str).str.strip()

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in selector output")

    df = df.set_index("record_id")

    for record_id in PILOT_IDS:
        if record_id not in df.index:
            raise ValueError(
                f"{record_id} is missing from the Week 7 selector output"
            )

    return df


# ==================================================
# 6. RUN PILOT
# ==================================================

def main():
    print("\nCOMBINED ADAPTIVE CWT + TUCKER PILOT")
    print("=" * 60)
    print("This is a technical pilot, not the final CV experiment.")

    selections = load_selections()
    summary = []
    feature_rows = []
    rank_history = []

    for record_id in PILOT_IDS:
        print(f"\nRecord: {record_id}")

        signal, fs = load_ecg(record_id)
        selection = selections.loc[record_id, "selected_configuration"]

        if selection == "CWT_1_40_Hz":
            upper_frequency = 40
        elif selection == "CWT_1_50_Hz":
            upper_frequency = 50
        else:
            raise ValueError(
                f"Unknown CWT selection for {record_id}: {selection}"
            )

        print(f"Selected CWT: 1–{upper_frequency} Hz")

        # Generate the tensor using the selected CWT range.
        tensor = build_tensor(signal, fs, upper_frequency)
        print("Tensor shape:", tensor.shape)

        # Search Tucker ranks on THIS tensor.
        start = time.perf_counter()
        baseline, selected, history = search_ranks(tensor)
        elapsed = time.perf_counter() - start

        print("Fixed rank:", baseline["rank"])
        print("Selected rank:", selected["rank"])
        print(f"Fixed error: {baseline['error']:.6f}")
        print(f"Selected error: {selected['error']:.6f}")
        print(f"Selected representation size: {selected['size']}")

        # Use a consistent fixed-rank core for the classifier features.
        # The adaptive rank values are appended as three metadata features.
        fixed_core, _ = tucker(
            tensor,
            rank=list(FIXED_RANK),
            init="svd",
            tol=1e-5
        )

        core_features = np.asarray(fixed_core).reshape(-1)

        if len(core_features) != 6000:
            raise ValueError(
                f"Unexpected core feature count: {len(core_features)}"
            )

        features = np.concatenate([
            core_features,
            np.asarray(selected["rank"], dtype=np.float32)
        ])

        if len(features) != 6003 or not np.isfinite(features).all():
            raise ValueError("Invalid final feature vector")

        feature_row = {"record_id": record_id}
        feature_row.update({
            f"feature_{i}": float(value)
            for i, value in enumerate(features)
        })
        feature_rows.append(feature_row)

        summary.append({
            "record_id": record_id,
            "selected_cwt": selection,
            "tensor_shape": str(tensor.shape),
            "fixed_rank": str(FIXED_RANK),
            "selected_rank": str(selected["rank"]),
            "fixed_error": baseline["error"],
            "selected_error": selected["error"],
            "fixed_representation_size": baseline["size"],
            "selected_representation_size": selected["size"],
            "improved_both": (
                selected["error"] < baseline["error"] - TOLERANCE
                and selected["size"] < baseline["size"]
            ),
            "feature_count": len(features),
            "rank_search_seconds": elapsed
        })

        for item in history:
            rank_history.append({
                "record_id": record_id,
                **item
            })

    # Save only to the new Week 8 directory.
    pd.DataFrame(summary).to_csv(
        OUT_DIR / "pilot_summary.csv", index=False
    )

    pd.DataFrame(feature_rows).to_csv(
        OUT_DIR / "pilot_features.csv", index=False
    )

    pd.DataFrame(rank_history).to_csv(
        OUT_DIR / "pilot_rank_search_history.csv", index=False
    )

    print("\n" + "=" * 60)
    print("PILOT COMPLETED")
    print("Summary:", OUT_DIR / "pilot_summary.csv")
    print("Features:", OUT_DIR / "pilot_features.csv")
    print("Rank history:", OUT_DIR / "pilot_rank_search_history.csv")
    print("Feature dimensions: 6000 core values + 3 rank values")
    print("Next step: validate the outputs before full-scale evaluation.")


if __name__ == "__main__":
    main()
