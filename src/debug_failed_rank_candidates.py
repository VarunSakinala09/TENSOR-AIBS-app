
import time
import traceback
import numpy as np
import pywt
import wfdb
from pathlib import Path
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

RECORD_ID = "2930238"
FIXED_RANK = (50, 20, 6)
MAX_RANK = (50, 36, 12)
FREQUENCIES = np.linspace(1, 40, 100)
TOLERANCE = 1e-8

def load_tensor(record_id):
    signal, fields = wfdb.rdsamp(str(DATA_DIR / record_id))
    scales = pywt.central_frequency("morl") * fields["fs"] / FREQUENCIES

    leads = []
    for i in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, i],
            scales,
            "morl",
            sampling_period=1 / fields["fs"]
        )
        leads.append(np.abs(coefficients).T)

    return np.stack(leads, axis=2).astype(np.float32)

def representation_size(shape, rank):
    return int(np.prod(rank) + sum(shape[i] * rank[i] for i in range(3)))

def evaluate(tensor, rank):
    print(f"    Evaluating {rank}", flush=True)

    core, factors = tucker(
        tensor, rank=list(rank), init="svd", tol=1e-5
    )
    reconstructed = tucker_to_tensor((core, factors))

    error = np.linalg.norm(tensor - reconstructed) / (
        np.linalg.norm(tensor) + 1e-12
    )

    size = representation_size(tensor.shape, rank)

    # Verify that TensorLy returned the requested dimensions.
    if tuple(core.shape) != tuple(rank):
        raise RuntimeError(
            f"Requested rank {rank}, but actual core is {core.shape}"
        )

    return {
        "rank": tuple(rank),
        "error": float(error),
        "size": size
    }

def neighbours(rank, shape, baseline_size, evaluated):
    candidates = set()

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

    return sorted(candidates)

def search(tensor):
    shape = tensor.shape
    baseline = evaluate(tensor, FIXED_RANK)
    evaluated = {FIXED_RANK: baseline}

    current_rank = FIXED_RANK
    current_result = baseline
    iteration = 0

    while True:
        iteration += 1
        print(f"\nSearch iteration {iteration}", flush=True)

        candidates = neighbours(
            current_rank,
            shape,
            baseline["size"],
            evaluated
        )

        print("Candidate count:", len(candidates), flush=True)

        if not candidates:
            break

        improving = []

        for candidate in candidates:
            try:
                result = evaluate(tensor, candidate)
                evaluated[candidate] = result

                if result["error"] < current_result["error"] - TOLERANCE:
                    improving.append(result)

            except Exception as exc:
                print(
                    f"FAILED rank {candidate}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )
                traceback.print_exc()
                raise

        if not improving:
            print("No improving neighbours; stopping.", flush=True)
            break

        current_result = min(
            improving,
            key=lambda item: (item["error"], item["size"])
        )
        current_rank = current_result["rank"]

        print("Moved to:", current_rank, flush=True)

    qualifying = [
        result for result in evaluated.values()
        if result["size"] < baseline["size"]
        and result["error"] < baseline["error"] - TOLERANCE
    ]

    selected = min(
        qualifying,
        key=lambda item: (item["error"], item["size"])
    ) if qualifying else baseline

    return baseline, selected, evaluated

if __name__ == "__main__":
    start = time.time()
    print("Loading:", RECORD_ID, flush=True)

    tensor = load_tensor(RECORD_ID)
    print("Tensor:", tensor.shape, flush=True)

    try:
        baseline, selected, evaluated = search(tensor)

        print("\nSEARCH COMPLETED", flush=True)
        print("Baseline:", baseline, flush=True)
        print("Selected:", selected, flush=True)
        print("Unique ranks evaluated:", len(evaluated), flush=True)

    except Exception:
        print("\nSEARCH FAILED", flush=True)
        traceback.print_exc()

    print(f"\nElapsed: {(time.time() - start) / 60:.2f} minutes", flush=True)
