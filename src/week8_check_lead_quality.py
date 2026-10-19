
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "raw" / "set-a"

OUTPUT_DIR = ROOT / "results" / "week8" / "lead_quality_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Compare missed, detected, and correctly classified records.
RECORDS = {
    "Missed_Unacceptable": "1105115",
    "Detected_Unacceptable": "1192090",
    "Correct_Acceptable": "2804853",
}


def find_record(record_id):
    candidates = [
        DATASET_DIR / record_id,
        DATASET_DIR / "set-a" / record_id,
    ]

    for path in candidates:
        if Path(str(path) + ".hea").exists():
            return path

    return None


rows = []

for category, record_id in RECORDS.items():
    record_path = find_record(record_id)

    if record_path is None:
        print(f"\nHeader not found: {record_id}")
        continue

    # Physical values are used for signal statistics.
    signal, fields = wfdb.rdsamp(str(record_path))

    lead_names = fields.get("sig_name", [])
    units = fields.get("units", [])
    fs = fields["fs"]

    # Read digital values too, to check whether constant physical
    # values correspond to constant raw samples.
    digital_record = wfdb.rdrecord(
        str(record_path),
        physical=False
    )
    digital_signal = np.asarray(digital_record.d_signal)

    print("\n" + "=" * 90)
    print(f"Record: {record_id} | Group: {category}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Signal shape: {signal.shape}")
    print(f"Units: {units}")
    print("=" * 90)

    for i in range(signal.shape[1]):
        x = signal[:, i]
        raw = digital_signal[:, i]
        name = lead_names[i] if i < len(lead_names) else f"Lead_{i+1}"
        unit = units[i] if i < len(units) else "unknown"

        finite = np.isfinite(x)
        x_valid = x[finite]

        if len(x_valid) == 0:
            print(f"{name}: no finite physical samples")
            continue

        # Longest consecutive run of identical physical samples.
        same_as_previous = np.diff(x_valid) == 0
        longest_run = 1
        current_run = 1

        for same in same_as_previous:
            if same:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1

        physical_unique = len(np.unique(x_valid))
        digital_unique = len(np.unique(raw))

        row = {
            "record_id": record_id,
            "category": category,
            "lead": name,
            "unit": unit,
            "physical_min": float(np.min(x_valid)),
            "physical_max": float(np.max(x_valid)),
            "physical_mean": float(np.mean(x_valid)),
            "physical_std": float(np.std(x_valid)),
            "physical_unique_values": physical_unique,
            "digital_min": int(np.min(raw)),
            "digital_max": int(np.max(raw)),
            "digital_unique_values": digital_unique,
            "nonfinite_count": int(np.sum(~finite)),
            "longest_identical_run": int(longest_run),
            "fraction_physical_exactly_minus80": float(
                np.mean(x_valid == -80)
            ),
        }

        rows.append(row)

        print(
            f"{name:>4} | unit={unit:>5} | "
            f"physical min={np.min(x_valid):9.4f} | "
            f"max={np.max(x_valid):9.4f} | "
            f"std={np.std(x_valid):9.4f} | "
            f"unique physical={physical_unique:5d} | "
            f"unique digital={digital_unique:5d} | "
            f"longest identical run={longest_run:5d} | "
            f"exactly -80={np.mean(x_valid == -80):.3f}"
        )


if rows:
    results = pd.DataFrame(rows)
    output_file = OUTPUT_DIR / "per_lead_statistics.csv"
    results.to_csv(output_file, index=False)

    print("\n" + "=" * 90)
    print("AUDIT COMPLETE")
    print(f"Saved: {output_file}")
    print("=" * 90)
else:
    print("\nNo records were successfully inspected.")
