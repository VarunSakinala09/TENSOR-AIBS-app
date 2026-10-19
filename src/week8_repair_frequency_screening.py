
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw" / "set-a"
RESULT_DIR = ROOT / "results" / "week8" / "frequency_screening"

MAPPING_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)
SCREEN_FILE = RESULT_DIR / "spectral_summary.csv"
OUTPUT_FILE = RESULT_DIR / "corrected_spectral_summary.csv"

LIMITS = [30, 40, 50, 70, 100]



def calculate_record(record_id, header):
    record = wfdb.rdrecord(str(header.with_suffix("")))
    signal = np.asarray(record.p_signal, dtype=float)
    fs = float(record.fs)

    if signal.ndim != 2 or signal.size == 0:
        raise ValueError("Invalid signal shape")

    lead_results = []

    for lead in range(signal.shape[1]):
        x = signal[:, lead]

        if not np.isfinite(x).all():
            continue

        # A constant signal has zero spectral power.
        # Record it as zero above each frequency limit.
        if np.std(x) == 0:
            lead_results.append({
                f"power_above_{limit}hz": 0.0
                for limit in LIMITS
            })
            continue

        frequencies, power = welch(
            x, fs=fs, nperseg=min(1024, len(x))
        )

        total_power = np.trapezoid(power, frequencies)

        if total_power <= 0:
            lead_results.append({
                f"power_above_{limit}hz": 0.0
                for limit in LIMITS
            })
            continue

        result = {}

        for limit in LIMITS:
            mask = frequencies > limit
            high_power = (
                np.trapezoid(power[mask], frequencies[mask])
                if mask.sum() >= 2 else 0.0
            )
            result[f"power_above_{limit}hz"] = high_power / total_power

        lead_results.append(result)

    if not lead_results:
        raise ValueError("All leads contain non-finite values")

    row = pd.DataFrame(lead_results).mean().to_dict()
    row["record_id"] = str(record_id)
    row["sampling_rate_hz"] = fs
    row["number_of_leads_used"] = len(lead_results)
    return row



RESULT_DIR.mkdir(parents=True, exist_ok=True)

mapping = pd.read_csv(MAPPING_FILE, dtype={"record_id": str})
screen = pd.read_csv(SCREEN_FILE, dtype={"record_id": str})

if mapping["record_id"].duplicated().any():
    raise ValueError("Mapping contains duplicate record IDs")

# Keep one existing screening result per ID and remove IDs
# outside the established 998-record mapping.
valid_ids = set(mapping["record_id"])
screen = screen[screen["record_id"].isin(valid_ids)]
screen = screen.drop_duplicates(subset="record_id", keep="first")

existing_ids = set(screen["record_id"])
missing_ids = sorted(valid_ids - existing_ids)

print("Mapped records:", len(mapping))
print("Existing matched screening IDs:", len(existing_ids))
print("Records needing calculation:", len(missing_ids))

new_rows = []
failed = []

for index, record_id in enumerate(missing_ids, start=1):
    headers = sorted(RAW_DIR.rglob(f"{record_id}.hea"))

    if not headers:
        failed.append({"record_id": record_id, "error": "Header not found"})
        continue

    try:
        # The audit confirmed duplicate header/data copies are identical.
        # Use only one copy of this record.
        new_rows.append(calculate_record(record_id, headers[0]))

    except Exception as error:
        failed.append({"record_id": record_id, "error": str(error)})

    if index % 10 == 0:
        print(f"Processed missing records: {index}/{len(missing_ids)}")

new_df = pd.DataFrame(new_rows)

combined = pd.concat([screen, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset="record_id", keep="first")

# Attach the established labels and signal groups.
combined = combined.merge(
    mapping[["record_id", "label", "signal_hash"]],
    on="record_id",
    how="inner",
    validate="one_to_one"
)

combined.to_csv(OUTPUT_FILE, index=False)

if failed:
    pd.DataFrame(failed).to_csv(
        RESULT_DIR / "repair_failed_records.csv", index=False
    )

print("\nREPAIR SUMMARY")
print("Corrected screening records:", len(combined))
print("Unique record IDs:", combined["record_id"].nunique())
print("Failed missing records:", len(failed))
print("Saved:", OUTPUT_FILE)

if len(combined) == len(mapping) and not failed:
    print("\nSUCCESS: All mapped records have screening results.")
else:
    print("\nINCOMPLETE: Check repair_failed_records.csv if created.")
    print("Do not treat the output as a complete 998-record summary.")
