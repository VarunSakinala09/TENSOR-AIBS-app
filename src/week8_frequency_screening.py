
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import welch

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw" / "set-a"
OUTPUT_DIR = ROOT / "results" / "week8" / "frequency_screening"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_LIMITS = [30, 40, 50, 70, 100]
rows = []

header_files = sorted(RAW_DIR.rglob("*.hea"))

print("Total header files:", len(header_files))
print("First 10 header paths:")
for header in header_files[:10]:
    print(header.relative_to(RAW_DIR))

print("ECG FREQUENCY-CONTENT SCREENING")
print("Records discovered:", len(header_files))
print("Candidate upper limits:", CANDIDATE_LIMITS)

for index, header in enumerate(header_files, start=1):
    record_path = header.with_suffix("")

    try:
        record = wfdb.rdrecord(str(record_path))
        signal = np.asarray(record.p_signal, dtype=float)
        fs = float(record.fs)

        if signal.ndim != 2 or signal.size == 0:
            continue

        # Welch power spectrum for each lead
        lead_results = []

        for lead in range(signal.shape[1]):
            x = signal[:, lead]

            if not np.isfinite(x).all():
                continue

            if np.std(x) == 0:
                continue

            frequencies, power = welch(
                x,
                fs=fs,
                nperseg=min(1024, len(x))
            )

            total_power = np.trapezoid(power, frequencies)

            if total_power <= 0:
                continue

            fractions = {}

            for limit in CANDIDATE_LIMITS:
                mask = frequencies > limit

                high_power = (
                    np.trapezoid(power[mask], frequencies[mask])
                    if mask.sum() >= 2
                    else 0.0
                )

                fractions[f"power_above_{limit}hz"] = (
                    high_power / total_power
                )

            lead_results.append(fractions)

        if not lead_results:
            continue

        # Average the per-lead fractions for this record
        result = pd.DataFrame(lead_results).mean().to_dict()
        result["record_id"] = header.stem
        result["sampling_rate_hz"] = fs
        result["number_of_leads_used"] = len(lead_results)

        rows.append(result)

        if index % 100 == 0:
            print(f"Processed {index}/{len(header_files)} records")

        # Save progress so an interruption doesn't lose completed records
        if index % 25 == 0:
            pd.DataFrame(rows).to_csv(
                OUTPUT_DIR / "spectral_summary.csv",
                index=False
            )

    except Exception as error:
        print(f"Skipped {header.stem}: {error}")

output_file = OUTPUT_DIR / "spectral_summary.csv"

if rows:
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_file, index=False)

    print("\nSCREENING COMPLETE")
    print("Records analyzed:", len(result_df))
    print("Saved to:", output_file)

    columns = [
        f"power_above_{limit}hz"
        for limit in CANDIDATE_LIMITS
    ]

    print("\nMean fraction of power above each limit:")
    print(result_df[columns].mean().round(4))

else:
    print("No valid records were processed.")
    print("Check the dataset path and WFDB installation.")
