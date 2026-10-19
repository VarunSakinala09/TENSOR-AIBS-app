
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw" / "set-a"

MAPPING_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)

SCREEN_FILE = (
    ROOT / "results" / "week8" / "frequency_screening"
    / "spectral_summary.csv"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


mapping = pd.read_csv(MAPPING_FILE, dtype={"record_id": str})
screen = pd.read_csv(SCREEN_FILE, dtype={"record_id": str})

screen_ids = set(screen["record_id"])
results = []

for record_id in mapping["record_id"]:
    headers = sorted(RAW_DIR.rglob(f"{record_id}.hea"))
    data_files = sorted(RAW_DIR.rglob(f"{record_id}.dat"))

    header_hashes = sorted(sha256_file(p) for p in headers)
    data_hashes = sorted(sha256_file(p) for p in data_files)

    results.append({
        "record_id": record_id,
        "header_count": len(headers),
        "data_count": len(data_files),
        "headers_identical": (
            len(header_hashes) > 0
            and len(set(header_hashes)) == 1
        ),
        "data_identical": (
            len(data_hashes) > 0
            and len(set(data_hashes)) == 1
        ),
        "in_screening": record_id in screen_ids,
    })

df = pd.DataFrame(results)

output = (
    ROOT / "results" / "week8" / "frequency_screening"
    / "raw_duplicate_audit.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)

print("MAPPED RECORDS:", len(df))
print("Records with multiple headers:", (df.header_count > 1).sum())
print("Records with multiple data files:", (df.data_count > 1).sum())
print("Records whose duplicate headers match:", df.headers_identical.sum())
print("Records whose duplicate data match:", df.data_identical.sum())
print("Records missing from screening:", (~df.in_screening).sum())
print("Records with missing raw files:",
      ((df.header_count == 0) | (df.data_count == 0)).sum())
print("Audit saved to:", output)
