
from pathlib import Path
from collections import defaultdict
import hashlib

data_dir = Path("dataset/raw/set-a")
hash_groups = defaultdict(list)
missing = []

for header in sorted(data_dir.glob("*.hea")):
    record_id = header.stem
    data_file = data_dir / f"{record_id}.dat"

    if not data_file.exists():
        missing.append(record_id)
        continue

    digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
    hash_groups[digest].append(record_id)

total = sum(len(ids) for ids in hash_groups.values())
duplicates = {
    digest: ids
    for digest, ids in hash_groups.items()
    if len(ids) > 1
}

print("Headers found:", len(list(data_dir.glob("*.hea"))))
print("Records with data files:", total)
print("Unique data-file contents:", len(hash_groups))
print("Duplicate hash groups:", len(duplicates))
print("Records missing .dat files:", len(missing))

print("\nFirst 10 duplicate groups:")
for digest, ids in list(duplicates.items())[:10]:
    print(f"{len(ids)} records | hash={digest[:16]} | IDs={ids[:12]}")

if missing:
    print("\nFirst missing record IDs:", missing[:20])
