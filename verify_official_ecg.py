
from pathlib import Path
from urllib.request import urlopen
import hashlib

local_dir = Path("dataset/raw/set-a")
verify_dir = Path("dataset/official_verification")
verify_dir.mkdir(parents=True, exist_ok=True)

record_ids = ["1076589", "1079854", "1129899"]
base_url = "https://physionet.org/files/challenge-2011/1.0.0/set-a"

def sha256(data):
    return hashlib.sha256(data).hexdigest()

for rid in record_ids:
    print(f"\n--- {rid} ---")
    local_path = local_dir / f"{rid}.dat"

    try:
        with urlopen(f"{base_url}/{rid}.dat", timeout=30) as response:
            official_data = response.read()

        # Save the reference copy separately; never overwrite the dataset.
        reference_path = verify_dir / f"{rid}.dat"
        reference_path.write_bytes(official_data)

        local_data = local_path.read_bytes()

        print("Local SHA256:   ", sha256(local_data))
        print("Official SHA256:", sha256(official_data))
        print("Local size:     ", len(local_data))
        print("Official size:  ", len(official_data))
        print("Matches official:", local_data == official_data)

    except Exception as exc:
        print("Verification failed:", repr(exc))
        print("Check your internet connection and try again.")
