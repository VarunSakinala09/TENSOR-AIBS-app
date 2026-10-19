
from pathlib import Path
import hashlib

data_dir = Path("dataset/raw/set-a")
record_ids = ["1076589", "1079854", "1129899"]

for rid in record_ids:
    print("\n" + "=" * 50)
    print("RECORD:", rid)

    for suffix in [".hea", ".dat", ".mat"]:
        path = data_dir / f"{rid}{suffix}"

        if path.exists():
            content = path.read_bytes()
            print(f"\nFile: {path.resolve()}")
            print("Size:", len(content), "bytes")
            print("SHA256:", hashlib.sha256(content).hexdigest())

            if suffix == ".hea":
                print("Header:")
                print(content[:1500].decode("ascii", errors="replace"))
        else:
            print("Missing:", path)
