
from pathlib import Path

folder = Path("dataset/raw/set-a/set-a")

acceptable_file = folder / "RECORDS-acceptable"
unacceptable_file = folder / "RECORDS-unacceptable"

acceptable = {
    x.strip()
    for x in acceptable_file.read_text().splitlines()
    if x.strip()
}

unacceptable = {
    x.strip()
    for x in unacceptable_file.read_text().splitlines()
    if x.strip()
}

print("Acceptable records:", len(acceptable))
print("Unacceptable records:", len(unacceptable))
print("IDs in both lists:", sorted(acceptable & unacceptable))

duplicate_groups = [
    ["1002603", "1030400", "1038105", "1076589",
     "1079854", "1129899", "1255536", "1295738"],
    ["1006637", "1815563"],
    ["1425383", "1452192", "1722066", "1770220",
     "2224949", "2338564", "2432060", "2750205",
     "2761488", "2853629"],
]

for group in duplicate_groups:
    print("\nDuplicate signal group:")
    for rid in group:
        if rid in acceptable:
            label = "Acceptable"
        elif rid in unacceptable:
            label = "Unacceptable"
        else:
            label = "Not found in either list"
        print(rid, "->", label)
