import os
import wfdb

# Path to Set A
dataset_path = "dataset/raw/set-a"

# Find all ECG records
records = []

for file in os.listdir(dataset_path):
    if file.endswith(".hea"):
        records.append(file[:-4])

records.sort()

print("=" * 60)
print("PHYSIONET CHALLENGE 2011 - SET A")
print("WHOLE DATASET VERIFICATION")
print("=" * 60)

print("\nTotal records found:", len(records))

# Store information
valid_records = []
invalid_records = []

sampling_frequencies = set()
lead_counts = set()
durations = set()
lead_names = set()

# Check every record
for i, record_name in enumerate(records):

    record_path = os.path.join(dataset_path, record_name)

    try:
        record = wfdb.rdrecord(record_path)

        valid_records.append(record_name)

        sampling_frequencies.add(record.fs)
        lead_counts.add(record.n_sig)

        duration = len(record.p_signal) / record.fs
        durations.add(duration)

        lead_names.add(tuple(record.sig_name))

    except Exception as e:
        invalid_records.append(record_name)

    # Progress
    if (i + 1) % 100 == 0:
        print(f"Checked {i + 1}/{len(records)} records")

# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET RESULTS")
print("=" * 60)

print("Total records:", len(records))
print("Valid records:", len(valid_records))
print("Invalid records:", len(invalid_records))

print("\nSampling frequencies:", sampling_frequencies)
print("Number of leads:", lead_counts)
print("Recording durations:", durations)

print("\nLead names:")

for names in lead_names:
    print(names)

# --------------------------------------------------
# Read labels
# --------------------------------------------------

def read_label_file(filename):

    if not os.path.exists(filename):
        return []

    with open(filename, "r") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


acceptable = read_label_file(
    os.path.join(dataset_path, "RECORDS-acceptable")
)

unacceptable = read_label_file(
    os.path.join(dataset_path, "RECORDS-unacceptable")
)

indeterminate = read_label_file(
    os.path.join(dataset_path, "RECORDS-indeterminate")
)

print("\n" + "=" * 60)
print("LABEL RESULTS")
print("=" * 60)

print("Acceptable:", len(acceptable))
print("Unacceptable:", len(unacceptable))
print("Indeterminate:", len(indeterminate))

print("\n" + "=" * 60)
print("WHOLE DATASET CHECK COMPLETE")
print("=" * 60)