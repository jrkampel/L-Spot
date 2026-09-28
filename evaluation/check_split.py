import re
from pathlib import Path

def patient_id(fname):
    m = re.match(r"^(P\d+)_", fname)
    return m.group(1) if m else None

splits = {}
for split in ["train", "val", "test"]:
    paths = sorted(Path(f"dataset/{split}/images").glob("*.png"))
    ids = {patient_id(p.name) for p in paths}
    if None in ids:
        print(f"WARNING: {split} — some filenames did not parse")
        ids.discard(None)
    splits[split] = ids
    print(f"{split}: {len(paths)} slices, {len(ids)} patients")

print()
ok = True
for a, b in [("train","val"), ("train","test"), ("val","test")]:
    overlap = splits[a] & splits[b]
    if overlap:
        ok = False
        print(f"OVERLAP {a}/{b}: {sorted(overlap)}")
    else:
        print(f"clean: {a} / {b}")

total = len(set().union(*splits.values()))
print(f"\ntotal unique patients: {total} (expect 94)")
print("PASS" if ok and total == 94 else "CHECK ABOVE")
