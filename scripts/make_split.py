"""
Train/test split manifest — shared contract for B and C.
"""
import os, csv, random

def make_split(raw_dir="data/raw", out_csv="data/split.csv", test_ratio=0.2, seed=42):
    random.seed(seed)
    rows = []
    for label in ["real", "fake"]:
        files = os.listdir(os.path.join(raw_dir, label))
        random.shuffle(files)
        n_test = int(len(files) * test_ratio)
        for i, f in enumerate(files):
            split = "test" if i < n_test else "train"
            rows.append([f.replace(".flac", ".npy"), label, split])

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filename", "label", "split"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    make_split()