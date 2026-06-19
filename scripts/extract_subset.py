# scripts/extract_subset.py
"""
Step 4: Extract only the 300 selected audio files from the
remote zip, byte-range by byte-range — not the full archive.
"""
import os, shutil
from remotezip import RemoteZip

URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
FLAC_PREFIX = "LA/ASVspoof2019_LA_eval/flac/"  # from step 1 output

def extract_selected(selected_txt, out_dir):
    with open(selected_txt) as f:
        wanted = set(line.strip() for line in f if line.strip())

    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = "data/_tmp_extract"

    with RemoteZip(URL) as zf:
        for n in zf.namelist():
            if not n.startswith(FLAC_PREFIX):
                continue
            stem = os.path.basename(n).replace(".flac", "")
            if stem in wanted:
                zf.extract(n, tmp_dir)

    # flatten into out_dir
    for root, _, files in os.walk(tmp_dir):
        for fname in files:
            shutil.move(os.path.join(root, fname), os.path.join(out_dir, fname))
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    extract_selected("data/external/selected_real.txt", "data/raw/real")
    extract_selected("data/external/selected_fake.txt", "data/raw/fake")
    print("Done.")