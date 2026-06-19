
"""
scripts/test_handoff.py
-----------------------
Handoff sanity-check script for Person A's preprocessing output.

Run this from the repo root BEFORE handing off to teammates B and C:

    python scripts/test_handoff.py

It mimics exactly what B (feature extraction) and C (classification)
would write on day one when they sit down to use Person A's output.
Any shape mismatch, naming error, or API surprise surfaces here first.

Expected output at the end: "ALL CHECKS PASSED — handoff is ready."
"""

import sys
import csv
import pathlib
import numpy as np

# ── 0. Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT   = pathlib.Path(__file__).resolve().parents[1]
SPLIT_CSV   = REPO_ROOT / "data" / "split.csv"
PROCESSED   = REPO_ROOT / "data" / "processed"
SRC         = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))          # so `from src.preprocessing import …` works

# ── 1. Import the preprocessing module (catches import errors immediately) ──────
print("─" * 60)
print("[1/7] Importing src.preprocessing …")
try:
    from src.preprocessing import (
        SR,
        load_and_frame,
        get_spectrogram,
        get_log_spectrogram,
    )
    print(f"      OK  SR={SR}  (expected 16000)")
    assert SR == 16000, f"SR should be 16000, got {SR}"
except Exception as e:
    sys.exit(f"  FAIL: {e}")

# ── 2. Parse split.csv and verify structure ────────────────────────────────────
print("[2/7] Reading data/split.csv …")
try:
    rows = []
    with open(SPLIT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) >= {"filename", "label", "split"}, \
            f"Missing columns. Got: {reader.fieldnames}"
        for row in reader:
            rows.append(row)

    labels  = {r["label"]  for r in rows}
    splits  = {r["split"]  for r in rows}
    n_real  = sum(1 for r in rows if r["label"] == "real")
    n_fake  = sum(1 for r in rows if r["label"] == "fake")
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test  = sum(1 for r in rows if r["split"] == "test")

    print(f"      rows={len(rows)}  labels={labels}  splits={splits}")
    print(f"      real={n_real}  fake={n_fake}  train={n_train}  test={n_test}")

    assert labels == {"real", "fake"},  f"Unexpected labels: {labels}"
    assert splits <= {"train", "test"}, f"Unexpected splits: {splits}"
    assert len(rows) == 300,            f"Expected 300 rows, got {len(rows)}"
    assert n_real == 150 and n_fake == 150, \
        f"Expected 150/150 real/fake, got {n_real}/{n_fake}"
    print("      OK")
except Exception as e:
    sys.exit(f"  FAIL: {e}")

# ── 3. Verify .npy files exist on disk for every split.csv row ────────────────
print("[3/7] Verifying .npy files exist for every split.csv row …")
missing = []
for row in rows:
    path = PROCESSED / row["label"] / row["filename"]
    if not path.exists():
        missing.append(str(path))

if missing:
    sys.exit(f"  FAIL: {len(missing)} missing .npy files:\n" +
             "\n".join(missing[:5]) + ("\n  …" if len(missing) > 5 else ""))
print(f"      OK  all 300 .npy files present")

# ── 4. Shape check on a handful of files (simulates what B does on load) ──────
print("[4/7] Shape-checking sampled .npy files …")
EXPECTED_FRAME_WIDTH = 400   # 25 ms × 16000 Hz
EXPECTED_SPEC_BINS   = 201   # rfft bins for 400-sample frame

sample_rows = [r for r in rows if r["split"] == "train"][:3] + \
              [r for r in rows if r["split"] == "test" ][:3]

for row in sample_rows:
    path   = PROCESSED / row["label"] / row["filename"]
    frames = np.load(path)

    # frames should be 2-D: (num_frames, 400)
    assert frames.ndim == 2, \
        f"{path.name}: expected 2-D array, got shape {frames.shape}"
    assert frames.shape[1] == EXPECTED_FRAME_WIDTH, \
        f"{path.name}: frame width {frames.shape[1]} ≠ {EXPECTED_FRAME_WIDTH}"

    n_frames = frames.shape[0]
    assert n_frames >= 1, f"{path.name}: zero frames"

    print(f"      {row['label']:4s} | {path.name:<35s} | "
          f"shape={frames.shape}")

print("      OK  all sampled frames are (num_frames, 400)")

# ── 5. Test get_spectrogram() and get_log_spectrogram() ──────────────────────
#       This is EXACTLY what B will call in features.py
print("[5/7] Testing spectrogram functions on the same samples …")
for row in sample_rows[:2]:   # just 2 is enough; they all go through the same path
    path   = PROCESSED / row["label"] / row["filename"]
    frames = np.load(path)

    spec     = get_spectrogram(frames)
    log_spec = get_log_spectrogram(frames)

    assert spec.shape == (frames.shape[0], EXPECTED_SPEC_BINS), \
        f"get_spectrogram shape {spec.shape} ≠ ({frames.shape[0]}, {EXPECTED_SPEC_BINS})"
    assert log_spec.shape == spec.shape, \
        f"get_log_spectrogram shape {log_spec.shape} ≠ {spec.shape}"
    assert np.all(np.isfinite(spec)),     f"get_spectrogram contains NaN/Inf"
    assert np.all(np.isfinite(log_spec)), f"get_log_spectrogram contains NaN/Inf"

    print(f"      spec={spec.shape}  log_spec={log_spec.shape}  "
          f"spec_min={spec.min():.3f}  log_min={log_spec.min():.3f}")

print("      OK  spectrogram functions return (num_frames, 201) finite arrays")

# ── 6. Test load_and_frame() end-to-end on raw .flac files ───────────────────
#       B/C might call this directly instead of loading .npy; verify it matches
print("[6/7] Testing load_and_frame() on raw .flac files …")
RAW_DIRS = {
    "real": REPO_ROOT / "data" / "raw" / "real",
    "fake": REPO_ROOT / "data" / "raw" / "fake",
}

flac_checked = 0
for label, raw_dir in RAW_DIRS.items():
    flacs = sorted(raw_dir.glob("*.flac"))
    if not flacs:
        print(f"      SKIP ({label}): no .flac files found in {raw_dir}")
        continue
    # check 2 files per class
    for flac_path in flacs[:2]:
        frames_live = load_and_frame(flac_path)
        # Compare to the pre-saved .npy (base name differs only in extension)
        npy_path    = PROCESSED / label / (flac_path.stem + ".npy")
        if npy_path.exists():
            frames_saved = np.load(npy_path)
            assert frames_live.shape == frames_saved.shape, (
                f"load_and_frame({flac_path.name}) shape {frames_live.shape} "
                f"≠ saved .npy shape {frames_saved.shape}"
            )
            assert np.allclose(frames_live, frames_saved, atol=1e-6), \
                f"load_and_frame output doesn't match saved .npy for {flac_path.name}"
            print(f"      {label:4s} | {flac_path.name:<40s} | "
                  f"live={frames_live.shape}  ✓ matches .npy")
        else:
            print(f"      {label:4s} | {flac_path.name:<40s} | "
                  f"live={frames_live.shape}  (no saved .npy to compare)")
        flac_checked += 1

if flac_checked == 0:
    print("      SKIP: no .flac files available in data/raw/ "
          "(OK if running on a machine without the raw dataset)")
else:
    print(f"      OK  load_and_frame() matches saved .npy on {flac_checked} files")

# ── 7. Summarise the data contract one more time (acts as living docs) ─────────
print("[7/7] Data-contract summary for teammates B and C:")
print("""
    File:   data/split.csv
    Cols:   filename (.npy), label (real|fake), split (train|test)

    To load frames for a row:
        frames = np.load(f"data/processed/{row['label']}/{row['filename']}")
        # shape: (num_frames, 400)   — 25 ms Hamming-windowed frames at 16 kHz

    To get magnitude spectrogram (what B needs for most features):
        from src.preprocessing import get_spectrogram
        spec = get_spectrogram(frames)
        # shape: (num_frames, 201)   — rfft magnitude, NOT log-scaled

    To get log-magnitude spectrogram (closest to Alzantot et al.):
        from src.preprocessing import get_log_spectrogram
        log_spec = get_log_spectrogram(frames)
        # shape: (num_frames, 201)   — log10(|X[k]| + 1e-10)

    SR constant (always 16000):
        from src.preprocessing import SR

    Alternative — load from raw .flac directly (bypasses .npy cache):
        from src.preprocessing import load_and_frame
        frames = load_and_frame("data/raw/real/LA_E_1000265.flac")
        # same shape contract as above
""")

# ── Done ───────────────────────────────────────────────────────────────────────
print("─" * 60)
print("ALL CHECKS PASSED — handoff is ready.")
print("─" * 60)