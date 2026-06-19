# scripts/verify_subset.py
"""
Step 5: Sanity check that extracted files actually load and
report their sample rate / duration.
"""
import os, librosa

for label in ["real", "fake"]:
    folder = f"data/raw/{label}"
    files = os.listdir(folder)[:3]
    print(f"\n--- {label} samples ---")
    for fname in files:
        y, sr = librosa.load(os.path.join(folder, fname), sr=None)
        print(f"{fname}: sr={sr}, duration={len(y)/sr:.2f}s")