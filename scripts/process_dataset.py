"""
Step: batch-convert all raw audio into windowed frame arrays.
Output: data/processed/{real,fake}/<filename>.npy
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.preprocessing import load_and_frame


def process_dataset(raw_dir="data/raw", out_dir="data/processed"):
    for label in ["real", "fake"]:
        in_folder = os.path.join(raw_dir, label)
        out_folder = os.path.join(out_dir, label)
        os.makedirs(out_folder, exist_ok=True)

        files = os.listdir(in_folder)
        for i, fname in enumerate(files):
            frames = load_and_frame(os.path.join(in_folder, fname))
            out_name = fname.replace(".flac", ".npy")
            np.save(os.path.join(out_folder, out_name), frames)

        print(f"{label}: processed {len(files)} files -> {out_folder}")

if __name__ == "__main__":
    process_dataset()