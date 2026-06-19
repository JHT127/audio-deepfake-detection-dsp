"""
Feature extraction module — Person B.

Build on top of src/preprocessing.py. Do not re-implement framing or FFT.

Expected interface (for Person C's classify.py):
    extract_features(frames) -> 1D np.ndarray of DSP features for one file
        Input:  frames, shape (num_frames, 400), from np.load() or load_and_frame()
        Output: flat feature vector, shape (num_features,)

Required features (Task 5 — at least 6):
    - Spectral centroid
    - Spectral bandwidth
    - Spectral roll-off or flatness
    - Zero-crossing rate
    - Short-time energy
    - Autocorrelation-based feature

Usage pattern (load from split.csv):
    import pandas as pd, numpy as np
    from src.preprocessing import get_spectrogram, get_log_spectrogram
    from src.features import extract_features

    split_df = pd.read_csv("data/split.csv")
    for _, row in split_df.iterrows():
        frames = np.load(f"data/processed/{row.label}/{row.filename}")
        feat_vec = extract_features(frames)   # -> (num_features,)
"""

import numpy as np
from src.preprocessing import SR, get_spectrogram, get_log_spectrogram


def extract_features(frames):
    """Extract a flat DSP feature vector from windowed frames.

    Args:
        frames: np.ndarray, shape (num_frames, 400).

    Returns:
        np.ndarray, shape (num_features,) — one value per feature, averaged over frames.
    """
    raise NotImplementedError("Person B: implement feature extraction here")