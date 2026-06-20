"""
Feature extraction module — Person B.

Build on top of src/preprocessing.py. Do not re-implement framing or FFT.

Expected interface:
    extract_features(frames) -> 1D np.ndarray of DSP features for one file
"""

import numpy as np
from src.preprocessing import SR, get_spectrogram


def _safe_mean_std(values):
    """Return mean and standard deviation safely."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return 0.0, 0.0

    return float(np.mean(values)), float(np.std(values))


def _zero_crossing_rate(frames):
    """
    Compute zero-crossing rate for each frame.
    ZCR measures how frequently the signal changes sign.
    """
    signs = np.sign(frames)
    signs[signs == 0] = 1

    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    return zcr


def _short_time_energy(frames):
    """
    Compute short-time energy for each frame.
    Energy shows the power of the speech signal in each short segment.
    """
    return np.sum(frames ** 2, axis=1)


def _autocorrelation_feature(frames):
    """
    Compute an autocorrelation-based feature for each frame.
    We use the maximum normalized autocorrelation after lag 0.
    This describes periodicity/similarity inside the frame.
    """
    autocorr_values = []

    for frame in frames:
        frame = frame - np.mean(frame)
        energy = np.sum(frame ** 2)

        if energy < 1e-12:
            autocorr_values.append(0.0)
            continue

        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]  # keep non-negative lags
        corr = corr / (corr[0] + 1e-12)

        # Ignore lag 0 because it is always 1
        if len(corr) > 1:
            autocorr_values.append(float(np.max(corr[1:])))
        else:
            autocorr_values.append(0.0)

    return np.array(autocorr_values)


def extract_features(frames):
    """
    Extract a flat DSP feature vector from windowed frames.

    Args:
        frames: np.ndarray, shape (num_frames, 400)

    Returns:
        np.ndarray, shape (num_features,)
    """

    frames = np.asarray(frames, dtype=float)

    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array with shape (num_frames, frame_length)")

    if frames.shape[0] == 0:
        raise ValueError("frames array is empty")

    # Magnitude spectrogram: shape (num_frames, num_freq_bins)
    magnitude = get_spectrogram(frames)

    # Frequency axis in Hz
    freqs = np.fft.rfftfreq(frames.shape[1], d=1 / SR)

    # Avoid division by zero
    mag_sum = np.sum(magnitude, axis=1) + 1e-12

    # 1. Spectral centroid
    centroid = np.sum(magnitude * freqs, axis=1) / mag_sum

    # 2. Spectral bandwidth
    bandwidth = np.sqrt(
        np.sum(magnitude * (freqs - centroid[:, np.newaxis]) ** 2, axis=1) / mag_sum
    )

    # 3. Spectral roll-off at 85%
    cumulative_energy = np.cumsum(magnitude, axis=1)
    threshold = 0.85 * cumulative_energy[:, -1]
    rolloff_indices = np.array([
        np.searchsorted(cumulative_energy[i], threshold[i])
        for i in range(cumulative_energy.shape[0])
    ])
    rolloff_indices = np.clip(rolloff_indices, 0, len(freqs) - 1)
    rolloff = freqs[rolloff_indices]

    # 4. Spectral flatness
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-12), axis=1))
    arithmetic_mean = np.mean(magnitude + 1e-12, axis=1)
    flatness = geometric_mean / arithmetic_mean

    # 5. Zero-crossing rate
    zcr = _zero_crossing_rate(frames)

    # 6. Short-time energy
    energy = _short_time_energy(frames)

    # 7. Autocorrelation-based feature
    autocorr = _autocorrelation_feature(frames)

    # Create one fixed-length feature vector for the whole audio file
    feature_vector = []

    for feature in [
        centroid,
        bandwidth,
        rolloff,
        flatness,
        zcr,
        energy,
        autocorr,
    ]:
        mean_value, std_value = _safe_mean_std(feature)
        feature_vector.extend([mean_value, std_value])

    return np.array(feature_vector, dtype=float)


def get_feature_names():
    """
    Return names of the extracted features.
    Useful when saving features.csv for Person C.
    """
    base_names = [
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "zero_crossing_rate",
        "short_time_energy",
        "autocorrelation",
    ]

    names = []
    for name in base_names:
        names.append(f"{name}_mean")
        names.append(f"{name}_std")

    return names