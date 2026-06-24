"""DSP feature extraction. Uses src/preprocessing for framing and FFT."""

import numpy as np
from src.preprocessing import SR, get_spectrogram


def _safe_mean_std(values):
    """Return mean and standard deviation safely."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return 0.0, 0.0

    return float(np.mean(values)), float(np.std(values))


def _zero_crossing_rate(frames):
    """Compute zero-crossing rate."""
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    return zcr


def _short_time_energy(frames):
    """Compute short-time energy."""
    return np.sum(frames ** 2, axis=1)


def _autocorrelation_feature(frames):
    """Max normalized autocorrelation."""
    autocorr_values = []
    for frame in frames:
        frame = frame - np.mean(frame)
        energy = np.sum(frame ** 2)
        if energy < 1e-12:
            autocorr_values.append(0.0)
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        corr = corr / (corr[0] + 1e-12)
        if len(corr) > 1:
            autocorr_values.append(float(np.max(corr[1:])))
        else:
            autocorr_values.append(0.0)
    return np.array(autocorr_values)


def extract_features(frames):
    """Extract DSP feature vector from windowed frames."""

    frames = np.asarray(frames, dtype=float)

    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array with shape (num_frames, frame_length)")

    if frames.shape[0] == 0:
        raise ValueError("frames array is empty")

    magnitude = get_spectrogram(frames)
    freqs = np.fft.rfftfreq(frames.shape[1], d=1 / SR)
    mag_sum = np.sum(magnitude, axis=1) + 1e-12

    centroid = np.sum(magnitude * freqs, axis=1) / mag_sum

    bandwidth = np.sqrt(
        np.sum(magnitude * (freqs - centroid[:, np.newaxis]) ** 2, axis=1) / mag_sum
    )

    cumulative_energy = np.cumsum(magnitude, axis=1)
    threshold = 0.85 * cumulative_energy[:, -1]
    rolloff_indices = np.array([
        np.searchsorted(cumulative_energy[i], threshold[i])
        for i in range(cumulative_energy.shape[0])
    ])
    rolloff_indices = np.clip(rolloff_indices, 0, len(freqs) - 1)
    rolloff = freqs[rolloff_indices]

    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-12), axis=1))
    arithmetic_mean = np.mean(magnitude + 1e-12, axis=1)
    flatness = geometric_mean / arithmetic_mean

    zcr = _zero_crossing_rate(frames)
    energy = _short_time_energy(frames)
    autocorr = _autocorrelation_feature(frames)

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
    """Return names of the extracted features."""
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