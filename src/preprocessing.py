"""
Preprocessing module.

Public interface:
    load_and_frame(filepath) -> frames (num_frames, frame_len)
    get_spectrogram(frames) -> magnitude spectrogram (num_frames, freq_bins)
    get_log_spectrogram(frames) -> log-magnitude spectrogram
"""

import numpy as np
import librosa

SR = 16000
FRAME_MS = 25
OVERLAP = 0.5


def load_audio(filepath, sr=SR):
    """Load audio, resample to target sr, peak-normalize amplitude."""
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def pre_emphasis(y, alpha=0.97):
    """Pre-emphasis filter: y[n] - alpha * y[n-1]."""
    return np.append(y[0], y[1:] - alpha * y[:-1])


def frame_signal(y, sr=SR, frame_ms=FRAME_MS, overlap=OVERLAP, window="hamming"):
    """Split signal into overlapping, windowed frames.
    Returns shape (num_frames, frame_len).
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(frame_len * (1 - overlap))

    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))

    num_frames = 1 + (len(y) - frame_len) // hop_len

    if window == "hamming":
        win = np.hamming(frame_len)
    elif window == "hann":
        win = np.hanning(frame_len)
    else:
        raise ValueError(f"Unsupported window: {window}")

    frames = np.stack([
        y[i * hop_len: i * hop_len + frame_len] * win
        for i in range(num_frames)
    ])
    return frames


def load_and_frame(filepath, apply_preemph=False, window="hamming"):
    """Single entry point: filepath -> windowed frames."""
    y = load_audio(filepath)
    if apply_preemph:
        y = pre_emphasis(y)
    return frame_signal(y, window=window)


def get_spectrogram(frames):
    """frames (num_frames, frame_len) -> magnitude spectrogram (num_frames, freq_bins)."""
    return np.abs(np.fft.rfft(frames, axis=1))


def get_log_spectrogram(frames):
    spec = get_spectrogram(frames)
    return 20 * np.log10(spec + 1e-9)