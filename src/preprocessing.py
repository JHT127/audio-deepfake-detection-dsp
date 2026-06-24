"""
Preprocessing: raw audio -> normalized waveform -> windowed frames -> spectrogram.

Public interface:
    load_and_frame(filepath) -> frames, shape (num_frames, 400)
    get_spectrogram(frames) -> magnitude spectrogram, shape (num_frames, 201)
    get_log_spectrogram(frames) -> log-magnitude spectrogram, same shape

data/processed/{real,fake}/*.npy contain windowed frames from load_and_frame(),
not spectrograms. Call get_spectrogram() after loading if needed.
"""

import numpy as np
import librosa

SR = 16000          # target sample rate (Hz); ASVspoof2019 LA is natively 16kHz
FRAME_MS = 25        # frame length in milliseconds (course spec: 20-30ms)
OVERLAP = 0.5        # fractional overlap between consecutive frames


def load_audio(filepath, sr=SR):
    """Load audio, resample to sr, and peak-normalize."""
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def pre_emphasis(y, alpha=0.97):
    """Pre-emphasis filter."""
    return np.append(y[0], y[1:] - alpha * y[:-1])


def frame_signal(y, sr=SR, frame_ms=FRAME_MS, overlap=OVERLAP, window="hamming"):
    """Split signal into overlapping windowed frames."""
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
    """Load audio file and return windowed frames, shape (num_frames, 400)."""
    y = load_audio(filepath)
    if apply_preemph:
        y = pre_emphasis(y)
    return frame_signal(y, window=window)


def get_spectrogram(frames):
    """Magnitude spectrogram via rFFT."""
    return np.abs(np.fft.rfft(frames, axis=1))


def get_log_spectrogram(frames):
    """Log-magnitude spectrogram."""
    spec = get_spectrogram(frames)
    return 20 * np.log10(spec + 1e-9)