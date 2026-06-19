"""
Preprocessing module.

Pipeline: raw audio file -> normalized waveform -> windowed frames -> spectrogram.

Public interface (this is the contract teammates B and C build against):
    load_and_frame(filepath) -> frames, shape (num_frames, 400)
        400 = 25ms at 16kHz, already Hamming-windowed, 50% overlap.
    get_spectrogram(frames) -> magnitude spectrogram, shape (num_frames, 201)
        201 = 400 // 2 + 1 rfft bins (real-valued FFT, only positive freqs kept).
    get_log_spectrogram(frames) -> log-magnitude spectrogram, same shape as above.

Note: data/processed/{real,fake}/*.npy already contain the OUTPUT of
load_and_frame() (raw windowed frames, NOT spectrograms). Call
get_spectrogram() / get_log_spectrogram() yourself after loading if you
need frequency-domain features.
"""

import numpy as np
import librosa

SR = 16000          # target sample rate (Hz); ASVspoof2019 LA is natively 16kHz
FRAME_MS = 25        # frame length in milliseconds (course spec: 20-30ms)
OVERLAP = 0.5        # fractional overlap between consecutive frames


def load_audio(filepath, sr=SR):
    """Load an audio file, resample to `sr`, and peak-normalize amplitude.

    Peak normalization (divide by max |sample|) puts every file on a
    comparable amplitude scale regardless of original recording level,
    so downstream features aren't confounded by loudness differences
    between real and fake clips.

    Args:
        filepath: path to a .flac/.wav file.
        sr: target sample rate in Hz.

    Returns:
        1D np.ndarray of float samples, peak-normalized to [-1, 1].
    """
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def pre_emphasis(y, alpha=0.97):
    """Apply a first-order pre-emphasis filter: y[n] - alpha * y[n-1].

    This is a simple high-pass FIR filter (difference equation above,
    transfer function H(z) = 1 - alpha*z^-1) that boosts high frequencies,
    which are naturally attenuated in voiced speech. Optional — not
    applied by default in load_and_frame().

    Args:
        y: 1D waveform.
        alpha: pre-emphasis coefficient, typically 0.95-0.97.

    Returns:
        Filtered 1D waveform, same length as input.
    """
    return np.append(y[0], y[1:] - alpha * y[:-1])


def frame_signal(y, sr=SR, frame_ms=FRAME_MS, overlap=OVERLAP, window="hamming"):
    """Split a signal into overlapping, windowed frames.

    Why windowing matters: cutting a frame out of a continuous signal
    with a rectangular boundary introduces sharp discontinuities at the
    frame edges. The FFT assumes (implicitly) that the frame repeats
    periodically, so those edge discontinuities leak energy across many
    frequency bins (spectral leakage), distorting the true spectrum.
    A tapered window (Hamming/Hann) forces the frame to fade to ~0 at
    both edges, eliminating the discontinuity and producing a much
    cleaner FFT magnitude estimate. This is required before any
    frame-by-frame FFT/STFT analysis (Task 3 -> Task 4 dependency).

    Args:
        y: 1D waveform (already loaded/normalized).
        sr: sample rate in Hz, used to convert frame_ms to samples.
        frame_ms: frame length in milliseconds.
        overlap: fractional overlap between consecutive frames (0-1).
        window: "hamming" or "hann".

    Returns:
        np.ndarray of shape (num_frames, frame_len), each row a windowed frame.
    """
    frame_len = int(sr * frame_ms / 1000)        # e.g. 16000 * 25/1000 = 400 samples
    hop_len = int(frame_len * (1 - overlap))      # e.g. 400 * 0.5 = 200 samples (50% overlap)

    # Pad short signals so we always get at least one full frame.
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))

    # Standard overlapping-frame count. Floor division means any trailing
    # partial frame (shorter than frame_len) is dropped rather than
    # zero-padded — a deliberate simplification, not a bug.
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
    """Single entry point: filepath -> windowed frames.

    This is the function B and C should call to go from a raw file path
    straight to model-ready frames. It chains load_audio() -> (optional
    pre_emphasis()) -> frame_signal().

    Args:
        filepath: path to a .flac/.wav file.
        apply_preemph: if True, applies pre_emphasis() before framing.
        window: "hamming" or "hann", passed through to frame_signal().

    Returns:
        np.ndarray of shape (num_frames, 400) — windowed frames.
    """
    y = load_audio(filepath)
    if apply_preemph:
        y = pre_emphasis(y)
    return frame_signal(y, window=window)


def get_spectrogram(frames):
    """Compute the magnitude spectrogram from windowed frames via rFFT.

    rfft (rather than full fft) is used because the input is real-valued,
    so the negative-frequency half of the spectrum is a redundant mirror
    image and can be discarded.

    Args:
        frames: np.ndarray of shape (num_frames, frame_len), e.g. from load_and_frame().

    Returns:
        np.ndarray of shape (num_frames, frame_len // 2 + 1), magnitude per bin.
    """
    return np.abs(np.fft.rfft(frames, axis=1))


def get_log_spectrogram(frames):
    """Compute the log-magnitude spectrogram (dB-like scale).

    Taking the log compresses the dynamic range, which matches how
    energy differences are perceived and is the standard representation
    used in the Alzantot et al. baseline (log-magnitude STFT feature).
    A small epsilon (1e-9) avoids log(0) for silent frames.

    Args:
        frames: np.ndarray of shape (num_frames, frame_len).

    Returns:
        np.ndarray of shape (num_frames, frame_len // 2 + 1), in log scale.
    """
    spec = get_spectrogram(frames)
    return 20 * np.log10(spec + 1e-9)