"""Generate comparison plots: time domain, FFT magnitude, spectrogram."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import load_audio, load_and_frame, get_log_spectrogram, SR

REAL_DIR = "data/raw/real"
FAKE_DIR = "data/raw/fake"
OUT_DIR = "results/figures"


def pick_one_file(directory):
    files = sorted(f for f in os.listdir(directory) if f.endswith(".flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found in {directory}")
    return os.path.join(directory, files[0])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    real_path = pick_one_file(REAL_DIR)
    fake_path = pick_one_file(FAKE_DIR)
    print(f"Real file: {real_path}")
    print(f"Fake file: {fake_path}")

    real_audio = load_audio(real_path)
    fake_audio = load_audio(fake_path)
    sr = SR

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(np.arange(len(real_audio)) / sr, real_audio, linewidth=0.5, color="tab:blue")
    axes[0].set_title("Real speech — time domain")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(np.arange(len(fake_audio)) / sr, fake_audio, linewidth=0.5, color="tab:red")
    axes[1].set_title("Fake (spoofed) speech — time domain")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "time_domain_real_vs_fake.png"), dpi=150)
    plt.close(fig)
    print("Saved: time_domain_real_vs_fake.png")

    real_frames = load_and_frame(real_path)
    fake_frames = load_and_frame(fake_path)

    frame_idx = min(10, real_frames.shape[0] - 1, fake_frames.shape[0] - 1)
    real_frame = real_frames[frame_idx]
    fake_frame = fake_frames[frame_idx]

    n_fft = real_frame.shape[0]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    real_mag = np.abs(np.fft.rfft(real_frame))
    fake_mag = np.abs(np.fft.rfft(fake_frame))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, real_mag, label="Real", color="tab:blue")
    ax.plot(freqs, fake_mag, label="Fake", color="tab:red", alpha=0.8)
    ax.set_title(f"FFT magnitude — single frame (index {frame_idx})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fft_magnitude_real_vs_fake.png"), dpi=150)
    plt.close(fig)
    print("Saved: fft_magnitude_real_vs_fake.png")

    real_log_spec = get_log_spectrogram(real_frames)
    fake_log_spec = get_log_spectrogram(fake_frames)

    frame_len_s = real_frames.shape[1] / sr
    hop_s = frame_len_s / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    im = None
    for ax, spec, title in zip(axes, [real_log_spec, fake_log_spec], ["Real", "Fake"]):
        n_frames = spec.shape[0]
        extent = [0, n_frames * hop_s, 0, sr / 2]
        im = ax.imshow(spec.T, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax.set_title(f"{title} — log spectrogram")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=list(axes), label="Log magnitude")
    fig.savefig(os.path.join(OUT_DIR, "spectrogram_real_vs_fake.png"), dpi=150)
    plt.close(fig)
    print("Saved: spectrogram_real_vs_fake.png")


if __name__ == "__main__":
    main()