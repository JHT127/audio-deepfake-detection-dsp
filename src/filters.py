"""
Filter design and analysis.

First-order FIR pre-emphasis: H(z) = 1 - alpha*z^(-1), y[n] = x[n] - alpha*x[n-1]
"""

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def design_preemphasis_fir(alpha=0.97):
    """First-order FIR pre-emphasis filter."""
    b = np.array([1.0, -alpha])
    a = np.array([1.0])
    return b, a


def apply_preemphasis(x, alpha=0.97):
    """Apply pre-emphasis to a 1D audio signal."""
    b, a = design_preemphasis_fir(alpha)
    return signal.lfilter(b, a, x)


def plot_pole_zero(b, a, save_path="results/figures/preemphasis_pole_zero.png"):
    """Plot and save the pole-zero diagram."""
    zeros, poles, _ = signal.tf2zpk(b, a)

    plt.figure(figsize=(6, 6))

    theta = np.linspace(0, 2 * np.pi, 500)
    plt.plot(np.cos(theta), np.sin(theta), linestyle="--", label="Unit circle")

    plt.scatter(np.real(zeros), np.imag(zeros), marker="o", s=80, label="Zeros")
    plt.scatter(np.real(poles), np.imag(poles), marker="x", s=80, label="Poles")

    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title("Pole-Zero Plot of Pre-emphasis FIR Filter")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_frequency_response(b, a, save_path="results/figures/preemphasis_magnitude_phase.png"):
    """Plot and save magnitude and phase response."""
    w, h = signal.freqz(b, a, worN=1024)

    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)
    phase = np.unwrap(np.angle(h))

    plt.figure(figsize=(8, 5))
    plt.plot(w / np.pi, magnitude_db)
    plt.xlabel("Normalized frequency (×π rad/sample)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude Response of Pre-emphasis FIR Filter")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path.replace(".png", "_magnitude.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(w / np.pi, phase)
    plt.xlabel("Normalized frequency (×π rad/sample)")
    plt.ylabel("Phase (radians)")
    plt.title("Phase Response of Pre-emphasis FIR Filter")
    plt.grid(True)

    plt.savefig(save_path.replace(".png", "_phase.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_delay(b, a, save_path="results/figures/preemphasis_group_delay.png"):
    """Plot and save group delay."""
    w, gd = signal.group_delay((b, a), w=1024)

    plt.figure(figsize=(8, 5))
    plt.plot(w / np.pi, gd)
    plt.xlabel("Normalized frequency (×π rad/sample)")
    plt.ylabel("Group delay (samples)")
    plt.title("Group Delay of Pre-emphasis FIR Filter")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_preemphasis_filter(alpha=0.97):
    """Generate filter analysis plots. Returns (b, a) coefficients."""
    b, a = design_preemphasis_fir(alpha)

    plot_pole_zero(b, a)
    plot_frequency_response(b, a)
    plot_group_delay(b, a)

    return b, a


def print_filter_summary(alpha=0.97):
    """Print filter equations and properties."""
    b, a = design_preemphasis_fir(alpha)
    zeros, poles, _ = signal.tf2zpk(b, a)

    print("Pre-emphasis FIR filter")
    print("-----------------------")
    print(f"Difference equation: y[n] = x[n] - {alpha}x[n-1]")
    print(f"Transfer function: H(z) = 1 - {alpha}z^-1")
    print(f"Numerator coefficients b: {b}")
    print(f"Denominator coefficients a: {a}")
    print(f"Zeros: {zeros}")
    print(f"Poles: {poles}")

    if len(poles) == 0 or np.all(np.abs(poles) < 1):
        print("Stability: Stable")
    else:
        print("Stability: Unstable")

    print("Filter type: High-pass / pre-emphasis FIR filter")


if __name__ == "__main__":
    b, a = analyze_preemphasis_filter(alpha=0.97)
    print_filter_summary(alpha=0.97)
    print("Filter plots saved to results/figures/")