"""
Filter design and analysis module — Person B.

Task 6: Design at least one digital filter.
Required deliverables:
    - Difference equation
    - Transfer function H(z)
    - Pole-zero plot
    - Magnitude response
    - Phase response
    - Group delay

Chosen filter:
    First-order FIR pre-emphasis filter

    Difference equation:
        y[n] = x[n] - alpha*x[n-1]

    Transfer function:
        H(z) = 1 - alpha*z^(-1)

For alpha = 0.97, the filter emphasizes high-frequency speech components.
"""

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def design_preemphasis_fir(alpha=0.97):
    """
    First-order FIR pre-emphasis filter.

    H(z) = 1 - alpha*z^(-1)

    Difference equation:
        y[n] = x[n] - alpha*x[n-1]

    Args:
        alpha: pre-emphasis coefficient, usually 0.95 to 0.97.

    Returns:
        b, a: filter coefficients.
    """
    b = np.array([1.0, -alpha])
    a = np.array([1.0])
    return b, a


def apply_preemphasis(x, alpha=0.97):
    """
    Apply the pre-emphasis filter to a 1D audio signal.

    Args:
        x: 1D input signal.
        alpha: pre-emphasis coefficient.

    Returns:
        y: filtered signal.
    """
    b, a = design_preemphasis_fir(alpha)
    return signal.lfilter(b, a, x)


def plot_pole_zero(b, a, save_path="results/figures/preemphasis_pole_zero.png"):
    """
    Plot and save the pole-zero diagram.
    """
    zeros, poles, _ = signal.tf2zpk(b, a)

    plt.figure(figsize=(6, 6))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 500)
    plt.plot(np.cos(theta), np.sin(theta), linestyle="--", label="Unit circle")

    # Plot zeros and poles
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
    """
    Plot and save magnitude and phase response.
    """
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
    """
    Plot and save group delay.
    """
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
    """
    Generate all required filter analysis plots.

    Returns:
        b, a: filter coefficients.
    """
    b, a = design_preemphasis_fir(alpha)

    plot_pole_zero(b, a)
    plot_frequency_response(b, a)
    plot_group_delay(b, a)

    return b, a


def print_filter_summary(alpha=0.97):
    """
    Print filter equations and basic properties.
    """
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