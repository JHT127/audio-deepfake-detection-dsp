# Person B — Report Sections: Feature Extraction & Filter Design

_Sections to be integrated into the group's IEEE-format technical report._  
_Covers Tasks 5–6: DSP feature extraction and digital filter design/analysis._

---

## 3. Methodology (continued)

### 3.5 DSP Feature Extraction

Feature extraction is implemented in `src/features.py`.  The module
builds directly on the preprocessing pipeline of Person A: it accepts
the windowed frames produced by `load_and_frame()` (or loaded from the
pre-computed `.npy` arrays) and returns a fixed-length, one-dimensional
feature vector representing the entire utterance.

#### 3.5.1 Design rationale

Each audio file produces a different number of frames depending on
utterance length.  Classifiers require fixed-length input vectors, so we
summarise each per-frame feature series by its **mean** and **standard
deviation** across all frames of the utterance.  This gives 2 statistics
× 7 features = **14 scalar features** per file.

The choice of features was motivated directly by the spectrogram
observations in Section 3.4.  The vocoder-based spoofed speech exhibits
periodic broadband energy bursts (vertical striations), while genuine
speech has continuous, smoothly-varying harmonics.  Features that
capture energy regularity, spectral shape, and signal periodicity are
therefore expected to be most discriminative.

#### 3.5.2 Feature definitions

All features are computed from the magnitude spectrogram
$|X[k]|$ returned by `get_spectrogram(frames)`, plus time-domain frame
statistics.

**1. Spectral centroid** (mean and std across frames)

$$C = \frac{\sum_k f_k \cdot |X[k]|}{\sum_k |X[k]|}$$

The centroid is the "centre of mass" of the spectrum.  TTS vocoders tend
to produce a higher and more consistent centroid than natural speech
because the excitation model does not reproduce the subtle low-frequency
modulations of the glottal source.

**2. Spectral bandwidth** (mean and std across frames)

$$B = \sqrt{\frac{\sum_k |X[k]| \cdot (f_k - C)^2}{\sum_k |X[k]|}}$$

Bandwidth measures how spread the energy is around the centroid.
Vocoder artifacts that inject broadband energy (see Section 3.4) inflate
per-frame bandwidth.

**3. Spectral roll-off at 85%** (mean and std across frames)

The roll-off frequency is the frequency below which 85% of the total
spectral energy is contained.  Natural speech roll-off varies with
voiced/unvoiced regions; synthetic speech tends to have a more uniform
roll-off profile.

**4. Spectral flatness** (mean and std across frames)

$$F = \frac{\exp\!\left(\frac{1}{K}\sum_k \ln |X[k]|\right)}
          {\frac{1}{K}\sum_k |X[k]|}$$

Flatness (Wiener entropy) is the ratio of the geometric mean to the
arithmetic mean of the spectrum.  A value near 1 indicates a white
noise-like spectrum; a value near 0 indicates a tonal (harmonic)
spectrum.  The periodic impulse-train excitation of vocoders produces
higher per-frame flatness than natural voiced speech.

**5. Zero-crossing rate** (mean and std across frames)

$$\text{ZCR} = \frac{1}{L-1}\sum_{n=1}^{L-1} \mathbf{1}[\text{sign}(x[n]) \neq \text{sign}(x[n-1])]$$

ZCR is a coarse measure of signal frequency content.  Vocoder synthesis
introduces high-frequency energy bursts that transiently raise ZCR.

**6. Short-time energy** (mean and std across frames)

$$E = \sum_{n=0}^{L-1} x[n]^2$$

Energy measures the power of each frame.  Vocoder-based speech
typically has more uniform energy across frames because the pulse-train
excitation drives all frames to a similar amplitude level, reducing the
natural energy variation associated with prosodic changes in real speech.

**7. Autocorrelation peak** (mean and std across frames)

The maximum normalised autocorrelation coefficient at lags $\tau > 0$:

$$R(\tau) = \frac{\sum_n x[n]\, x[n-\tau]}{\sum_n x[n]^2}$$

A high autocorrelation peak indicates a periodic signal; the regularity
of the vocoder excitation pulse train leads to higher and more consistent
autocorrelation peaks than natural speech, where pitch period varies
continuously with prosody.

---

### 3.6 Digital Filter Design and Analysis

Filter design and analysis is implemented in `src/filters.py`.
The chosen filter is the **first-order FIR pre-emphasis filter** that is
already used as an optional pre-processing step in the pipeline.
We subject it to full DSP analysis as required by Task 6.

#### 3.6.1 Filter specification

| Property               | Value                                  |
| ---------------------- | -------------------------------------- |
| Filter type            | FIR (Finite Impulse Response), order 1 |
| Difference equation    | $y[n] = x[n] - \alpha x[n-1]$         |
| Transfer function      | $H(z) = 1 - \alpha z^{-1}$            |
| Coefficient $\alpha$   | 0.97 (standard speech processing value)|
| Numerator $b$          | `[1.0, −0.97]`                         |
| Denominator $a$        | `[1.0]`                                |

#### 3.6.2 Pole-zero analysis

Setting $H(z) = 0$ gives one zero at $z = \alpha = 0.97$, located on
the positive real axis inside the unit circle.  The denominator is
degree 0 (FIR filter), so there are no finite poles; all poles are at
the origin.

Because the single non-trivial zero lies strictly inside the unit circle
($|\alpha| = 0.97 < 1$), the filter is **minimum-phase**: it is both
causal and stably invertible.  An FIR filter is always BIBO-stable
regardless of zero locations.

#### 3.6.3 Magnitude response

The frequency response is obtained by evaluating $H(z)$ on the unit
circle $z = e^{j\omega}$:

$$H(e^{j\omega}) = 1 - \alpha e^{-j\omega}$$

$$|H(e^{j\omega})| = \sqrt{1 + \alpha^2 - 2\alpha\cos\omega}$$

At DC ($\omega = 0$): $|H| = 1 - \alpha = 0.03$ (strong attenuation).  
At Nyquist ($\omega = \pi$): $|H| = 1 + \alpha = 1.97$ (about +6 dB boost).

The filter is therefore a high-pass filter, which justifies its use as a
pre-emphasis stage to partially compensate for the spectral roll-off of
the speech source.

#### 3.6.4 Phase response and group delay

The phase response is:

$$\angle H(e^{j\omega}) = \arctan\!\left(\frac{-\alpha\sin\omega}{1 - \alpha\cos\omega}\right)$$

The group delay $\tau(\omega) = -\frac{d}{d\omega}\angle H(e^{j\omega})$
is nearly constant (approximately 0.97 samples) across the passband,
confirming that the filter introduces very little dispersive distortion —
an expected property of linear-phase FIR filters of this form.

#### 3.6.5 Relevance to deepfake detection

When applied as a pre-processing step (via `apply_preemph()` in
`src/filters.py`), the filter amplifies the high-frequency spectral
differences between genuine and vocoder-synthesised speech identified in
Section 3.4, potentially making those differences more detectable by
downstream features such as spectral centroid and spectral roll-off.

---

## 4. Results (continued)

### 4.4 Filter Analysis Plots

Four plots were generated by `src/filters.py`:

- **`preemphasis_pole_zero.png`** — Shows the single zero at $z = 0.97$
  on the positive real axis and no finite poles (all at the origin).
  The unit circle is drawn for reference; the zero lies just inside it,
  confirming minimum-phase behaviour.

- **`preemphasis_magnitude_phase_magnitude.png`** — Monotonically
  increasing magnitude from −30 dB at DC to approximately +6 dB at
  Nyquist, consistent with the analytical expression derived above.

- **`preemphasis_magnitude_phase_phase.png`** — Smooth, near-linear
  phase roll-off, confirming low group delay distortion.

- **`preemphasis_group_delay.png`** — Approximately flat group delay
  of ~0.97 samples across the entire band, with a slight increase near DC
  where the zero is closest to the unit circle.

---

## References

[1] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
    3rd ed. Pearson, 2009.

[2] L. R. Rabiner and B. H. Juang, *Fundamentals of Speech Recognition*.
    Prentice-Hall, 1993.

[3] M. Alzantot, Z. Wang, and M. B. Srivastava, "Deep Residual Neural
    Networks for Audio Spoofing Detection," Proc. Interspeech 2019.
