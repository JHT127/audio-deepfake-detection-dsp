# Person A — Report Sections: Methodology & Results

_Sections to be integrated into the group's IEEE-format technical report._
_Covers Tasks 1–4: dataset preparation, preprocessing, framing/windowing, FFT/STFT analysis._

---

## 3. Methodology

### 3.1 Dataset Preparation

We used the ASVspoof 2019 Logical Access (LA) evaluation partition, obtained from the Edinburgh DataShare repository [1]. The full archive is approximately 7.1 GB, so rather than downloading it in its entirety, we used HTTP range requests via the Python `remotezip` library to selectively extract only the required files. A protocol file (`ASVspoof2019.LA.cm.eval.trl.txt`) lists all evaluation utterances with their ground-truth labels; from this we randomly selected 150 bonafide (genuine) and 150 spoofed utterances, giving a balanced subset of 300 files.

All audio files are stored in FLAC format and are natively sampled at 16 kHz. This sampling rate was verified programmatically for every file before further processing. The 300 files were split into training and test sets, and a manifest (`data/split.csv`) was generated with three columns — `filename`, `label` (real/fake), and `split` (train/test) — to serve as the shared data contract for all group members.

### 3.2 Preprocessing

Each audio file was processed through the following steps, implemented in `src/preprocessing.py`:

**Loading and resampling.** Files were loaded with `librosa`, resampling to 16 kHz where necessary (all source files were already 16 kHz). The function returns a 1-D NumPy array; the sampling rate is a module-level constant (`SR = 16000`) rather than a per-call return value, enforcing a single consistent rate across the pipeline.

**Amplitude normalization.** Each waveform was peak-normalized by dividing by its maximum absolute amplitude, scaling all signals to the range $[-1, 1]$. This removes recording-level gain differences between speakers and systems, ensuring that downstream feature magnitudes reflect spectral shape rather than recording loudness.

**Pre-emphasis (optional).** A first-order FIR high-pass filter of the form

$$y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha = 0.97$$

was made available as an optional pre-processing step. Pre-emphasis partially compensates for the natural roll-off of the speech spectral envelope, boosting high-frequency components and improving the numerical conditioning of spectral estimates.

### 3.3 Framing and Windowing

Short-time analysis was performed by dividing each normalized waveform into overlapping frames before computing any spectral transform. The framing parameters follow standard speech processing practice and the conventions of the reference paper [2]:

| Parameter        | Value                                   |
| ---------------- | --------------------------------------- |
| Frame length     | 25 ms → **400 samples** at 16 kHz       |
| Frame hop (step) | 12.5 ms → **200 samples** (50% overlap) |
| Window function  | Hamming                                 |
| FFT size         | 400 (same as frame length)              |
| rfft output bins | **201** (400/2 + 1)                     |

The number of complete frames for a signal of $N$ samples is computed as:

$$N_{\text{frames}} = \left\lfloor \frac{N - L}{H} \right\rfloor + 1$$

where $L = 400$ is the frame length and $H = 200$ is the hop size. Any trailing partial frame (with fewer than $L$ samples) is discarded using integer (floor) division, avoiding the need for zero-padding at the signal boundary.

**Why windowing is required before FFT.** The DFT assumes the signal is periodic within the analysis window. When a rectangular (implicit) window is applied to a finite frame, the sharp discontinuities at the frame edges introduce spectral leakage: energy from a sinusoid at one frequency spreads into neighboring frequency bins, masking weaker spectral components. A Hamming window tapers the signal smoothly to near-zero at both endpoints:

$$w[n] = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{L - 1}\right), \quad 0 \le n \le L-1$$

This taper eliminates the edge discontinuities, reducing the sidelobe level of each spectral bin and preserving the frequency resolution needed to distinguish fine-grained spectral differences between genuine and spoofed speech. For audio deepfake detection, the artifacts that distinguish TTS/VC output from natural speech are often subtle frequency-domain patterns; spectral leakage without windowing would smear these patterns and reduce detectability.

The processed frames for all 300 files were saved as NumPy arrays (`.npy`) in `data/processed/{real,fake}/`, each with shape `(num_frames, 400)`. These arrays are the input to all downstream feature extraction and classification steps.

### 3.4 FFT and STFT Analysis

The magnitude spectrum of each frame was computed using NumPy's `rfft`, which returns the positive-frequency half of the DFT for a real-valued input:

$$X[k] = \sum_{n=0}^{L-1} x[n]\, w[n]\, e^{-j2\pi kn/L}, \quad k = 0, 1, \ldots, 201$$

Two derived representations are provided:

- **Magnitude spectrogram:** $|X[k]|$, shape `(num_frames, 201)`
- **Log-magnitude spectrogram:** $\log_{10}(|X[k]| + \epsilon)$ with $\epsilon = 10^{-10}$ for numerical stability, same shape

The log-magnitude representation matches the feature used by Alzantot et al. [2] for their Spec-ResNet model. In our system it serves as the primary input representation for feature extraction.

---

## 4. Results

### 4.1 Time-Domain Comparison

Figure 1 shows one-second segments of a genuine (real) and a spoofed (fake) speech signal in the time domain. Both waveforms exhibit the expected quasi-periodic voiced-speech structure during vowel regions, with irregular aperiodic segments corresponding to consonants. At this scale the two signals are visually difficult to distinguish, which is consistent with the challenge motivating the ASVspoof benchmark: modern TTS and voice-conversion systems produce waveforms that closely resemble natural speech in amplitude and temporal envelope. This confirms that time-domain amplitude features alone are insufficient for reliable deepfake detection, motivating the spectral and cepstral analysis in the following subsections.

### 4.2 FFT Magnitude Comparison

Figure 2 shows the average FFT magnitude spectrum (averaged across all frames) for a representative genuine and a representative spoofed file. Both spectra follow the expected spectral envelope of speech, with high energy in the low-frequency region (fundamental frequency and first few formants) and a gradual roll-off toward higher frequencies. However, the genuine signal exhibits smoother harmonic peaks with more natural inter-harmonic valleys, while the spoofed signal shows subtle but consistent differences in the relative heights of harmonic peaks, particularly above 3 kHz. These high-frequency differences are characteristic of TTS vocoders, which reconstruct the high-frequency envelope using statistical models that do not perfectly replicate the natural excitation source.

The pre-emphasis filter, when applied, partially amplifies these high-frequency differences, making them more prominent in subsequent feature extraction.

### 4.3 Spectrogram Comparison

Figure 3 shows the log-magnitude spectrograms of the same genuine and spoofed signals. This is the most diagnostically informative of the three visualisations.

The **genuine signal** (top panel) displays continuous, smoothly-varying harmonic bands across time. The harmonics are visible as horizontal stripes of high energy at multiples of the fundamental frequency. The harmonic structure evolves naturally with the pitch and vocal tract configuration of the speaker, and the inter-harmonic regions show gradual, organic transitions.

The **spoofed signal** (bottom panel) exhibits a strikingly different texture: a pattern of **comb-like vertical striations** — narrow, regularly-spaced columns of elevated energy that repeat at roughly uniform time intervals across the entire utterance. This pattern is a characteristic artifact of **vocoder-based speech synthesis**, where a pulse-train excitation signal (a periodic impulse sequence) is used to drive the synthesis filter. Each impulse in the excitation train produces a brief broadband energy burst that spans all frequency bins simultaneously, creating the vertical stripes visible in the spectrogram. The regularity and uniformity of these striations contrast sharply with the continuous harmonic bands of natural speech, providing a clear visual basis for detection.

This observation has direct implications for feature design: frame-to-frame energy variance, short-time autocorrelation of the spectral magnitude, and spectral flatness are all likely to differ significantly between genuine and vocoder-synthesized speech due to this artifact, and these are therefore prioritised features for Person B's extraction stage.

---

## References

[1] Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald. "CSTR VCTK Corpus / ASVspoof 2019 LA." University of Edinburgh, DataShare, 2019. https://doi.org/10.7488/ds/2555

[2] M. Alzantot, Z. Wang, and M. B. Srivastava, "Deep Residual Neural Networks for Audio Spoofing Detection," Proc. Interspeech 2019, pp. 1078–1082.

[3] M. Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Proc. Interspeech 2019.
