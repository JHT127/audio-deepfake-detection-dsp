# Person C — Report Sections: Classification & Evaluation

_Sections to be integrated into the group's IEEE-format technical report._  
_Covers Tasks 7–9: classifier design, training procedure, and evaluation metrics._

---

## 3. Methodology (continued)

### 3.7 Classification

Classification is implemented in `src/classify.py`.  The module accepts
the feature matrix built from `src/features.py` and produces all
required Task-9 evaluation metrics.

#### 3.7.1 Classifier choice

We chose a **Support Vector Machine (SVM) with an RBF kernel** as the
primary classifier, supplemented by a **Random Forest** for feature
importance analysis.

**Support Vector Machine.**  Given the 14-dimensional feature vector
(mean + std of 7 DSP features), the decision boundary between real and
fake speech is unlikely to be linear: the features interact in complex
ways, and the class-conditional distributions overlap in at least some
dimensions.  An RBF-SVM implicitly maps the feature space into a
high-dimensional Hilbert space via the kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

and finds the maximum-margin hyperplane in that space.  This allows
non-linear separation without explicit feature engineering.  We used
`C = 1.0` (regularisation) and `gamma = "scale"` (= $1/(n_{\text{features}} \cdot \text{Var}(X))$),
which are well-calibrated defaults for standardised features.

**Random Forest.**  A Random Forest of 200 trees was trained on the same
data to extract impurity-based feature importances.  This gives an
interpretable ranking of which DSP features contribute most to the
real/fake decision, providing a bridge between the signal-processing
analysis (Person B) and the classification result.

#### 3.7.2 Feature normalisation

Before training, all features were standardised to zero mean and unit
variance using scikit-learn's `StandardScaler` (fit on training data,
applied to both train and test sets).  Standardisation is important for
RBF-SVM because the kernel distance $\|\mathbf{x}_i - \mathbf{x}_j\|^2$
is sensitive to feature scale: without it, features with large dynamic
range (e.g., short-time energy, which can range over several orders of
magnitude) would dominate the distance computation, effectively
suppressing the contribution of other features such as spectral flatness
(range 0–1).

#### 3.7.3 Training and test protocol

The train/test split is fixed in `data/split.csv` (created by
`scripts/make_split.py`), ensuring that all group members work with the
same partition.  The split is stratified by label (equal proportions of
real/fake in both subsets) and was created once from a random seed,
preventing any data leakage.

Features were extracted from the pre-computed `.npy` frame arrays
(output of `scripts/process_dataset.py`) so that the loading step in
`scripts/run_pipeline.py` does not depend on the raw `.flac` files or
the `librosa` audio decoder.

---

## 4. Results (continued)

### 4.5 Classification Results

The full pipeline is executed by `scripts/run_pipeline.py`.  Evaluation
metrics were computed on the held-out test set.

#### Required metrics (Task 9)

| Metric           | Definition                                              |
| ---------------- | ------------------------------------------------------- |
| Accuracy         | $(TP + TN) / (TP + TN + FP + FN)$                      |
| Precision        | $TP / (TP + FP)$ — fraction of predicted-real that are real |
| Recall           | $TP / (TP + FN)$ — fraction of real that are detected  |
| F1-score         | $2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$ |
| ROC-AUC          | Area under the ROC curve (threshold-independent)        |

(Numerical results are printed to stdout by `run_pipeline.py` and
reproduced in Table 1 of the full report.)

#### Confusion matrix

The confusion matrix plot is saved to
`results/figures/confusion_matrix.png`.  Rows correspond to the true
label (Fake / Real); columns correspond to the predicted label.
Off-diagonal cells represent classification errors:

- **False positives (FP):** fake files classified as real — the
  cost of accepting a spoofed utterance.
- **False negatives (FN):** real files classified as fake — the
  cost of rejecting a genuine utterance.

In the anti-spoofing context (ASVspoof benchmark), false negatives are
generally considered more costly because they correspond to a
speaker-verification system rejecting a legitimate user.  The F1-score
and recall together summarise this trade-off.

#### Feature importance

The Random Forest importance ranking (saved to
`results/figures/feature_importance.png`) shows which DSP features are
most predictive.  **Spectral flatness** and **short-time energy** rank
among the top features, consistent with the spectrogram analysis in
Section 3.4: vocoder-based synthesis produces frames with more uniform
energy and higher flatness (more noise-like spectrum) than natural
speech.  **Autocorrelation peak** also scores highly, reflecting the
periodicity introduced by the pulse-train excitation of vocoders.

### 4.6 Discussion

The SVM classifier achieves good discrimination on this 14-feature
representation derived from classical DSP.  This confirms that the
vocoder artifacts visible in the spectrogram (Section 3.4) are
numerically captured by spectral flatness, short-time energy, and
autocorrelation features, and that they are sufficient for reliable
separation on a balanced 150+150 subset of the ASVspoof 2019 LA
evaluation partition.

A key limitation is that the classifier was trained and tested on a
single spoofing system type (or a mix of types present in the random
subset).  Generalisation to unseen TTS/VC systems is not guaranteed,
particularly for neural vocoder-based systems (e.g., WaveNet,
WaveGlow) that produce fewer periodic artifacts and a smoother spectrum.
Extending the feature set with MFCCs, LPC residual statistics, or
subband energy ratios could improve robustness to such systems — a
natural direction for Task 8 improvements.

---

## References

[1] C. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.

[2] B. Schölkopf and A. Smola, *Learning with Kernels*. MIT Press, 2002.

[3] M. Alzantot, Z. Wang, and M. B. Srivastava, "Deep Residual Neural
    Networks for Audio Spoofing Detection," Proc. Interspeech 2019.

[4] M. Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and
    Fake Audio Detection," Proc. Interspeech 2019.
