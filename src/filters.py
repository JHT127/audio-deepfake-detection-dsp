"""
Filter design and analysis module — Person B.

Task 6: Design at least one digital filter.
Required deliverables per the project brief:
    - Difference equation
    - Transfer function H(z)
    - Pole-zero plot
    - Magnitude response
    - Phase response
    - Group delay

This module should contain the filter design code and produce
the required analysis plots (save to results/figures/).
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def design_preemphasis_fir(alpha=0.97):
    """First-order FIR pre-emphasis filter: H(z) = 1 - alpha*z^{-1}.
    Already implemented in src/preprocessing.pre_emphasis() as a NumPy
    operation — duplicate here as a scipy filter object for formal analysis.

    Returns:
        b, a: FIR numerator/denominator coefficients (a=[1]).
    """
    raise NotImplementedError("Person B: implement filter design here")