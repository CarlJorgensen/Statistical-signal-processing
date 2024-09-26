#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

"""
TODO:
 1. Generate the matrix H_t
    > What is the frequency? f_{n,j}

 2. Implement the classifier
    > Need to determine the tuning mismatch alpha ()= 0.975 or 1.025).
    > Need to determine the melody m
        - argmax_j sum(||H_t * y_n||^2)
"""

"""
Signal generator:
- Generates one of the m melodies, selected uniformly at random.
- Generated melody is off pitch by a factor of 0.975 or 1.025, but never 1.

"""

"""
Iteration variables:
- m=10: number of melodies
- l=2: number of pitch mismatches
- j=20: number of hypotheses (=m*l), equally likely.
- k: number of samples
"""

def generate_matrix(nr_tone_samples: int, alpha: float, 
                    f_nj: int, fs: int) -> np.array:
    """
    Generates observation matrix for single tone detection.
    alpha = mismatch, f_nj = fundamental freq.
    """
    H = np.zeros((nr_tone_samples, 2))
    for k in range(nr_tone_samples):
        H[k, 0] = np.cos(2 * alpha * np.pi * f_nj/fs * k)
        H[k, 1] = np.sin(2 * alpha * np.pi * f_nj/fs * k)

    return H


def classifier_single(melodies: list, melody: list, 
                      K: int, frequencies: dict) -> int:
    """
    argmax_{j} sum_{0}{nr_tones}(||H_{n,j} * y_{n}||^2)
    """

    nr_samples = len(melody)
    nr_tones = 12 # all melodies have 12 tones
    tone = melody[:int(nr_samples/nr_tones)]
    nr_tone_samples = int(len(tone)/K)

    mismatches = [0.975, 1.025]

    j_hat = None
    alpha_hat = None
    max_sum = -10000

    for alpha in mismatches:
        for j, notes in enumerate(melodies):
            curr_sum = 0
            for i in range(nr_tones):
                y = melody[i*nr_tone_samples: (i+1)*nr_tone_samples]
                note = notes[i]
                H_nj = generate_matrix(nr_tone_samples, alpha, frequencies[note], 8820)

                H_nj = np.transpose(H_nj)
                curr_sum += np.linalg.norm(np.matmul(H_nj, y)) ** 2

            if curr_sum > max_sum:
                max_sum = curr_sum
                j_hat = j
                alpha_hat = alpha
    

    return j_hat, alpha


