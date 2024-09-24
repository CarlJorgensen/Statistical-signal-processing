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


def melody2frequency(melodies: list, frequencies: dict) -> np.array:
    """
    Parse the notes in the melodies.txt file and convert to their frequencies.

    NOTE: Taken from SignalGenerator.py line 34-37
    """

    # Convert notes to frequencies
    melodies_f = np.zeros(shape=(len(melodies), len(melodies[0])))
    for i, m in enumerate(melodies):
        for j, n in enumerate(m):
            melodies_f[i][j] = frequencies[n]

    melody_mm1 = 0.975*melodies_f
    melody_mm2 = 1.025*melodies_f

    return melody_mm1, melody_mm2


def generate_matrix(melody, melody_mm1: np.array, melody_mm2: np.array,
                    fs: int) -> np.array:
    """
    Generate the matrix H
    """
    nr_samples = len(melody)
    nr_tones = 12  # all melodies have 12 tones
    tone = melody[:int(nr_samples/nr_tones)]
    #nr_tone_samples = int(len(tone)/10)
    nr_tone_samples = len(tone)
    melody_mm = np.vstack((melody_mm1, melody_mm2))
    H = np.zeros(shape=(20, nr_tones, nr_tone_samples, 2))

    for i, m in enumerate(melody_mm):
        for j, f in enumerate(m):  # Each frequency f corresponds to a tone
            for k in range(nr_tone_samples):
                H[i, j, k, 0] = np.cos(2*np.pi*f/fs * k)
                H[i, j, k, 1] = np.sin(2*np.pi*f/fs * k)

    return H


def divide_melody(melody: np.array) -> np.array:
    """
    Divide the melody into 12 tones
    (nr_tones x nr_tone_samples)
    """
    nr_samples = len(melody)
    nr_tones = 12  # all melodies have 12 tones
    tone = melody[:int(nr_samples/nr_tones)]
    #nr_tone_samples = int(len(tone)/10)
    nr_tone_samples = len(tone)

    y = np.zeros(shape=(nr_tones, nr_tone_samples))

    for m in range(nr_tones):
        for n in range(nr_tone_samples):
            y[m][n] = melody[m*nr_tone_samples + n]

    return y


def classifier(y: np.array, H: np.array) -> int:
    """
    argmax_{j} sum_{0}{nr_tones}(||H_{n,j} * y_{n}||^2)
    """

    j_hat = 0
    max_sum = 0
    alpha = 0

    for j in range(H.shape[0]):  # H.shape[0] is 20
        current_sum = 0

        # Iterate over each tone (12 tones in total)
        for n in range(H.shape[1]):  # H.shape[1] is 12 tones
            H_nj = H[j, n]  # Shape (nr_tone_samples, 2)
            y_n = y[n]  # Shape (nr_tone_samples,)

            H_nj = np.transpose(H_nj)

            norm_value = np.linalg.norm(np.matmul(H_nj, y_n)) ** 2
            current_sum += norm_value

        if current_sum > max_sum:
            max_sum = current_sum
            if j >= 10:
                j_hat = j - 10
                alpha = 1.025
            else:
                j_hat = j
                alpha = 0.975

    return j_hat, alpha
