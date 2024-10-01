#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

"""
TODO
Single:
    1[ok]. Generate the matrix H
        > What is the frequency? f_{n,j}

    2[ok]. Implement the classifier
        > Need to determine the tuning mismatch alpha ()= 0.975 or 1.025).
        > Need to determine the melody m
            - argmax_j sum(||H_t * y_n||^2)

    3[x]. Clean up code. Fix hard coded constants.

    4[x]. Comment code and note argument and return types

Three:
    1[x]. Generate matrix H

    2[x]. Implement classifier.
"""

"""
NOTE Remeber flake!

there is pause note in melody

"""

"""
Signal generator:
- Generates one of the m melodies, selected uniformly at random.
- Generated melody is off pitch by a factor of 0.975 or 1.025, but never 1.

"""

"""
Iteration variables:
- m=10: number of melodies
- n=12: number of tones
- l=2: number of pitch mismatches
- j=20: number of hypotheses (=m*l), equally likely.
- k: number of samples
"""

def single_matrix(fs: int, melodies: np.array, frequencies: dict, K: int) -> np.array:
    """
    Generates observation matrix for single tone detection.
    alpha = mismatch, f_nj = fundamental freq.
    """

    nr_tone_samples = int((43392/12)/K)

    H = np.zeros((nr_tone_samples, 2))

    big_H = {0.975: {}, 1.025: {}}

    alpha = [0.975, 1.025]

    for mm in alpha:
        for j in range(len(melodies)):
            for m in melodies[j]:
                f_nj = mm * frequencies[m]
                for k in range(nr_tone_samples):
                    H[k, 0] = np.cos(2 * np.pi * f_nj/fs * k)
                    H[k, 1] = np.sin(2 * np.pi* f_nj/fs * k)
                big_H[mm][m] = H.copy()

    return big_H


def three_matrix(fs: int, melodies: np.array, frequencies: dict, K: int) -> np.array:
    """
    
    """
    nr_tone_samples = int((43392/12)/K)
    big_H = {0.975: {}, 1.025: {}}

    H = np.zeros((nr_tone_samples, 6))

    alpha = [0.975, 1.025]

    for mm in alpha:
        for j in range(len(melodies)):
            for m in melodies[j]:
                f_nj = mm * frequencies[m]
                for k in range(nr_tone_samples):
                    # Tone 1 
                    H[k, 0] = np.cos(2 * np.pi * f_nj/fs * k)
                    H[k, 1] = np.sin(2 * np.pi * f_nj/fs * k)
                    # Tone 2 (second-order harmonics)
                    H[k, 2] = np.cos(2 * np.pi * 3*f_nj/fs * k)
                    H[k, 3] = np.sin(2 * np.pi * 3*f_nj/fs * k)
                    # Tone 3 (fourth-order harmonics)
                    H[k, 4] = np.cos(2 * np.pi * 5*f_nj/fs * k)
                    H[k, 5] = np.sin(2 * np.pi * 5*f_nj/fs * k)
                big_H[mm][m] = H.copy()

    return big_H


def classifier(melodies: list, melody: list, H: np.array,
               K: int) -> int:
    """
    argmax_{j} sum_{0}->{nr_tones}(||H_{n,j} * y_{n}||^2)
    """

    nr_tone_samples = int((43392/12)/K)

    mismatches = [0.975, 1.025]

    j_hat = None
    alpha_hat = None
    max_sum = -10000

    y = np.split(melody, 12)

    i = 0
    curr_sum = 0
    for alpha in mismatches:
        for j in range(len(melodies)):
            for tone in melodies[j]:
                y_col = y[i].reshape(-1, 1)             
                y_col = y_col[:nr_tone_samples]
                norm = (np.linalg.norm(np.matmul((H[alpha][tone]).T, y_col))) ** 2
                curr_sum += norm
                i += 1  
            if curr_sum > max_sum:
                max_sum = curr_sum
                j_hat = j
                alpha_hat = alpha

            curr_sum = 0
            i = 0    

    return j_hat, alpha_hat

