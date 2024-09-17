#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from SignalGenerator import SignalGenerator
import numpy as np
import math
import matplotlib.pyplot as plt

""" TODO:
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
- Generated melody is off pitch by a factor of 0.975 or 1.025

"""

"""
Iteration variables:
- m=10: number of melodies
- l=2: number of pitch mismatches
- j=20: number of hypotheses (=m*l), equally likely.
- k: number of samples
"""


def generate_matrix(length):
    # Frequency ??
    f = 440 # For now
    H_t = np.zeros(shape=(2, length))
    for i in range(length):
        H_t[0][i] = math.cos(2*math.pi * f * i)
        H_t[1][i] = math.sin(2*math.pi * f * i)

    return H_t

def classifier(melody, m):
    """
    argmax_j sum(||H_t * y_n||^2)
    Idk return value yet
    """
    return 0

def main():
    import sys
    try:
        assert sys.version_info >= (3,0)
    except AssertionError:
        print("This script requires python version 3.4.3 or higher")
        raise

    sg = SignalGenerator()
    # generate a random melody, with SNR 100 dB, and 3 tones
    melody, idx, mismatch = sg.generate_random_melody(100, 3)
    nr_samples = len(melody)

    H_t = generate_matrix(len(melody))

    classifier(melody, len(melody))

    nr_tones = 12 # all melodies have 12 tones
    tone = melody[:int(nr_samples/nr_tones)]
    nr_tone_samples = len(tone)
    spectrum = np.abs(np.fft.fft(tone))
    fs = sg.sampling_frequency
    freqs = np.arange(nr_tone_samples) * fs / nr_tone_samples
    plt.figure()
    plt.plot(freqs[:int(nr_tone_samples/2)], spectrum[:int(nr_tone_samples/2)])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('magnitude')
    plt.savefig('python-example1.png')




if __name__=="__main__":
    main()