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

def note2frequency() -> dict:
    """
    Generates a dictionary mapping notes 
    to their corresponding frequencies [Hz].
    returns: dictionary where key=note, value=frequency
    """
    octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes = [octave[note] + str(oct) for oct in range(1, 8) for note in range(12)]
    n = np.arange(4, 88)
    frequencies = dict(zip(notes, 440.0 * 2.0**((n-49)/12)))
    return frequencies


def parseNotes(filename) -> np.array:
    """
    Parse the notes in the melodies.txt file and convert to their frequencies.
    returns: np.array with frequencies
    """    
    with open(filename, 'r') as file:
        melodies = file.readlines()
        melodies = [melody.strip().split(', ') for melody in melodies]

    melodies_f = np.zeros(shape=(len(melodies), len(melodies[0])))
    
    # Get the tones in melodies and convert to frequencies and place in melodies_f that should be the
    # same as melodies but with frequencies instead of notes
    # Get the tones in melodies and convert to frequencies and place in melodies_f
    frequencies = note2frequency()
    for i, m in enumerate(melodies):
        for j, n in enumerate(m):
            melodies_f[i][j] = frequencies[n]

    for i in range(len(melodies_f)):
        print(melodies_f[i])

    return melodies_f


def generate_matrix(melody):
    """
    Generate the transposed matrix H_t
    """
    # Frequency ??
    f = 440.0 * 2.0**((4-49)/12)
    H_t = np.zeros(shape=(2, melody))
    for i in range(melody):
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

    placeholder = parseNotes('melodies.txt')
    


if __name__=="__main__":
    main()