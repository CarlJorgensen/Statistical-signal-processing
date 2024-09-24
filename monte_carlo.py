#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from SignalGenerator import SignalGenerator
import single_tone_classifier as stc
import sys

"""
TODO
1. Run code many times
2. Plot
"""

"""
NOTE
Not working when scaling the samples down by 10
"""


def main():
    try:
        assert sys.version_info >= (3, 0)
    except AssertionError:
        print("This script requires python version 3.4.3 or higher")
        raise

    sg = SignalGenerator()

    melody, idx, mismatch = sg.generate_random_melody(100, 1)

    y = stc.divide_melody(melody)

    melodies = sg.melodies
    frequencies = sg.dict_note2frequency
    fs = sg.sampling_frequency

    melody_mm1, melody_mm2 = stc.melody2frequency(melodies, frequencies)

    H_t = stc.generate_matrix(melody, melody_mm1, melody_mm2, fs)

    j_hat, alpha_hat = stc.classifier(y, H_t)

    print(f"Guess: {j_hat}, Actual: {idx}")
    print(f"Guess: {alpha_hat}, Actual: {mismatch}")


if __name__ == "__main__":
    main()
