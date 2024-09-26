#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Tone_classifier as tc
import sys
from tqdm import tqdm
from SignalGenerator import SignalGenerator
from matplotlib import pyplot as plt
import numpy as np


def montecarlo_single_tone(nr_sim: int, SNR_range: range, sample_div: int, three_tone: bool):
    setup = SignalGenerator()

    frequencies = setup.dict_note2frequency
    melodies = setup.melodies

    K = sample_div

    prob_error = []

    for SNR in tqdm(SNR_range, desc="SNR: ", leave=True):
        error = 0
        for _ in tqdm(range(nr_sim), desc="Iteration: ", leave=False):
            sg = SignalGenerator()

            melody = None
            idx = None
            mismatch = None
            if three_tone != True:
                melody, idx, mismatch = sg.generate_random_melody(SNR, 1)
            else:
                melody, idx, mismatch = sg.generate_random_melody(SNR, 3)
            
            nr_samples = len(melody)
            nr_tones = 12
            tone = melody[:int(nr_samples/nr_tones)]
            nr_tone_samples = int(len(tone)/K)

            j_hat, alpha = tc.classifier_single(melodies, melody, K, frequencies)

            if j_hat != idx:
                error += 1

        prob_error.append(error/nr_sim)
    
    return prob_error


def main():
    nr_sim = 200
    SNR_range = range(-40,0)
    sample_div = 10
    error_data_single_tone = montecarlo_single_tone(nr_sim, 
                                                    SNR_range, 
                                                    sample_div,
                                                    False) 

    plt.figure()
    plt.plot(SNR_range, error_data_single_tone)
    plt.xlabel("SNR [dB]")
    plt.ylabel("P(error)")
    plt.title("Single-tone classifier on a single-tone melody")
    plt.savefig("/home/carljoergensen/Courses/Master/TSKS15/music_detector/single_classifier_single_tone.png")
    plt.clf()

    error_data_three_tone = montecarlo_single_tone(nr_sim,
                                                   SNR_range, 
                                                   sample_div,
                                                   True) 

    plt.figure()
    plt.plot(SNR_range, error_data_three_tone)
    plt.xlabel("SNR [dB]")
    plt.ylabel("P(error)")
    plt.title("Single-tone classifier on a three-tone melody")
    plt.savefig("/home/carljoergensen/Courses/Master/TSKS15/music_detector/single_classifier_three_tone.png")
    plt.clf()



if __name__ == "__main__":
    main()
