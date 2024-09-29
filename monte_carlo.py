#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Tone_classifier as tc
import sys
from tqdm import tqdm
from SignalGenerator import SignalGenerator
from matplotlib import pyplot as plt
import numpy as np


def monte_carlo(nr_sim: int, SNR_range: range, sample_div: int, three_tone: bool):
    setup = SignalGenerator()

    frequencies = setup.dict_note2frequency
    melodies = setup.melodies

    K = sample_div

    prob_error = []

    tot_error = 0

    for SNR in tqdm(SNR_range, desc="SNR: ", leave=True):
        error = 0
        for _ in tqdm(range(nr_sim), desc="Iteration: ", leave=False):
            sg = SignalGenerator()

            melody = None
            idx = None
            mismatch = None
            # if three_tone != True:
            #     melody, idx, mismatch = sg.generate_random_melody(SNR, 1)
            # else:
            #     melody, idx, mismatch = sg.generate_random_melody(SNR, 3)

            melody, idx, mismatch = sg.generate_random_melody(SNR, 1)
            
            nr_samples = len(melody)
            nr_tones = 12
            tone = melody[:int(nr_samples/nr_tones)]
            nr_tone_samples = int(len(tone)/K)

            if three_tone != True:
                j_hat, alpha = tc.classifier(melodies, melody, K, frequencies, False)
            else:
                j_hat, alpha = tc.classifier(melodies, melody, K, frequencies, True)
                
            if j_hat != idx:
                error += 1

        prob_error.append(error/nr_sim)
        tot_error += error

    return prob_error, tot_error

def plot(SNR_range, error_data, title):
    plt.plot(SNR_range, error_data)
    plt.xlabel("SNR [dB]")
    plt.ylabel("P(error)")
    plt.title(title)
    plt.savefig("/home/carljoergensen/Courses/Master/TSKS15/music_detector/three_classifier_single_tone_1k.png")
    plt.clf()


def main():
    nr_sim = 1000
    SNR_range = range(-40,0)
    sample_div = 10

    error_data_single_tone, nr_errors = monte_carlo(nr_sim,
                                                    SNR_range,
                                                    sample_div,
                                                    True)
    print(f"Total errors: {nr_errors}, out of {nr_sim*len(SNR_range)}")
    print(f"Total successes {nr_sim*len(SNR_range) - nr_errors}")

    plot(SNR_range, error_data_single_tone,
        "Three-tone classifier on a single-tone melody")
    
    # print(f"Total successes {nr_sim*len(SNR_range) - nr_errors}")
    # error_data_single_tone, nr_errors = monte_carlo(nr_sim, 
    #                                                 SNR_range, 
    #                                                 sample_div,
    #                                                 False) 
    # print(f"Total errors: {nr_errors}, out of {nr_sim*len(SNR_range)}")
    # print(f"Total successes {nr_sim*len(SNR_range) - nr_errors}")

    # plot(SNR_range, error_data_single_tone,
    #      "Single-tone classifier on a single-tone melody")

    # error_data_three_tone, nr_errors = monte_carlo(nr_sim,
    #                                                SNR_range, 
    #                                                sample_div,
    #                                                True) 
    
    # print(f"Total errors: {nr_errors}, out of {nr_sim*len(SNR_range)}")
    # print(f"Total successes {nr_sim*len(SNR_range) - nr_errors}")

    # plot(SNR_range, error_data_three_tone,
    #     "Single-tone classifier on a three-tone melody")



if __name__ == "__main__":
    main()
