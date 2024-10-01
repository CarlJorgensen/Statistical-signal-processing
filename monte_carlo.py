#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Tone_classifier as tc
import sys
from tqdm import tqdm
from SignalGenerator import SignalGenerator
from matplotlib import pyplot as plt
import numpy as np


def monte_carlo(nr_sim: int, SNR_range: range, sample_div: int,
                three_matrix: bool, three_tone: bool):
    setup = SignalGenerator()

    frequencies = setup.dict_note2frequency
    melodies = setup.melodies
    K = sample_div

    if three_matrix != True:
        H = tc.single_matrix(8820, melodies, frequencies, K)
    else:
        H = tc.three_matrix(8820, melodies, frequencies, K)
    
    prob_error = []
    tot_error = 0

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

            j_hat, alpha = tc.classifier(melodies, melody, H, K)
                
            if j_hat != idx or alpha != mismatch:
                error += 1

        prob_error.append(error/nr_sim)
        tot_error += error

    return prob_error, tot_error


def main():

    SNR_range = range(-40, 0)
    nr_sim = 1000
    sample_div = 10

    # Single tone classifier on single tone melody
    error_data1, tot_error = monte_carlo(nr_sim, SNR_range, sample_div, False, False)
    print(f"Total error: {tot_error}, Total simulations: {nr_sim*len(SNR_range)}")

    # Single tone classifier on three tone melody
    error_data2, tot_error = monte_carlo(nr_sim, SNR_range, sample_div, False, True)
    print(f"Total error: {tot_error}, Total simulations: {nr_sim*len(SNR_range)}")

    # Three tone classifier on single tone melody
    error_data3, tot_error = monte_carlo(nr_sim, SNR_range, sample_div, True, False)
    print(f"Total error: {tot_error}, Total simulations: {nr_sim*len(SNR_range)}")

    # Three tone classifier on three tone melody
    error_data4, tot_error = monte_carlo(nr_sim, SNR_range, sample_div, True, True)
    print(f"Total error: {tot_error}, Total simulations: {nr_sim*len(SNR_range)}")


    plt.figure(figsize=(10, 6))
    plt.plot(SNR_range, error_data1, 'r-o', label="Single tone classifier on single tone melody")
    plt.plot(SNR_range, error_data2, 'g-s', label="Single tone classifier on three tone melody")
    plt.plot(SNR_range, error_data3, 'b-^', label="Three tone classifier on single tone melody")
    plt.plot(SNR_range, error_data4, 'm-d', label="Three tone classifier on three tone melody")
    plt.xlabel("SNR [dB]")
    plt.ylabel("P(error)")
    plt.title("Monte Carlo Simulation")
    plt.legend(loc="upper right")
    plt.savefig("/home/carljoergensen/Courses/Master/TSKS15/music_detector/monte_carlo.png")


if __name__ == "__main__":
    main()
