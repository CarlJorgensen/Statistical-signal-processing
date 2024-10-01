from SignalGenerator import SignalGenerator
import numpy as np
import matplotlib.pyplot as plt
sg = SignalGenerator()

def create_hmatrix_single(div_k):
    nr_of_samples = int((43392/12)/div_k)
    all_hmatrix = {0.975 : {}, 1.025 : {}}
    freq = sg.dict_note2frequency
    alpha = [0.975, 1.025]
    hmatrix = np.zeros((nr_of_samples,2))
    for g in alpha:
        for j in range(len(sg.melodies)):
            for m in sg.melodies[j]:
                fnj = freq[m] * g
                t = 2*np.pi*fnj/8820
                for i in range(nr_of_samples):
                    hmatrix[i,0] = np.cos(t*i)
                    hmatrix[i,1] = np.sin(t*i)
                all_hmatrix[g][m] = hmatrix.copy()
    return all_hmatrix

def create_hmatrix_multi(div_k):
    nr_of_samples = int((43392/12)/div_k)
    all_hmatrix = {0.975 : {}, 1.025 : {}}
    freq = sg.dict_note2frequency
    alpha = [0.975, 1.025]
    hmatrix = np.zeros((nr_of_samples,6))
    for g in alpha:
        for j in range(len(sg.melodies)):
            for m in sg.melodies[j]:
                fnj = freq[m] * g
                t = 2*np.pi*fnj/8820
                for i in range(nr_of_samples):
                    hmatrix[i,0] = np.cos(t*i)
                    hmatrix[i,1] = np.sin(t*i)
                    hmatrix[i,2] = np.cos(3*t*i)
                    hmatrix[i,3] = np.sin(3*t*i)
                    hmatrix[i,4] = np.cos(5*t*i)
                    hmatrix[i,5] = np.sin(5*t*i)
                all_hmatrix[g][m] = hmatrix.copy()
    return all_hmatrix

def detector(hmatrix, snr, div_k, nr_of_tones):
    nr_of_samples = int((43392/12)/div_k)
    melody, idx, mismatch = sg.generate_random_melody(snr, nr_of_tones)
    subarrays = np.split(melody, 12)
    alpha = [0.975, 1.025]
    prob_mel = 0
    best_mel = 0
    subarray = 0
    for q in alpha:
        for mel in range(len(sg.melodies)):
            for tone in sg.melodies[mel]:
                subarray_column = subarrays[subarray].reshape(-1, 1)
                subarray_column = subarray_column[:nr_of_samples]
                norm = (np.linalg.norm(np.matmul((hmatrix[q][tone]).T, subarray_column))) ** 2
                prob_mel = prob_mel + norm
                subarray = subarray + 1
            if prob_mel > best_mel:
                best_mel = prob_mel
                idx_guess = mel
                mismatch_guess = q
            prob_mel = 0
            subarray = 0
    if idx_guess == idx and mismatch_guess == mismatch:
        return 0
    else:
        return 1

def main():
    div_k = 10 #divide samples per note by this value

    montecarlo = 1000
    nr_of_snr = 27
    snr_step = 1
    start_snr = -8

    hmatrix1 = create_hmatrix_single(div_k)
    hmatrix2 = create_hmatrix_multi(div_k)
    print("Hmatrix redy")

    #classifier 1, tones 1
    error_list1 = []
    snr_list1 = []
    sum_error = 0
    for i in range(nr_of_snr):
        snr = start_snr-snr_step*i
        for _ in range(montecarlo):
            error = detector(hmatrix1, snr, div_k, 1)
            sum_error = sum_error + error
        error_list1.append(sum_error/montecarlo)
        snr_list1.append(snr)
        sum_error = 0
        print(i)

    #classifier 3, tones 1
    error_list2 = []
    snr_list2 = []
    sum_error = 0
    for i in range(nr_of_snr):
        snr = start_snr-snr_step*i
        for _ in range(montecarlo):
            error = detector(hmatrix2, snr, div_k, 1)
            sum_error = sum_error + error
        error_list2.append(sum_error/montecarlo)
        snr_list2.append(snr)
        sum_error = 0
        print(i)

    #classifier 1, tones 3
    error_list3 = []
    snr_list3 = []
    sum_error = 0
    for i in range(nr_of_snr):
        snr = start_snr-snr_step*i
        for _ in range(montecarlo):
            error = detector(hmatrix1, snr, div_k, 3)
            sum_error = sum_error + error
        error_list3.append(sum_error/montecarlo)
        snr_list3.append(snr)
        sum_error = 0
        print(i)

    #classifier 3, tones 3
    error_list4 = []
    snr_list4 = []
    sum_error = 0
    for i in range(nr_of_snr):
        snr = start_snr-snr_step*i
        for _ in range(montecarlo):
            error = detector(hmatrix2, snr, div_k, 3)
            sum_error = sum_error + error
        error_list4.append(sum_error/montecarlo)
        snr_list4.append(snr)
        sum_error = 0
        print(i)


    plt.figure(figsize=(15, 10))

    plt.plot(snr_list1, error_list1, label='1 tone classifier & 1 tone')
    plt.plot(snr_list2, error_list2, label='3 tone classifier & 1 tone')
    plt.plot(snr_list3, error_list3, label='1 tone classifier & 3 tones')
    plt.plot(snr_list4, error_list4, label='3 tone classifier & 3 tones')

    plt.xlabel('SNR')
    plt.ylabel('P(error)')
    plt.title('Compare classifiers')

    plt.legend()

    plt.savefig('combined_plot.png', dpi=300) 
    plt.show()

main()