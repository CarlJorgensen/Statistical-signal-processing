import numpy as np
import matplotlib.pyplot as plt
from SignalGenerator import SignalGenerator

#constants

"""
indexes:
- m: 10, the number of melodies
- l: 2, the number of pitch mismatches
- j: 20, the number of different hypotheses (10 melodies * 2 pitch mismatches)
- n: 12, the notes
- K: antal samples för en ton
- Y: mätvärden för en ton
"""

class SignalClassifier:
   """ Class for the classifcation single-tone and three-tone melodies."""
   def __init__(self, fs = 8820, three_tone= False) -> None:
       self.sg = SignalGenerator()
       self.freq = self.sg.dict_note2frequency
       self.melodies = self.__get_melodies()
       self.fs = fs
       self.three_tone = three_tone

   def single_tone_matrix(self, fnj, k_, missmatch, K_div) -> np.ndarray:
       """Construct the matrix for single-tone"""
       H_nj = np.zeros((k_, 2))
       nr_samples = int(k_/K_div)
       for k in range(0, nr_samples):
           H_nj[k, 0] = np.cos(2 * missmatch* np.pi *  fnj * k / self.fs )  
           H_nj[k, 1] = np.sin(2 * missmatch* np.pi * fnj * k /self.fs)
       return H_nj
   def three_tone_matrix(self, fnj, k_, missmatch, K_div) -> np.ndarray:
       """Construct the matrix for single-tone"""
       H_nj = np.zeros((k_, 6))
       nr_samples = int(k_/K_div)
       for k in range(0, nr_samples):
           H_nj[k, 0] = np.cos(2 * missmatch* np.pi *  fnj * k / self.fs )  #column 1
           H_nj[k, 1] = np.sin(2 * missmatch* np.pi * fnj * k /self.fs)    #column 2
           H_nj[k, 2] = np.cos(2 * missmatch* np.pi *  3 * fnj * k / self.fs ) #column 3
           H_nj[k, 3] = np.sin(2 * missmatch* np.pi * 3 * fnj * k /self.fs)    #column 4
           H_nj[k, 4] = np.cos(2 * missmatch* np.pi *  5 * fnj * k / self.fs ) #column 5
           H_nj[k, 5] = np.sin(2 * missmatch* np.pi * 5 * fnj * k /self.fs)    #column 6
       return H_nj
   
   def __get_melodies(self):
       self.melodies = self.sg.melodies
       return self.melodies
   
   def classifier(self, melody, K=1):
       """Formula ||H_nj^T * y||"""

       nr_samples = len(melody)
       nr_tones = 12  # all melodies have 12 tones
       tone_length = int(nr_samples / nr_tones)  # 3616 values per tone

       best_melody_idx = None
       best_mismatch = None
       max_sum = -10000  
       for missmatch in self.sg.pitch_mismatches:
           for melody_idx, melody_notes in enumerate(self.melodies):
               total_sum = 0
               for tone_index in range(nr_tones): #toneindex between 1-12        
                   y = melody[tone_index * tone_length: (tone_index + 1) * tone_length]
                   note = melody_notes[tone_index]
                   if self.three_tone:
                       H_nj = self.three_tone_matrix(fnj = self.freq[note], k_ = tone_length, missmatch = missmatch, K_div = K)
                   else:
                       H_nj = self.single_tone_matrix(fnj = self.freq[note], k_ = tone_length, missmatch = missmatch, K_div = K)
                   #H_nj = self.three_tone_matrix(fnj = self.freq[note], k_ = tone_length, missmatch = missmatch, K_div = K)

                   mult = np.matmul(H_nj.T, y)
                   norm_value = np.linalg.norm(mult) ** 2
                   total_sum += norm_value
               #print(total_sum)
               if total_sum > max_sum:
                   max_sum = total_sum
                   best_melody_idx = melody_idx  
                   best_mismatch = missmatch  
       return best_melody_idx, best_mismatch

def monte_carlo_simulation(rounds, K_div, nr_tones = 1):
   sg = SignalGenerator()
   sc = SignalClassifier()
   correct_count = 0
   incorrect_count = 0
   for _ in range(rounds):
       melody, idx, mismatch = sg.generate_random_melody(snr_db=50, nr_tones=nr_tones)
       best_melody_idx, best_mismatch = sc.classifier(melody, K_div)
       print(f"Best melody index: {best_melody_idx}, Best mismatch: {best_mismatch}, Actual melody index: {idx}, Actual mismatch: {mismatch}")
       if best_melody_idx == idx and best_mismatch == mismatch:
           correct_count += 1
       else:
           incorrect_count += 1
   print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")