import librosa
import numpy as np
import os
import data_utils
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

constants = data_utils.Constants()

audio_folder = 'data/guitar_samples'
onsets_folder = 'data/onsets'

guitars = ['firebrand', 'martin']

guitar_samples = {}
for guitar in guitars:
    guitar_samples[guitar] = {}
    for gidx in range(1, 11, 1):
        for sidx in range(1, 7, 1):
            for fidx in range(1, 13, 1):
                # load audio with librosa
                # get audio from index of starting onset
                # load onset txt
                # transform str to float and to sample number with sr
                pass