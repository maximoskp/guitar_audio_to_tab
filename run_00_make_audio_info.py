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

audio_folder = 'data/guitar_samples/'
onsets_folder = 'data/onsets/'

guitars = ['firebrand', 'martin']

guitar_samples = {}
for guitar in guitars:
    guitar_samples[guitar] = data_utils.GuitarSamples(guitar, constants)
    for gidx in range(1, 11, 1):
        for sidx in range(1, 7, 1):
            for fidx in range(0, 13, 1):
                print(guitar + str(gidx) + ': \t' + str(sidx) + '\t' + str(fidx))
                audio_path = audio_folder + guitar + str(gidx) + '/' + 'string' + str(sidx) + '/' + str(fidx) + '.wav'
                onset_path = onsets_folder + guitar + str(gidx) + '/' + 'string' + str(sidx) + '/' + str(fidx) + '.txt'
                guitar_samples[guitar].append_sample( sidx, fidx, audio_path, onset_path )


# %% augment octaves
for guitar in guitars:
    guitar_samples[guitar].augment_octaves()


# %%

with open('data/' + os.sep + 'guitar_samples.pickle', 'wb') as handle:
    pickle.dump(guitar_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)