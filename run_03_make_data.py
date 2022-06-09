# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:51:21 2022

@author: user
"""

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import data_utils
import os
import numpy as np
import data_utils

with open('data/dataset_audio_patterns_hand.pickle', 'rb') as handle:
    patterns = pickle.load(handle)

# %% XY matrices

x_audio = np.zeros( ( len( patterns ) , len( patterns[0]['audio'] ) ) ).astype(np.float32)
x_hand = np.zeros( ( len( patterns ) , len(patterns[0]['hand']) ) ).astype(np.bool)
y_tab = np.zeros( ( len( patterns ) , patterns[0]['tab'].shape[0] , patterns[0]['tab'].shape[1] ) ).astype(np.bool)

for i, p in enumerate( patterns ):
    if i%1000 == 0:
        print(str(i) + ' / ' + str(len(patterns)))
    x_audio[i,:] = p['audio'].astype(np.float32)
    x_hand[i,:] = p['hand'].astype(np.bool)
    y_tab[i,:,:] = p['tab'].astype(np.bool)

# %% 

with open('data/x_audio.pickle', 'wb') as handle:
    pickle.dump(x_audio, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/x_hand.pickle', 'wb') as handle:
    pickle.dump(x_hand, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/y_tab.pickle', 'wb') as handle:
    pickle.dump(y_tab, handle, protocol=pickle.HIGHEST_PROTOCOL)