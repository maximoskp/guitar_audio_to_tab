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

with open('data/dataset_audio_patterns_short.pickle', 'rb') as handle:
    patterns = pickle.load(handle)

# %% XY matrices

x = np.zeros( ( len( patterns ) , len( patterns[0]['audio'] ) ) )
y = np.zeros( ( len( patterns ) , patterns[0]['tab'].shape[0] , patterns[0]['tab'].shape[1] ) )

for i, p in enumerate( patterns ):
    x[i,:] = p['audio']
    y[i,:,:] = p['tab']