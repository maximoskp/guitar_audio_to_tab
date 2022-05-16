#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:30:18 2022

@author: max
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

with open('data/patterns.pickle', 'rb') as handle:
    patterns = pickle.load(handle)

# sort by frequency
counts = np.zeros(len(patterns))
for i,k in enumerate(patterns.keys()):
    counts[i] = patterns[k]['count']

sorted_idxs = np.argsort( counts )[::-1]

k = list(patterns.keys())
sorted_patterns = [ patterns[k[i]] for i in sorted_idxs ]

# sonify patterns
# load audio samples
with open('data/guitar_samples.pickle', 'rb') as handle:
    guitar_samples = pickle.load(handle)
g = guitar_samples['firebrand']
sr = g.constants.sample_rate

dataset = []

# keep top_k
top_k = 500
for i in range(top_k):
    print('pattern: ' + str(i) + '/' + str(top_k))
    p = sorted_patterns[i]['pattern']
    # non-rollable have free chord and a span greater than 5 frets
    rollable = True
    tmp_sum = np.sum( p , axis=1 )
    if np.sum(tmp_sum) != 0:
        nnz_tmp_sum = np.where( tmp_sum != 0 )[0]
        if nnz_tmp_sum[-1] == 0 and nnz_tmp_sum[-1]-nnz_tmp_sum[0] > 5:
            rollable = False
        if rollable:
            highest_fret = np.where( np.sum( p, axis=0 ) > 0 )[0][-1]
            while highest_fret < 24:
                s = np.zeros(sr)
                for string in range(6):
                    if np.sum(p[string,:]) > 0:
                        fret = np.where( p[string,:] != 0 )[0][0]
                        s += guitar_samples['firebrand'].get_random_sample( 6-string, fret, duration_samples=sr )
                sample = {'audio': s, 'tab': p}
                dataset.append( sample )
                p = np.roll( p , [0,1] )
                highest_fret = np.where( np.sum( p, axis=0 ) > 0 )[0][-1]
            # end while
        else:
            s = np.zeros(sr)
            for string in range(6):
                if np.sum(p[string,:]) > 0:
                    fret = np.where( p[string,:] != 0 )[0][0]
                    s += guitar_samples['firebrand'].get_random_sample( string+1, fret, duration_samples=sr )
            sample = {'audio': s, 'tab': p}
            dataset.append( sample )
    else:
        print('empty tab')
    # add empty tab
    sample = {'audio':  np.zeros(sr), 'tab': np.zeros( (6,25) )}
    dataset.append( sample )
# end for

# %% 

with open('data/dataset_audio1sec.pickle', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)