#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 21:27:14 2022

@author: max
"""

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import data_utils
import os

# RUN to save first time
track_event_files = os.listdir( 'data/dadaGP_event_parts' )

# load audio samples
with open('data/guitar_samples.pickle', 'rb') as handle:
    guitar_samples = pickle.load(handle)

total_duration = 0

for te_file in track_event_files:
    with open('data/dadaGP_event_parts' + os.sep + te_file, 'rb') as handle:
        pieces = pickle.load(handle)
        for p in pieces:
            for t in p.track_events:
                for te in t:
                    for e in te:
                        total_duration += 8000
                        print('total duration: ', total_duration)