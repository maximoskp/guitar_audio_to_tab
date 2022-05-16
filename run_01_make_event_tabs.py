#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:31:55 2022

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

if not os.path.exists('data/event_tabs'):
    os.makedirs('data/event_tabs')

patterns = {}

for te_file in track_event_files:
    with open('data/dadaGP_event_parts' + os.sep + te_file, 'rb') as handle:
        pieces = pickle.load(handle)
        for p in pieces:
            print(p.name)
            for t in p.track_events:
                for te in t:
                    te['tab'] = data_utils.event2fulltab( te )
                    p = data_utils.patternOf2DTab( te['tab'] )
                    if str(p) not in patterns.keys():
                        patterns[ str(p) ] = {
                            'pattern': p,
                            'count': 1
                        }
                    else:
                        patterns[ str(p) ]['count'] += 1
    with open('data/event_tabs' + os.sep + te_file, 'wb') as handle:
        pickle.dump(pieces, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/patterns.pickle', 'wb') as handle:
    pickle.dump(patterns, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
# with open('data/patterns.pickle', 'rb') as handle:
#     patterns = pickle.load(handle)
