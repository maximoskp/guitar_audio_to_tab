import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import data_utils

with open('data/x_audio.pickle', 'rb') as handle:
    x_audio = pickle.load(handle)

with open('data/x_hand.pickle', 'rb') as handle:
    x_hand = pickle.load(handle)

with open('data/y_tab.pickle', 'rb') as handle:
    y_tab = pickle.load(handle)

# %%

rnd_idxs = np.random.permutation( x_audio.shape[0] )
# make sure noise and empty are first
rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-1)
rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-3)

tr = 2*x_audio.shape[0]//3
v = x_audio.shape[0]//6
te = x_audio.shape[0]//6

x_audio_train = np.expand_dims( x_audio[ rnd_idxs[:tr] ,:], axis=2 )
x_hand_train = np.expand_dims( x_hand[ rnd_idxs[:tr] ,:], axis=2 )
y_tab_train = np.expand_dims( y_tab[ rnd_idxs[:tr] ,:,:], axis=3 )

x_audio_valid = np.expand_dims( x_audio[ rnd_idxs[tr:tr+v] ,:], axis=2 )
x_hand_valid = np.expand_dims( x_hand[ rnd_idxs[tr:tr+v] ,:], axis=2 )
y_tab_valid = np.expand_dims( y_tab[ rnd_idxs[tr:tr+v] ,:,:], axis=3 )

x_audio_test = np.expand_dims( x_audio[ rnd_idxs[tr+v:tr+v+te] ,:], axis=2 )
x_hand_test = np.expand_dims( x_hand[ rnd_idxs[tr+v:tr+v+te] ,:], axis=2 )
y_tab_test = np.expand_dims( y_tab[ rnd_idxs[tr+v:tr+v+te] ,:,:], axis=3 )


# load model
model = keras.models.load_model( 'models/hand/tab_hand_full_CNN_out_current_best.hdf5' )
# model = keras.models.load_model( 'models/tab_full_CNN_out_epoch478_valLoss0.000795.hdf5' )

import matplotlib.pyplot as plt

if not os.path.exists('figs'):
    os.makedirs('figs')

for session in range(10):
    tmp_rnd_idx = np.random.randint( x_audio_test.shape[0] )

    # y_pred = model.predict( x_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:] )
    # y_true = y_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:,:]
    print(x_audio_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:].shape)
    print(x_hand_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:].shape)
    y_pred = model.predict( [x_audio_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:], [x_hand_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:]]] )
    y_true = y_tab_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:,:]

    fig, ax = plt.subplots(2,1)
    # fig.subplot(3,1,1)
    # plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
    ax[0].imshow( y_pred[0,:,:] , cmap='gray_r' )
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('string')
    ax[0].title.set_text('probabilities')
    # _,ax = plt.subplot(3,1,2)
    # plt.bar( np.arange(y_pred.shape[1]) , y_pred[0] )
    ax[1].imshow( y_true[0,:,:,:], cmap='gray_r' )
    ax[1].set_xticklabels([])
    ax[1].set_ylabel('string')
    ax[1].title.set_text('true')
    fig.savefig('figs/test' + str(session) + '.png', dpi=300)