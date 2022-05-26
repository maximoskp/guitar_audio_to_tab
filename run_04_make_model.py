# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:25:09 2022

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

with open('data/x.pickle', 'rb') as handle:
    x = pickle.load(handle)

with open('data/y.pickle', 'rb') as handle:
    y = pickle.load(handle)

# %%

rnd_idxs = np.random.permutation( x.shape[0] )
# make sure noise and empty are first
rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-1)
rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-3)

tr = 2*x.shape[0]//3
v = x.shape[0]//6
te = x.shape[0]//6

x_train = np.expand_dims( x[ rnd_idxs[:tr] ,:], axis=2 )
y_train = np.expand_dims( y[ rnd_idxs[:tr] ,:,:], axis=3 )

x_valid = np.expand_dims( x[ rnd_idxs[tr:tr+v] ,:], axis=2 )
y_valid = np.expand_dims( y[ rnd_idxs[tr:tr+v] ,:,:], axis=3 )

x_test = np.expand_dims( x[ rnd_idxs[tr+v:tr+v+te] ,:], axis=2 )
y_test = np.expand_dims( y[ rnd_idxs[tr+v:tr+v+te] ,:,:], axis=3 )

# %% 

import tensorflow as tf
from tensorflow import keras

# %% 

max_norm_value = 2.0
input_shape = [1600,1]

latent_size = 6*64

# create the model
encoder = keras.models.Sequential()
encoder.add(keras.layers.Conv1D(64, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
# encoder.add(keras.layers.MaxPooling1D(2))
encoder.add(keras.layers.Conv1D(32, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# encoder.add(keras.layers.MaxPooling1D(2))
encoder.add(keras.layers.Conv1D(16, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# encoder.add(keras.layers.MaxPooling1D(2))
encoder.add(keras.layers.Conv1D(8, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
encoder.add(keras.layers.Flatten())

latent = keras.models.Sequential([
    keras.layers.Dense(latent_size),
    keras.layers.Reshape( (1,6,latent_size//6) )
])

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(latent_size//6, kernel_size=3, strides=2, padding='valid',
                                 activation='selu', input_shape=[1,6,latent_size//6]),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same',
                                 activation='selu'),
    keras.layers.Lambda(lambda x: x[:,:,:-1,:]),
    keras.layers.Reshape([6, 25])
])
'''
out_layer = keras.models.Sequential([
    keras.layers.Lambda(lambda x: x[:,:,:-1,:]),
    keras.layers.Reshape([6*25]),
    keras.layers.Dense(y_train.shape[1], activation='sigmoid')
])
'''
model = keras.models.Sequential([encoder, latent, decoder])
# model = keras.models.Sequential([encoder, latent])

encoder.summary()
latent.summary()
decoder.summary()
model.summary()

# %%

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['cosine_similarity'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rounded_accuracy])

# %%

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

os.makedirs( 'models', exist_ok=True )

filepath = 'models/tab_full_CNN_out_epoch{epoch:02d}_valLoss{val_loss:.6f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

filepath_current_best = 'models/tab_full_CNN_out_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

if os.path.exists('/models/full_tab_logger.csv'):
    os.remove('/models/full_tab_logger.csv')
csv_logger = CSVLogger('models/full_tab_logger.csv', append=True, separator=';')

# %% 

history = model.fit( x_train, y_train, epochs=1000, batch_size=64, validation_data=(x_valid, y_valid) , callbacks=[checkpoint, checkpoint_current_best, csv_logger] )

# %% 
'''
import matplotlib.pyplot as plt

# %% 

tmp_rnd_idx = np.random.randint( x_test.shape[0] )

# y_pred = model.predict( x_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:] )
# y_true = y_test[tmp_rnd_idx:tmp_rnd_idx+1,:,:,:]
y_pred = model.predict( x_train[tmp_rnd_idx:tmp_rnd_idx+1,:,:] )
y_true = y_train[tmp_rnd_idx:tmp_rnd_idx+1,:,:,:]

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
'''
