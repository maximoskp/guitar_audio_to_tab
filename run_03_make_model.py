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
tr = 100000
v = 20000
te = 50000

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
encoder.add(keras.layers.Conv1D(64, kernel_size=16, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
encoder.add(keras.layers.MaxPooling1D(2))
encoder.add(keras.layers.Conv1D(32, kernel_size=8, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
encoder.add(keras.layers.MaxPooling1D(2))
encoder.add(keras.layers.Conv1D(16, kernel_size=4, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
encoder.add(keras.layers.MaxPooling1D(2))
encoder.add(keras.layers.Conv1D(1, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
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

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['cosine_similarity']) 

# %% 

history = model.fit( x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_valid, y_valid)  )