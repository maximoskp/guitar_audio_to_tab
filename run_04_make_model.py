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

with open('data/x_audio.pickle', 'rb') as handle:
    x_audio = pickle.load(handle)

with open('data/x_hand.pickle', 'rb') as handle:
    x_hand = pickle.load(handle)

with open('data/y_tab.pickle', 'rb') as handle:
    y_tab = pickle.load(handle)

idxs = np.random.permutation( x_audio.shape[0] )
x_audio = x_audio[idxs,:]
x_hand = x_hand[idxs,:]
y_tab = y_tab[idxs,:,:]

# %%

rnd_idxs = np.random.permutation( x_audio.shape[0] )
# make sure noise and empty are first
rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-1)
rnd_idxs = np.insert(rnd_idxs, 0 , rnd_idxs.size-3)

tr = 2*x_audio.shape[0]//3
v = x_audio.shape[0]//6
te = x_audio.shape[0]//6

x_audio_train = np.expand_dims( x_audio[ rnd_idxs[:tr] ,:], axis=2 )
# x_hand_train = np.expand_dims( x_hand[ rnd_idxs[:tr] ,:], axis=2 )
x_hand_train = x_hand[ rnd_idxs[:tr] ,:]
y_tab_train = np.expand_dims( y_tab[ rnd_idxs[:tr] ,:,:], axis=3 )

x_audio_valid = np.expand_dims( x_audio[ rnd_idxs[tr:tr+v] ,:], axis=2 )
# x_hand_valid = np.expand_dims( x_hand[ rnd_idxs[tr:tr+v] ,:], axis=2 )
x_hand_valid = x_hand[ rnd_idxs[tr:tr+v] ,:]
y_tab_valid = np.expand_dims( y_tab[ rnd_idxs[tr:tr+v] ,:,:], axis=3 )

x_audio_test = np.expand_dims( x_audio[ rnd_idxs[tr+v:tr+v+te] ,:], axis=2 )
# x_hand_test = np.expand_dims( x_hand[ rnd_idxs[tr+v:tr+v+te] ,:], axis=2 )
x_hand_test = x_hand[ rnd_idxs[tr+v:tr+v+te] ,:]
y_tab_test = np.expand_dims( y_tab[ rnd_idxs[tr+v:tr+v+te] ,:,:], axis=3 )

# %% 

import tensorflow as tf
from tensorflow import keras

# %% 

# max_norm_value = 2.0
input_audio_shape = [x_audio_train[0].size,1]
# input_hand_shape = [x_hand_train[0].size,1]
input_hand_shape = x_hand_train[0].size

latent_size = 6*64

# # create the model
# encoder = keras.models.Sequential()
# encoder.add(keras.layers.Conv1D(128, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
# # encoder.add(keras.layers.MaxPooling1D(2))
# encoder.add(keras.layers.Conv1D(164, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# # encoder.add(keras.layers.MaxPooling1D(2))
# encoder.add(keras.layers.Conv1D(32, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# # encoder.add(keras.layers.MaxPooling1D(2))
# encoder.add(keras.layers.Conv1D(16, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# encoder.add(keras.layers.Conv1D(8, kernel_size=3, kernel_constraint=keras.constraints.max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# encoder.add(keras.layers.Flatten())

input_audio = keras.layers.Input(shape=input_audio_shape)
input_hand = keras.layers.Input(shape=input_hand_shape)

# create the model
# INPUTS
# audio encoder
# audio_encoder = keras.layers.Conv1D(64, kernel_size=16, activation='relu')(input_audio)
# audio_encoder = keras.layers.Dropout(0.3)(audio_encoder)
# audio_encoder = keras.layers.BatchNormalization()(audio_encoder)
# audio_encoder = keras.layers.Conv1D(32, kernel_size=8, activation='relu')(audio_encoder)
# audio_encoder = keras.layers.Dropout(0.3)(audio_encoder)
# audio_encoder = keras.layers.BatchNormalization()(audio_encoder)
# audio_encoder = keras.layers.Conv1D(16, kernel_size=4, activation='relu')(audio_encoder)
# audio_encoder = keras.layers.Dropout(0.3)(audio_encoder)
# audio_encoder = keras.layers.BatchNormalization()(audio_encoder)
# audio_encoder = keras.layers.Conv1D(8, kernel_size=3, activation='relu')(audio_encoder)
# audio_encoder = keras.layers.Conv1D(4, kernel_size=3, activation='relu')(audio_encoder)
# audio_encoder = keras.layers.Flatten()(audio_encoder)
# audio_encoder = keras.models.Model( inputs=input_audio, outputs=audio_encoder )

audio_encoder = keras.layers.Conv1D(128, kernel_size=8, strides=2, activation='relu')(input_audio)
audio_encoder = keras.layers.MaxPooling1D(2)(audio_encoder)
audio_encoder = keras.layers.Dropout(0.3)(audio_encoder)
audio_encoder = keras.layers.BatchNormalization()(audio_encoder)
audio_encoder = keras.layers.Conv1D(256, kernel_size=4, strides=2, activation='relu')(audio_encoder)
audio_encoder = keras.layers.Dropout(0.3)(audio_encoder)
audio_encoder = keras.layers.BatchNormalization()(audio_encoder)
audio_encoder = keras.layers.MaxPooling1D(2)(audio_encoder)
audio_encoder = keras.layers.Conv1D(512, kernel_size=3, strides=2, activation='relu')(audio_encoder)
audio_encoder = keras.layers.Dropout(0.3)(audio_encoder)
audio_encoder = keras.layers.BatchNormalization()(audio_encoder)
audio_encoder = keras.layers.MaxPooling1D(2)(audio_encoder)
audio_encoder = keras.layers.Conv1D(1024, kernel_size=3, activation='relu')(audio_encoder)
audio_encoder = keras.layers.Flatten()(audio_encoder)
audio_encoder = keras.models.Model( inputs=input_audio, outputs=audio_encoder )

# hand dense
hand_dense = keras.layers.Dense(256, activation='relu')(input_hand)
hand_dense = keras.layers.Dense(128, activation='relu')(hand_dense)
hand_dense = keras.models.Model( inputs=input_hand, outputs=hand_dense )

# combine inputs
combined = keras.layers.concatenate([ audio_encoder.outputs[0], hand_dense.outputs[0] ], axis=-1)

latent = keras.layers.Dense(latent_size, activation='relu')(combined)
latent = keras.layers.Dropout(0.3)(latent)
latent = keras.layers.BatchNormalization()(latent)
latent = keras.layers.Dense(2*latent_size, activation='relu')(latent)
latent = keras.layers.Dropout(0.3)(latent)
latent = keras.layers.BatchNormalization()(latent)
latent = keras.layers.Dense(latent_size, activation='relu')(latent)
latent = keras.layers.Reshape( (1,6,latent_size//6) )(latent)
# latent = keras.models.Model( inputs=[audio_encoder.input, hand_dense.input], outputs=latent )

decoder = keras.layers.Conv2DTranspose(latent_size//6, kernel_size=3, strides=2, padding='valid', activation='selu', input_shape=[1,6,latent_size//6])(latent)
decoder = keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='selu')(decoder)
decoder = keras.layers.Lambda(lambda x: x[:,:,:-1,:])(decoder)
decoder = keras.layers.Reshape([6, 25])(decoder)

model = keras.models.Model( inputs=[audio_encoder.input, hand_dense.input], outputs=decoder )

'''
out_layer = keras.models.Sequential([
    keras.layers.Lambda(lambda x: x[:,:,:-1,:]),
    keras.layers.Reshape([6*25]),
    keras.layers.Dense(y_train.shape[1], activation='sigmoid')
])
'''
# model = keras.models.Sequential([encoder, latent, decoder])
# model = keras.models.Sequential([encoder, latent])

audio_encoder.summary()
hand_dense.summary()
# latent.summary()
# decoder.summary()
model.summary()

# %%

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['cosine_similarity'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rounded_accuracy])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['cosine_similarity'])

# %%

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

os.makedirs( 'models', exist_ok=True )
os.makedirs( 'models/hand', exist_ok=True )

filepath = 'models/hand/tab_hand_full_CNN_out_epoch{epoch:02d}_valLoss{val_loss:.6f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

filepath_current_best = 'models/hand/tab_hand_full_CNN_out_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath_current_best,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

if os.path.exists('/models/hand/full_tab_hand_logger.csv'):
    os.remove('/models/hand/full_hand_tab_logger.csv')
csv_logger = CSVLogger('models/hand/full_tab_hand_logger.csv', append=True, separator=';')

# %% 

history = model.fit( [x_audio_train, x_hand_train], y_tab_train, epochs=10000, batch_size=128, validation_data=([x_audio_valid, x_hand_valid], y_tab_valid) , callbacks=[checkpoint, checkpoint_current_best, csv_logger] )

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
