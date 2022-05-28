import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import pyaudio
# import wave
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from threading import Thread
import copy

# load model
model = keras.models.load_model( 'models/tab_full_CNN_out_current_best.hdf5' )

device_1_index = 0
device_2_index = -1

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    print(d)
    if 'Mic/Inst/Line In 1/2 (Studio 18' in d['name'] and d['hostApi'] == 0:
        device_1_index = d['index']
    if 'Mic/Line In 3/4 (Studio 18' in d['name'] and d['hostApi'] == 0:
        device_2_index = d['index']

WINDOW_SIZE = 2048
CHANNELS = 1
RATE = 16000

FFT_FRAMES_IN_SPEC = 100

# global
# n = np.zeros(1)
global_block = np.zeros( WINDOW_SIZE*2 )
fft_frame = np.array( WINDOW_SIZE//2 )
win = np.hamming(WINDOW_SIZE)
spec_img = np.zeros( ( WINDOW_SIZE//2 , FFT_FRAMES_IN_SPEC ) )

user_terminated = False

# %% call back with global

def callback( in_data, frame_count, time_info, status):
    global global_block, f, fft_frame, win, spec_img
    # global_block = f.readframes(WINDOW_SIZE)
    n = np.frombuffer( in_data , dtype='int16' )
    global_block = 6*(n/np.iinfo(np.int16).max).astype('float32')
    # begin with a zero buffer
    b = np.zeros( (n.size , CHANNELS) , dtype='int16' )
    # 0 is left, 1 is right speaker / channel
    b[:,0] = n
    # for plotting
    # audio_data = np.fromstring(in_data, dtype=np.float32)
    if len(win) == len(n):
        frame_fft = np.fft.fft( win*n )
        p = np.abs( frame_fft )*2/np.sum(win)
        fft_frame = 20*np.log10( p[ :WINDOW_SIZE//2 ] / 32678 )
        spec_img = np.roll( spec_img , -1 , axis=1 )
        spec_img[:,-1] = fft_frame[::-1]
    return (b, pyaudio.paContinue)

def user_input_function():
    k = input('press "s" to terminate (then press "Enter"): ')
    if k == 's' or k == 'S':
        global user_terminated
        user_terminated = True

# %% create output stream
p = pyaudio.PyAudio()

output1 = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                input_device_index=device_1_index,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=callback)

if device_2_index >= 0:
    output2 = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    input=True,
                    input_device_index=device_2_index,
                    frames_per_buffer=WINDOW_SIZE,
                    stream_callback=callback)


output1.start_stream()
if device_2_index >= 0:
    output2.start_stream()

threaded_input = Thread( target=user_input_function )
threaded_input.start()

# after starting, check when n empties (file ends) and stop
while output1.is_active() and not user_terminated:
    bb = copy.deepcopy( global_block[:1600] )
    if np.max( np.abs( bb ) ) > 0.05:
        bb = bb/np.max( np.abs( bb ) )
    y_pred = model.predict( np.reshape( bb, (1,1600,1) ) )
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot( bb )
    plt.ylim([-1,1])
    plt.xticks([])
    plt.title('input')
    plt.subplot(2,1,2)
    plt.imshow( y_pred[0,:,:] , cmap='gray_r' )
    plt.title('output')
    # plt.imshow( spec_img[ WINDOW_SIZE//4: , : ] , aspect='auto' )
    plt.show()
    plt.pause(0.01)

print('stopping audio')
output1.stop_stream()
if device_2_index >= 0:
    output2.stop_stream()
