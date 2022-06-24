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
import cv2
import Hands_lib
import mediapipe as mp
import math

# pip install opencv-python mediapipe protobuf==3.20.* matplotlib scipy numpy PyGuitarPro librosa

# load model
model = keras.models.load_model( 'models/hand/tab_hand_full_CNN_out_current_best.hdf5' )

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

#### gbastas ####
mu, cov = Hands_lib.learn_params()
cov_init = np.copy(cov)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(1)
prevTime = 0
prev_rel_dist_from_nut = 0
pinky_tip_x, pinky_tip_y = None, None
valid_Iout, valid_pb, valid_pn = None, None, None
#### gbastas ####
# after starting, check when n empties (file ends) and stop
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while output1.is_active() and not user_terminated and cap.isOpened():

        #### gbastas ####
        success, image = cap.read()
        width =  image.shape[1]
        height = image.shape[0]
        pinky_binary = np.zeros(25)
        if not success:
            print("Ignoring empty camera frame.")
            continue

        Ipr, I_out, pb, pn = Hands_lib.get_markers(image, mu, cov, threshold=0.15)
        if (pb is not None and pn is not None):
            image, pinky_pos, valid_Iout, valid_pb, valid_pn = Hands_lib.compute_pinky_rel_position(image, I_out, pb, pn, prev_rel_dist_from_nut, prevTime, 
                                                                                                    valid_pb, valid_pn, valid_Iout, pinky_tip_x, pinky_tip_y)
        #### gbastas ####

        c = 1.059463
        pinky_fret = int( math.log(1/(1-pinky_pos), c) )
        pinky_binary[pinky_fret] = 1

        bb = copy.deepcopy( global_block[:1600] )
        if np.max( np.abs( bb ) ) > 0.05:
            bb = bb/np.max( np.abs( bb ) )
        y_pred = model.predict( [ np.reshape( bb, (1,1600,1) ), pinky_binary ] )
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
