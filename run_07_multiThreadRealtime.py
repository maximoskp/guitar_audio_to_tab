import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from threading import Thread
import copy
import cv2
import Hands_lib
import mediapipe as mp
import math


WINDOW_SIZE = 2048
CHANNELS = 1
RATE = 16000
FFT_FRAMES_IN_SPEC = 100

print('[gb] initialize pyAudio:', p.get_device_count(), flush=True)
device_index = 0
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    print('[gb] devices', d, flush=True)
    if p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels') > 0 and p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') == 0:
        print()
        print('[gb] *** Chosen output device:', d, flush=True)
        print()
        device_index = d['index']
        break


class AudioProcessor():
    def __init__(self,  channels=2):
        self.audio = pyaudio.PyAudio()
        self.device_index = self.get_device_index()

        self.global_block = np.zeros( WINDOW_SIZE*2 )
        self.fft_frame = np.array( WINDOW_SIZE//2 )
        self.win = np.hamming(WINDOW_SIZE)
        self.spec_img = np.zeros( ( WINDOW_SIZE//2 , FFT_FRAMES_IN_SPEC ) )

        self.stream = self.audio.open(format=pyaudio.paInt16,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        output=True,
                                        input=True,
                                        input_device_index=self.device_1_index,
                                        frames_per_buffer=WINDOW_SIZE,
                                        stream_callback=self.callback)

        self.model = keras.models.load_model( 'models/hand/tab_hand_full_CNN_out_current_best.hdf5' )


    def get_device_index(self):
        print('[gb] initialize pyAudio:', self.audio.get_device_count(), flush=True)
        device_index = 0
        for i in range(self.audio.get_device_count()):
            d = self.audio.get_device_info_by_index(i)
            if self.audio.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels') > 0 and p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') == 0:
                print()
                print('[gb] *** Chosen output device:', d, flush=True)
                print()
                device_index = d['index']
                break
        return device_index

    def callback(self, in_data, frame_count, time_info, status):
        n = np.frombuffer( in_data , dtype='int16' )
        self.global_block = 6*(n/np.iinfo(np.int16).max).astype('float32')
        b = np.zeros( (n.size , CHANNELS) , dtype='int16' ) # begin with a zero buffer     
        b[:,0] = n # 0 is left, 1 is right speaker / channel
        # for plotting
        if len(self.win) == len(n):
            frame_fft = np.fft.fft( self.win*n )
            p = np.abs( frame_fft )*2/np.sum(self.win)
            self.fft_frame = 20*np.log10( p[ :WINDOW_SIZE//2 ] / 32678 )
            self.spec_img = np.roll( self.spec_img , -1 , axis=1 )
            self.spec_img[:,-1] = self.fft_frame[::-1]
        return (b, pyaudio.paContinue)

    def receive(self):
        global pinky_binary
        self.stream.start_stream()
        plt.ion() # interactive matplotlib mode
        while self.stream.is_active():
            bb = copy.deepcopy( self.global_block[:1600] )
            if np.max( np.abs( bb ) ) > 0.05:
                bb = bb/np.max( np.abs( bb ) )

            y_pred = self.model.predict( [ np.reshape( bb, (1,1600,1) ), [ np.reshape(pinky_binary, (1,25,1) )] ] )
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot( bb )
            plt.ylim([-1,1])
            plt.xticks([])
            plt.title('input')
            plt.subplot(2,1,2)
            plt.imshow( y_pred[0,:,:] , cmap='gray_r' )
            plt.title('output')
            plt.show()
            plt.pause(0.01)

    def start(self):
        threaded_input = Thread( target=self.receive )
        threaded_input.start()



def user_input_function():
    k = input('press "s" to terminate (then press "Enter"): ')
    if k == 's' or k == 'S':
        global user_terminated
        user_terminated = True

def start_AVprocessing():
    # global video_thread
    global audio_thread
    # video_thread = VideoProcessor()
    audio_thread = AudioProcessor()    
    audio_thread.start()
    # video_thread.start()

if __name__ == '__main__':
    pinky_binary = 1
    start_AVprocessing()