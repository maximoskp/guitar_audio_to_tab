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
# from time import sleep
from threading import Thread
import copy
import cv2
import Hands_lib
import mediapipe as mp
import math
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

WINDOW_SIZE = 2048
CHANNELS = 1
RATE = 16000
FFT_FRAMES_IN_SPEC = 100

class AudioProcessor():
	def __init__(self):
		self.audio = pyaudio.PyAudio()
		self.device_index = self.get_device_index()

		self.global_block = np.zeros( WINDOW_SIZE*2 )
		self.fft_frame = np.array( WINDOW_SIZE//2 )
		self.win = np.hamming(WINDOW_SIZE)
		self.spec_img = np.zeros( ( WINDOW_SIZE//2 , FFT_FRAMES_IN_SPEC ) )
		while True:
			try: 
				self.stream = self.audio.open(format=pyaudio.paInt16,
											channels=CHANNELS,
											rate=RATE,
											output=True,
											input=True,
											input_device_index=self.device_index,
											frames_per_buffer=WINDOW_SIZE,
											stream_callback=self.callback)
				break

			except OSError as e:
				print('[gb]', e)
				self.device_index = self.get_device_index(self.device_index+1)
				print('device_index', self.device_index)
		with tf.device('CPU'):
			self.model = keras.models.load_model( 'models/hand/tab_hand_full_CNN_out_current_best.hdf5' )

		print('Fine')

	def get_device_index(self, init_index=0):
		print('[gb] initialize pyAudio:', self.audio.get_device_count(), flush=True)
		device_index = init_index
		for i in range(device_index, self.audio.get_device_count()):
			d = self.audio.get_device_info_by_index(i)
			if self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0 and self.audio.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels') == 0:
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

	# def receive(self):

	def user_input_function(self):
		k = input('press "s" to terminate (then press "Enter"): ')
		if k == 's' or k == 'S':
			global user_terminated
			user_terminated = True

	def start(self):
		# threaded_input = Thread( target=self.receive )
		threaded_input = Thread( target=self.user_input_function )
		threaded_input.start()
		
		global hand_position, pinky_fret#, final_image
		self.stream.stop_stream() # for some reason I just need to stop it first...
		time.sleep(0.1)
		self.stream.start_stream()

		plt.ion() # interactive matplotlib mode
		print('stream.is_active():', self.stream.is_active())
		while self.stream.is_active():
			bb = copy.deepcopy( self.global_block[:1600] )
			if np.max( np.abs( bb ) ) > 0.05:
				bb = bb/np.max( np.abs( bb ) )

			# try:
			# 	cv2.imshow('MediaPipe Hands', final_image )
			# except NameError:
			# 	print("Not ready for image.")
			# try:
			# 	print('pinky_fret:', pinky_fret)
			# except NameError:
			# 	print("Not ready for image.")

			y_pred = self.model.predict( [ np.reshape( bb, (1,1600,1) ), [ np.reshape(hand_position, (1,25,1) )] ] )
			# print('stream.is_active():', self.stream.is_active())
			plt.clf()
			plt.subplot(2,1,1)
			plt.plot( bb )
			plt.ylim([-1,1])
			plt.xticks([])
			plt.title('input')
			plt.subplot(2,1,2)
			plt.imshow( y_pred[0,:,:] , cmap='gray_r' )
			plt.title('output')
			# plt.show(block=True)
			plt.show()
			plt.pause(0.01)
			# time.sleep(0.01)



class VideoProcessor():
	def __init__(self):
		self.L0 = self.calculate_nut2bridge_distance()
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_hands = mp.solutions.hands
		self.cap = cv2.VideoCapture(0)

	def calculate_nut2bridge_distance(self):
		c = 1.059463
		C = 1/c**24
		# Lₙ is the length of the string from any fret "n" to the bridge
		L24 = C / (1-C) # this is derived from Ln = L0/c**n standard formula, by setting LO = 1 + L24 assuming neck-length to be 1 [https://www.omnicalculator.com/other/fret]
		L0 = 1 + L24 # L0 is the distance from nut to bridge
		return L0

	def make_hand_box(self, highest_fret):
		handbox = np.zeros(25)
		handbox[highest_fret] = 1
		if highest_fret-1 >= 0:
			handbox[highest_fret-1] = 1
		if highest_fret-2 >= 0:
			handbox[highest_fret-2] = 1
		if highest_fret+1 <= 24:
			handbox[highest_fret+1] = 0.5
		if highest_fret+2 <= 24:
			handbox[highest_fret+2] = 0.25
		if highest_fret-3 <= 24:
			handbox[highest_fret-3] = 0.5
		if highest_fret-4 <= 24:
			handbox[highest_fret-4] = 0.25
		return handbox


	def receive(self):
		global hand_position, pinky_fret, final_image
		valid_Iout, valid_pb, valid_pn = None, None, None  
		pinky_pos=0
		c = 1.059463
		plt.ion()
		pb, pn = np.array([0,0]), np.array([0,0])
		model_filename = 'detector_good18.pth'
		# model_filename = 'detector_good_mobilenetv2.pth' # TODO: not running yet
		print('VISUAL MODEL:', model_filename)

		with self.mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.2) as hands:

			while self.cap.isOpened():
				#### gbastas ####
				success, image = self.cap.read()
				
				if not success:
					print("Ignoring empty camera frame.")
					continue

				image, pb, pn = Hands_lib.get_bbox(image, model_filename=model_filename)
				
				image, pinky_pos, valid_Iout, valid_pb, valid_pn = Hands_lib.compute_pinky_rel_position(image, np.zeros(image.shape), pb, pn, pinky_pos, 0, 
																										valid_pb, valid_pn, valid_Iout, None, None, 
																										hands, self.mp_drawing, self.mp_hands)
				pinky_fret = int( math.log(self.L0/(self.L0-pinky_pos), c) )  # this is derived from formula dist_from_nut = Lo - (L0 / c**n) [https://www.omnicalculator.com/other/fret]
				pinky_fret = min(pinky_fret, 24) 
				pinky_fret = max(pinky_fret, 0)
				hand_position = self.make_hand_box( pinky_fret )

				final_image = cv2.flip(image, 1)
				cv2.putText(final_image, f'Pinky Pos: {pinky_pos, pinky_fret}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
				cv2.imshow('MediaPipe Hands', final_image)
				key = cv2.waitKey(1)
				if key == 27:  # exit on ESC
					break				

	def start(self):
		threaded_input = Thread( target=self.receive )
		threaded_input.start()

def start_AVprocessing():
	video_thread = VideoProcessor()
	audio_thread = AudioProcessor()    
	video_thread.start()
	audio_thread.start()

if __name__ == '__main__':
	hand_position = np.zeros(25)
	start_AVprocessing()