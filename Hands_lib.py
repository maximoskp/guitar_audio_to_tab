# https://medium.com/augmented-startups/hand-tracking-30-fps-on-cpu-in-5-minutes-986a749709d7

# conda create --name cv python=3.8
# pip install opencv-python mediapipe protobuf==3.20.* matplotlib scipy numpy

import cv2
import mediapipe as mp
import scipy.io
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import time
import math

import torch
import imutils
from fretboard_detection.pyimagesearch import config
from torchvision import transforms
import os 

def compute_pinky_rel_position(image, I_out, pb, pn, prev_rel_dist_from_nut, prevTime, valid_pb, valid_pn, valid_Iout, pinky_tip_x, pinky_tip_y, hands, mp_drawing, mp_hands):
	if valid_pb is None or valid_pn is None:
		valid_pb, valid_pn = pb, pn

	if np.linalg.norm(pb - pn) > 0.05:
		valid_pb, valid_pn = np.copy(pb), np.copy(pn)
		valid_Iout = I_out * 255
		valid_Iout = valid_Iout[:, ::-1]

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	th, threshed = cv2.threshold(s, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

	# Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
	image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# To improve performance, optionally mark the image as not writeable to pass by reference.
	image.flags.writeable = False
	results = hands.process(image)

	# Draw the hand annotations on the image.
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	if results.multi_hand_landmarks:
		if results.multi_handedness[0].classification[0].label=='Left':
			pinky_tip_x = results.multi_hand_landmarks[0].landmark[-1].x
			pinky_tip_y = results.multi_hand_landmarks[0].landmark[-1].y
		elif results.multi_handedness[-1].classification[0].label=='Left':
			pinky_tip_x = results.multi_hand_landmarks[-1].landmark[-1].x
			pinky_tip_y = results.multi_hand_landmarks[-1].landmark[-1].y

		if pinky_tip_x is not None and pinky_tip_y is not None:
			pinky_tip = np.array([pinky_tip_x, 1-pinky_tip_y])
			neck_vector = (valid_pb - valid_pn)

			rel_dist_from_nut = np.linalg.norm(pinky_tip-valid_pn)/np.linalg.norm(neck_vector)

			# print(rel_dist_from_nut)

			# if abs(rel_dist_from_nut - prev_rel_dist_from_nut) <0.6 and rel_dist_from_nut<=1.0:
			if rel_dist_from_nut<=1.0:
				prev_rel_dist_from_nut = rel_dist_from_nut

		for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks(
						image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

	# currTime = time.time()
	# fps = 1 / (currTime - prevTime)
	# prevTime = currTime
	return image, round(prev_rel_dist_from_nut,2), valid_Iout, valid_pb, valid_pn


def get_markers(I, mu, cov, threshold=None):
	global cov_init
	I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
	Ihsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)  # (H, S, V) standard cv2 conversion from RGB
	Ihs = Ihsv[:, :, :2]  # keep first two items
	try: 
		Ipr = multivariate_normal.pdf(Ihs, mu, cov)  # skin probability image
	except ValueError as e:
		Ipr = multivariate_normal.pdf(Ihs, mu, cov_init)  # skin probability image
		print(e)
	Ipr = Ipr / np.max(Ipr)  # Normalize to [0,1]
	_, Ithr = cv2.threshold(Ipr, threshold, 1, cv2.THRESH_BINARY)  # threshold probability image

	# Minkowski Opening
	kernO = np.ones((7, 7))  # Create small morphological kernel for opening
	Iopn = cv2.morphologyEx(Ithr, cv2.MORPH_OPEN, kernO)

	# Minkowski Closing
	kernC = np.ones((25, 25))  # Create large morphological kernel for closing
	Icls = cv2.morphologyEx(Iopn, cv2.MORPH_CLOSE, kernC)

	# Automatic labeling (by neighbour)
	I_labeled, nb_labels = scipy.ndimage.label(Icls)

	try:
		body_marker = np.where(I_labeled == 1)
		xb, yb = int(np.mean(body_marker[1]))/I_labeled.shape[1], int(np.mean(body_marker[0]))/I_labeled.shape[0]

		nut_marker = np.where(I_labeled == 2)
		xn, yn = int(np.mean(nut_marker[1]))/I_labeled.shape[1], int(np.mean(nut_marker[0]))/I_labeled.shape[0]

		if xb > xn: # choose body-point as the right area always (for right-hand people)
			return Ipr, Icls, np.array([1-xn, 1-yn]), np.array([1-xb, 1-yb])
		else:  
			return Ipr, Icls, np.array([1-xb, 1-yb]), np.array([1-xn, 1-yn])
	except ValueError as e:
		return Ipr, Icls, None, None

def get_bbox(image, model_filename='detector_good.pth'): # image = cv2.imread(imagePath).astype(np.float32) / 255 # NOTE: range [0,1]
	MODEL_PATH = os.path.join('./fretboard_detection', 'output', model_filename)
	# model = torch.load(MODEL_PATH).to(config.DEVICE)
	model = torch.load(MODEL_PATH, map_location=torch.device(config.DEVICE))
	model.eval()
	transforms_ = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
		transforms.Normalize(mean=config.MEAN, std=config.STD)
	])
	
	orig = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = image / 255 # NOTE: range [0,1], EXTRA

	image = cv2.resize(image, (224, 224))
	image = image.transpose((2, 0, 1))
	image = torch.from_numpy(image)
	image = transforms_(image).to(config.DEVICE)
	image = image.unsqueeze(0)

	# predict the bounding box of the fretboard
	(boxPreds, labelPreds) = model(image)

	(centerX, centerY, widthX, heightY) = boxPreds[0]
	(startX, startY, endX, endY) = (centerX - widthX/2, centerY - heightY/2, centerX + widthX/2, centerY + heightY/2)
	(bodyX, bodyY, nutX, nutY) = (centerX - widthX/2, centerY + heightY/2, centerX + widthX/2, centerY - heightY/2)
	(bodyX, bodyY, nutX, nutY) = (1-bodyX.detach(), 1-bodyY.detach(), 1-nutX.detach(), 1-nutY.detach())

	orig = imutils.resize(orig, width=600)
	(h, w) = orig.shape[:2]
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(orig, 'neck', (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 255, 0), 2)
	cv2.rectangle(orig, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	# I_bboxd = 255*orig    
	return orig, np.array([bodyX, bodyY]), np.array([nutX, nutY])

def learn_params():
	Mrk = cv2.imread('marker_samples4.png')
	mrk_hsv = cv2.cvtColor(Mrk, cv2.COLOR_BGR2HSV)
	mh, ms, mv = cv2.split(mrk_hsv)
	mu = [np.mean(mh), np.mean(ms)]  # 2x1
	cov = np.array(np.cov(mh.reshape(-1), ms.reshape(-1)))  # 2x2
	return mu, cov

if __name__ == "__main__":

	mu, cov = learn_params()
	cov_init = np.copy(cov)

	mp_drawing = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands

	threshold=15
	valid_Iout, valid_pb, valid_pn = None, None, None


	c = 1.059463
	C = 1/c**24
	L24 = C / (1-C) # this is derived from Ln = L0/c**n standard formula,  by setting LO = 1 + L24 [https://www.omnicalculator.com/other/fret]
	L0 = 1 + L24

	## For webcam input:
	cap = cv2.VideoCapture(0)
	prevTime = 0
	pinky_pos = 0
	pinky_tip_x, pinky_tip_y = None, None
	pinky_binary = np.zeros(25)
	pinky_fret = 0
	with mp_hands.Hands(
			min_detection_confidence=0.5,       #Detection Sensitivity
			min_tracking_confidence=0.5) as hands:
		while cap.isOpened():
			success, image = cap.read()
			width =  image.shape[1]
			height = image.shape[0]
			if not success:
				print("Ignoring empty camera frame.")
				continue

			# image = image/255
			image, pb, pn = get_bbox(image)
			I_out = np.zeros(image.shape)
			image, pinky_pos, valid_Iout, valid_pb, valid_pn = compute_pinky_rel_position(image, np.zeros(image.shape), pb, pn, pinky_pos, prevTime, valid_pb, valid_pn, valid_Iout, pinky_tip_x, pinky_tip_y, hands, mp_drawing, mp_hands)

			# print(pb, pn)
			# print(pinky_pos)

			pinky_fret = int( math.log(L0/(L0-pinky_pos), c) )  # this is derived from formula dist_from_nut = Lo - (L0 / c**n) [https://www.omnicalculator.com/other/fret]
			pinky_fret = min(pinky_fret, 24) 
			pinky_fret = max(pinky_fret, 0)
			pinky_binary[pinky_fret] = 1

			image = cv2.flip(image, 1)
			cv2.putText(image, f'Pinky Pos: {pinky_pos, pinky_fret}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)


			# Ipr, I_out, pb, pn = get_markers(image, mu, cov, threshold=threshold/100)
			# if (pb is not None and pn is not None):
			# 	image, pinky_pos, valid_Iout, valid_pb, valid_pn = compute_pinky_rel_position(image, I_out, pb, pn, pinky_pos, prevTime, valid_pb, valid_pn, valid_Iout, pinky_tip_x, pinky_tip_y, hands)
			# 	image.flags.writeable = True
			# 	# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			# 	pinky_fret = int( math.log(L0/(L0-pinky_pos), c) )  # this is derived from formula dist_from_nut = Lo - (L0 / c**n) [https://www.omnicalculator.com/other/fret]
			# 	pinky_fret = min(pinky_fret, 24) 
			# 	pinky_fret = max(pinky_fret, 0)
			# 	pinky_binary[pinky_fret] = 1

			# 	cv2.putText(image, f'Pinky Position: {pinky_pos, pinky_fret}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

			# 	if valid_Iout is not None:
			# 		image[:,:,0] = np.maximum(image[:,:,0], valid_Iout)
			# 		image[:, :, 1] = np.maximum(image[:, :, 1], valid_Iout)
			# 		image[:, :, 2] = np.maximum(image[:, :, 2], valid_Iout)
			# 		image = cv2.circle(image, (int(valid_pb[0]*width), int((1-valid_pb[1])*height)), 5, (255, 0, 0), 2)
			# 		image = cv2.circle(image, (int(valid_pn[0]*width), int((1-valid_pn[1])*height)), 5, (255, 0, 0), 2)
			cv2.imshow('MediaPipe Hands', image )
			if cv2.waitKey(5) & 0xFF == 27:
				break

	cap.release()
	# Learn more AI in Computer Vision by Enrolling in our AI_CV Nano Degree:
	# https://bit.ly/AugmentedAICVPRO


