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

def get_markers(I, mu, cov, threshold=None):
  global cov_init
  I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
  IYCrCb = cv2.cvtColor(I, cv2.COLOR_RGB2YCR_CB)  # (Y, Cr, Cb) standard cv2 conversion from RGB
  Ihsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)  # (H, S, V) standard cv2 conversion from RGB
  # ICbCr = IYCrCb[:, :,:-3:-1]  # keep last two items, reversed (check: https://stackoverflow.com/questions/509211/understanding-slice-notation)
  # Ihsv = Ihsv[:, :, 0:3:2]  # keep first an last items
  Ihs = Ihsv[:, :, :2]  # keep first two items
  # Ipr = multivariate_normal.pdf(ICbCr, mu, cov)  # marker probability image
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

  # TODO: maybe need to keep top2 classes with regard to area

  # if nb_labels<2 or nb_labels>4:
  #   return Icls, None, None

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


def onTrack1(val):
    global mu
    print(mu[0], val)
    mu[0]=val
    # print('Mu Hue',mu[0])
def onTrack2(val):
    global mu
    mu[1]=val
    # print('Mu Sat',mu[1])
def onTrack3(val):
    global cov
    # print(type(cov), cov.shape, cov[0,0], val)
    cov[0,0]=val
    # print('Hue-Hue',cov[0,0])
def onTrack4(val):
    global cov
    cov[0,1]=val
    # print('Hue-Sat',cov[0,1])
def onTrack5(val):
    global threshold
    threshold=val



if __name__ == "__main__":
  Mrk = cv2.imread('marker_samples4.png')

  # Get Cr and Cb matrices (i.e. projections each pixel 3D-vector to Cb and Cr axes)
  mrk_hsv = cv2.cvtColor(Mrk, cv2.COLOR_BGR2HSV)
  mh, ms, mv = cv2.split(mrk_hsv)

  mu = [np.mean(mh), np.mean(ms)]  # 2x1
  cov = np.array(np.cov(mh.reshape(-1), ms.reshape(-1)))  # 2x2
  cov_init = np.copy(cov)
  # print(mu, cov)


  #Created by MediaPipe
  #Modified by Augmented Startups 2021
  #Pose-Estimation in 5 Minutes
  #Watch 5 Minute Tutorial at www.augmentedstartups.info/YouTube
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands


  threshold=15
  cv2.namedWindow('myTracker') 
  cv2.createTrackbar('Mu Hue','myTracker',int(mu[0]),179,onTrack1)
  cv2.createTrackbar('Mu Sat','myTracker',int(mu[1]),255,onTrack2)
  cv2.createTrackbar('Cov Hue-Hue','myTracker',int(cov[0,0]),1000,onTrack3)
  cv2.createTrackbar('Cov Sat-Sat','myTracker',int(cov[1,1]),1000,onTrack4)
  cv2.createTrackbar('Threshold','myTracker',int(threshold),100,onTrack5)


  ## For webcam input:
  cap = cv2.VideoCapture(1)
  # cap = cv2.VideoCapture(-1)


  init_pb, init_pn = None, None
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    width =  image.shape[1]
    height = image.shape[0]
    Ipr, I_out, pb, pn = get_markers(image, mu, cov, threshold=threshold/100)


    if pb is not None and pn is not None:
      # print(np.round(pb,2), np.round(pn,2))
      # if np.linal
      init_pb, init_pn = pb, pn

    I_out = I_out * 255
    I_out = I_out[:, ::-1]
    Ipr = Ipr[:, ::-1]

    image = cv2.flip(image, 1)

    image[:, :, 0] = np.maximum(image[:, :, 0], I_out)
    image[:, :, 1] = np.maximum(image[:, :, 1], I_out)
    image[:, :, 2] = np.maximum(image[:, :, 2], I_out)
    if pb is not None and pn is not None:
      image = cv2.circle(image, (int(pb[0]*width), int((1-pb[1])*height)), 5, (255, 0, 0), 2)
      image = cv2.circle(image, (int(pn[0]*width), int((1-pn[1])*height)), 5, (255, 0, 0), 2)    
    cv2.imshow('Prob', Ipr)
    cv2.imshow('Binary', I_out)
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

  ##For Video
  #cap = cv2.VideoCapture("hands.mp4")
  # pb = np.array([0.8, 0.3])
  # pn = np.array([0.1, 0.5])
  valid_Iout, valid_pb, valid_pn = np.copy(I_out), np.copy(init_pb), np.copy(init_pn)
  # valid_pb, valid_pn = None, None
  print("pb, pn", pb, pn)
  prevTime = 0
  prev_rel_dist_from_nut = 0
  pinky_tip_x, pinky_tip_y = None, None
  with mp_hands.Hands(
      min_detection_confidence=0.5,       #Detection Sensitivity
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      width =  image.shape[1]
      height = image.shape[0]
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue


      Ipr, I_out, pb, pn = get_markers(image, mu, cov, threshold=threshold/100)
      if (pb is not None and pn is not None) and np.linalg.norm(pb - valid_pb) < 0.1 and np.linalg.norm(pn - valid_pn) < 0.1:
        valid_pb, valid_pn = np.copy(pb), np.copy(pn)
        valid_Iout = I_out * 255
        valid_Iout = valid_Iout[:, ::-1]
        # print('new points', valid_pb, valid_pn)

      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      h, s, v = cv2.split(hsv)
      ## threshold the S channel using adaptive method(`THRESH_OTSU`)
      th, threshed = cv2.threshold(s, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
      # print("Thresh : {}".format(th))
      # cv2.imwrite("result.png", threshed)

      # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to pass by reference.
      image.flags.writeable = False
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      pinky_prjection_to_neck=None
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
          #  NOTE: vector method failed
          # pinky_prjection_to_neck =  (np.dot(neck_vector, pinky_tip) / np.linalg.norm(neck_vector)) * neck_vector /  np.linalg.norm(neck_vector)
          # rel_dist_from_nut = np.linalg.norm(pinky_prjection_to_neck) / np.linalg.norm(neck_vector)
          # print(rel_dist_from_nut)
          
          rel_dist_from_nut = np.linalg.norm(pinky_tip-valid_pn)/np.linalg.norm(neck_vector)

          if abs(rel_dist_from_nut - prev_rel_dist_from_nut) <0.6 and rel_dist_from_nut<=1.0:
            prev_rel_dist_from_nut = rel_dist_from_nut

          for hand_landmarks in results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                  image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      currTime = time.time()
      fps = 1 / (currTime - prevTime)
      prevTime = currTime
      # cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
      cv2.putText(image, f'Pinky Position: {round(prev_rel_dist_from_nut,2)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

      image[:,:,0] = np.maximum(image[:,:,0], valid_Iout)
      image[:, :, 1] = np.maximum(image[:, :, 1], valid_Iout)
      image[:, :, 2] = np.maximum(image[:, :, 2], valid_Iout)
      image = cv2.circle(image, (int(valid_pb[0]*width), int((1-valid_pb[1])*height)), 5, (255, 0, 0), 2)
      image = cv2.circle(image, (int(valid_pn[0]*width), int((1-valid_pn[1])*height)), 5, (255, 0, 0), 2)
      if pinky_prjection_to_neck is not None:
        image = cv2.circle(image, (int(pinky_prjection_to_neck[0]*width), int((1-pinky_prjection_to_neck[1])*height)), 5, (255, 0, 0), 2)
      cv2.imshow('MediaPipe Hands', image )
      if cv2.waitKey(5) & 0xFF == 27:
        break

  cap.release()
  # Learn more AI in Computer Vision by Enrolling in our AI_CV Nano Degree:
  # https://bit.ly/AugmentedAICVPRO