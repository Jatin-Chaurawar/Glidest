import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands
cap=cv2.VedioCapture(0)
with mp_hands.Hands(min_detection_cofidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isopen() :
        ret,frame=cap.read()
        image=cv2.cvtColor(frame,cv2.COLOR_HSV2RGB)
        image.cv2.flip(image,1)
        image.flag.writable=False
        result=hands.process(image)
        image.flag.writable=True
        image=cv2.cvtColor(frame,cv2.COLOR_HSV2RGB)
        print(result)
        
        if result.multi_hand_landmarks:
            for num,hand in enumerate(result.multi_hand_landmarks) :
                mp_drawing.draw_landmarks(image,hand,mp,hands)