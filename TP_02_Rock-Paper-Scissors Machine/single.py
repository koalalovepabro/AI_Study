import cv2
import mediapipe as mp
import numpy as np

# 제스처 인식할 손의 갯수: 1
max_num_hands = 1

# 제스처 저장
gesture = {
    0: 'fist', 1:'one', 2: 'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10'ok'
}

# 가위바위보(RPS) 제스처 저장
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}  # R-P-S

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# Gesture recognition model


