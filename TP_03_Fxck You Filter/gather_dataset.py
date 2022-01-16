# TP_02_Rock-Paper-Scissors Machine의 'single.py' 코드 기반
# 손가락 마디의 각도 계산하는 부분만 남겨둠
import cv2
import mediapipe as mp
import numpy as np

# 인식할 손의 최대 갯수 (기본값: 2)
max_num_hands = 1

# 제스처의 클래스 저장 ( 손가락 관절의 각도와 각각의 label)
gesture = {
    0: 'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy' # 11번에 fsck you 제스처 저장
}

# MediaPipe hands model
mp_hands = mp.solutions.hands             # 손가락을 인식해서 뼈마디를 그려주는 기능
mp_drawing = mp.solutions.drawing_utils   # 손가락을 인식해서 뼈마디를 그려주는 기능
hands = mp_hands.Hands(                   # 손가락 인식 모듈 초기화
    max_num_hands = max_num_hands,        # 인식 가능한 손의 최대 갯수
    min_detection_confidence = 0.5,       # 인식이 성공한 것으로 간주되는 hand detection 모델의 최소 신뢰도 값 [0.0,1.0]
    min_tracking_confidence = 0.5)        # landmark가 성공적으로 추적된 것으로 간주되는 landmark tracking 모델의 최소 신뢰도 값 [0.0,1.0]

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')  # 데이터 갯수: 110개

# 웹캠의 이미지 읽어오기
cap = cv2.VideoCapture(0)

# 클릭했을때 실행할 함수
# 클릭했을 때 현재 각도 data를 원본 file에 추가
def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data))
        print(file.shape)
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     file = np.vstack((file, data))
    #     print(file.shape)

# 화면을 클릭해을 때만 데이터 저장 (openCV click event)
cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

while cap.isOpened():
    ret, img = cap.read()  # 웹캠에서 프레임 하나씩 읽어옴
    if not ret:            # 읽어오지 못했다면 (False)
        continue           # 다음 프레임으로 넘어감

    # MediaPipe 모델에 넣기전에 전처리
    img = cv2.flip(img, 1)  # 이미지 반전 (1: 좌우, 0: 상하)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB (MediaPipe에 넣기위해)
                                                # OpenCV는 BGR 컬러시스템, MediaPipe는 RGB 사용하기 때문

    result = hands.process(img)  # 전처리 및 모델 추론을 함께 실행 (전처리 된 이미지가 result에 저장됨)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR (이미지를 OpenCV로 출력해야 하니까 다시 변환)

    if result.multi_hand_landmarks is not None:    # 손을 인식했다면,
        for res in result.multi_hand_landmarks:   # 여러개의 손을 인식할 수 있기때문에 for문 사용
            joint = np.zeros((21, 3))             # 빨간 점으로 표시되는 각 마디(joint)의 좌표(x,y,z) 저장
                                                  # np.zeros((21,3)) : 21개의 조인트, x,y,z 3개의 좌표
            for j, lm in enumerate(res.landmark): # 각 joint마다 landmark저장
                joint[j] = [lm.x, lm.y, lm.z]     # landmark의 x,y,z 좌표를 각 joint에 저장. (21,3)의 array가 생성됨

            # compute angles between joints  ( 관절마다 벡터값 구하기 )
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20, 3]

            # Normalize v (유클리디안거리로 벡터의 길이 구하기)
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # get angle using arccos of dot product
            # 벡터a와 벡터b의 내적값 = 벡터a의 크기 x 벡터b의 크기 x 두 벡터가 이루는 각의 cos값 => 각도
            # 위에서 벡터의 크기를 모두 1로 표준화시켰기 때문에, 두 벡터의 내적값 = 두 벡터가 이루는 각의 cos값
            # 따라서 이것을 cos역함수인 arccos에 대입하면 두 벡터가 이루는 각을 구할 수 있음
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
                                        ))  # [15, ] 15개의 angle이 계산됨 (radian값으로 계산됨)

            # convert radian to degree
            # π radian = 180도(degree)
            angle = np.degrees(angle)

            # Inference gesture (제스처 추론)
            data = np.array([angle], dtype=np.float32)

            # 각도 데이터의 마지막에 label 11 추가
            data = np.append(data, 11)

            # 손가락 마디마디에 landmark 그리기
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

# 수집한 데이터를 csv 파일로 저장
np.savetxt('data/gesture_train_fy.csv', file, delimiter=',')
