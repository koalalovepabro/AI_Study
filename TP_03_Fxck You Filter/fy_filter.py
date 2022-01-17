# single.py 코드 복붙
import cv2
import mediapipe as mp
import numpy as np

# 인식할 손의 최대 갯수 (기본값: 2)
max_num_hands = 2

# 제스처 저장 ( 손가락 관절의 각도와 각각의 label)
gesture = {
    0: 'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands             # 손가락을 인식해서 뼈마디를 그려주는 기능
mp_drawing = mp.solutions.drawing_utils   # 손가락을 인식해서 뼈마디를 그려주는 기능
hands = mp_hands.Hands(                   # 손가락 인식 모듈 초기화
    max_num_hands = max_num_hands,        # 인식 가능한 손의 최대 갯수
    min_detection_confidence = 0.5,       # 인식이 성공한 것으로 간주되는 hand detection 모델의 최소 신뢰도 값 [0.0,1.0]
    min_tracking_confidence = 0.5)        # landmark가 성공적으로 추적된 것으로 간주되는 landmark tracking 모델의 최소 신뢰도 값 [0.0,1.0]

# Gesture recognition model
file = np.genfromtxt('data/gesture_train_fy.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)     # angle: 모든 행 , 가장 마지막 열을 제외한 값
label = file[:, -1].astype(np.float32)      # label: 모든 행 , 가장 마지막 열의 값
knn = cv2.ml.KNearest_create()              # KNN 알고리즘
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # angle, label 데이터를 가지고 knn 알고리즘 학습

# 웹캠의 이미지 읽어오기
cap = cv2.VideoCapture(0)

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
            # print(joint)
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
            ret, results, neighbors, dist = knn.findNearest(data, 3)  # k=3 일때의 값 구하기
            idx = int(results[0][0]) # results의 첫번째 인덱스 저장

            # 제스처 모델에서 추론한 결과가 11(뻐큐)라면
            if idx == 11:
                # print('min: ', joint.min())  # -0.13
                # print('max: ', joint.max())  #  0.89
                # print(img.shape)  # 480, 640, 3 (width, height, channel)

                # 손 landmark의 사각형 바운딩박스 구하기
                x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))  # 좌측 width, 위쪽 height 키우기
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))  # 우측 width, 아랫쪽 height 키우기

                # 사각형 영역을 잘라서 fy_img에 복사
                fy_img = img[y1:y2, x1:x2].copy()

                # 이미지의 크기를 0.05배로 작게 만듦
                # 배율이 작아질수록 모자이크 픽셀의 크기가 커짐(더 큰 비율로 축소했다가 다시 확대하니 픽셀이 많이 깨짐)
                # 절대크기 인수(dsize): 축소 후 사이즈를 지정하지 않고 None을 설정
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)

                # 작게 만들었던 이미지를 다시 원본 크기로 늘려줌
                # dsize: 축소 전의 원래 사이즈로 지정 (확대 후 사이즈. tuple(w,h))
                # interpolation : 보간법 지정
                fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                # 모자이크 처리된 이미지를 원본 이미지의 손 부분에 다시 붙여줌
                img[y1:y2, x1:x2] = fy_img

                # # 사각형 바운딩박스를 파란색으로 표시
                # cv2.rectangle(img, pt1=(x1, y1), pt2 = (x2, y2), color=255, thickness=2)

                # FY 텍스트 띄우기
                # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

            # 손가락 마디마디에 landmark 그리기
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Filter', img)
    if cv2.waitKey(1) == ord('q'):
        break

    # # 결과 이미지 저장
    # cv2.imwrite("output/output_fy_filter.jpg", img[:])