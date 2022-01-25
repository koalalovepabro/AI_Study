import cv2
import numpy as np
import time, argparse
import mediapipe as mp

############################################################### [ 손 제스처 인식 ]
# 손 제스처 인식하는 함수
# 하트제스처가 감지되었을 때 -> return 값 True 반환
def get_hand_gesture(mp_hands):
    idx = -13
    for res in mp_hands.multi_hand_landmarks:  # 여러개의 손을 인식할 수 있기 때문에 for문 사용
        joint = np.zeros((21, 3))              # 빨간 점으로 표시되는 각 마디(joint)의 좌표(x,y,z) 저장

        for j, lm in enumerate(res.landmark):  # 각 joint 마다 landmark 저장
            joint[j] = [lm.x, lm.y, lm.z]      # landmark의 x,y,z 좌표를 각 joint에 저장. (21,3)의 array가 생성됨

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

        print(gesture[idx])

    # 손 모양이 하트 제스처(label=12)라면, True 반환
    if idx == 12:
        return True

    return False
####################################################################################

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser()
# 입력받을 인자값 등록
parser.add_argument('--video', help='Input video path')
# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

########################## [MediaPipe sefie segmentation 초기화]
### MediaPipe, selfie-segmentation ( 배경분리해주는 solution )
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # 0: general model, 1: landscape model (속도 더 빠름)
################################################################

########################## [손모양 KNN 학습]
# 인식할 손의 최대 갯수 (기본값: 2)
max_num_hands = 2

# 제스처 저장 ( 손가락 관절의 각도와 각각의 label)
gesture = {
    0: 'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 12:'heart'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands             # 손가락을 인식해서 뼈마디를 그려주는 기능
mp_drawing = mp.solutions.drawing_utils   # 손가락을 인식해서 뼈마디를 그려주는 기능
hands = mp_hands.Hands(                   # 손가락 인식 모듈 초기화
    max_num_hands = max_num_hands,        # 인식 가능한 손의 최대 갯수
    min_detection_confidence = 0.5,       # 인식이 성공한 것으로 간주되는 hand detection 모델의 최소 신뢰도 값 [0.0,1.0]
    min_tracking_confidence = 0.5)        # landmark가 성공적으로 추적된 것으로 간주되는 landmark tracking 모델의 최소 신뢰도 값 [0.0,1.0]

# Gesture recognition model
file = np.genfromtxt('data/gesture_train_heart.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)     # angle: 모든 행 , 가장 마지막 열을 제외한 값
label = file[:, -1].astype(np.float32)      # label: 모든 행 , 가장 마지막 열의 값
knn = cv2.ml.KNearest_create()              # KNN 알고리즘
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # angle, label 데이터를 가지고 knn 알고리즘 학습
############################################

# video를 불러오거나, video가 없으면 웹캠 사용
cap = cv2.VideoCapture(args.video if args.video else 0)
time.sleep(3)  # 웹캠이 켜지기까지 잠깐 멈춰서 기다리기

# 비디오 앞부분에 사람이 나오지 않은 배경이 꼭 필요함 (60프레임 캡쳐)
# Grap background image from first part of the video
for i in range(60):
    ret, background = cap.read()

# 결과 video 저장
# fourcc (Four Character Code. 4-문자코드): 동영상 파일의 코덱, 압축방식, 색상, 픽셀 포맷 등을 정의하는 정수 값
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')               # video이기 때문에, MP4V로 코덱셋팅

# 저장1. 투명인간 적용 이미지
out = cv2.VideoWriter('output/output_levelup_02.mp4', fourcc,
                      20,                                          # 영상 FPS(Frame Per Second): 고정값으로 설정해야 에러 안남
                      (background.shape[1], background.shape[0]))  # 프레임크기는 background와 같은 사이즈로 설정
# 저장2. 웹캠상 이미지
out2 = cv2.VideoWriter('output/original_levelup_02.mp4', fourcc,
                       20,
                       (background.shape[1], background.shape[0]))

while (cap.isOpened()):
    ret, img = cap.read()  # 웹캠을 한 프레임씩 읽어오기
    if not ret:
        break

    # MediaPipe 모델에 넣기 전에 전처리
    # BGR -> RGB (OpenCV는 BGR 컬러시스템, MediaPipe는 RGB 사용하기 때문)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ###################################### [손 탐지 후, inference]
    image2 = image.copy()
    image2 = cv2.flip(image2, 1)  # 좌우반전
    hands_result = hands.process(image2)

    if hands_result.multi_hand_landmarks is not None:  # 손 landmark 탐지된게 있다면,
        que_signal = get_hand_gesture(hands_result)
        print(que_signal)
    else:
        que_signal = False
    ##############################################################

    if not que_signal: # que_signal이 없다면
        result = img   # 웹캠의 이미지 그대로 !
    else:
        ################# [selfie segmentation으로 mask 생성]
        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        results = selfie_segmentation.process(image)

        # RGB -> BGR (이미지를 OpenCV로 출력해야 하니까 다시 변환)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 웹캠에서 사람영역만 masking
        mask1 = results.segmentation_mask
        mask1 = np.where(mask1 > 0.1, 255, 0)
        mask1 = mask1.astype('uint8')
        #####################################################

        # Remove noise
        # Refining the mask corresponding to the detected red color
        # 망토(흰색영역)
        mask_cloak = cv2.morphologyEx(mask1, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
        mask_cloak = cv2.dilate(mask_cloak, kernel=np.ones((3, 3), np.uint8), iterations=1)

        # 사람영역을 제외시키기
        mask_bg = cv2.bitwise_not(mask_cloak)

        cv2.imshow('mask_cloak', mask_cloak)  # mask_cloack  (사람영역만 뽑아서 흰색으로 표시한 것)

        # Generate the final output
        res1 = cv2.bitwise_and(background, background, mask=mask_cloak)  # background에서 사람영역만큼만 뽑은 것
        res2 = cv2.bitwise_and(img, img, mask=mask_bg)                   # 현재 웹캠 이미지에서 사람영역을 제외한 영역

        # 2개 이미지를 합성하기 ( cv2.addWeighted )
        result = cv2.addWeighted(src1=res1, alpha=1, src2=res2, beta=1, gamma=0)

    # cv2.imshow('res1', res1)      # res1   (background에서 사람영역만큼만 뽑은 것)
    # cv2.imshow('res2', res2)      # res2   (웹캠상 이미지에서 사람영역 제외한 것)
    cv2.imshow('ori', img)        # 웹캠 원본 이미지
    cv2.imshow('result', result)  # result (투명인간 적용한 결과 이미지)

    out.write(result)  # 결과 이미지
    out2.write(img)    # 원본 이미지 (웹캠상 이미지)

    if cv2.waitKey(1) == ord('q'):
        break

    # 결과를 이미지로 저장
    cv2.imwrite("output/original_levelup_02.jpg", img[:])               # 웹캠 원본 이미지
    # cv2.imwrite("output/mask_cloack_levelup_02.jpg", mask_cloak[:])     # mask_cloack  (사람영역만 뽑아서 흰색으로 표시한 것)
    # cv2.imwrite("output/res1_levelup_02.jpg", res1[:])                  # res1         (background에서 사람영역만큼만 뽑은 것)
    # cv2.imwrite("output/res2_levelup_02.jpg", res2[:])                  # res2         (웹캠상 이미지에서 사람영역 제외한 것)
    cv2.imwrite("output/result_levelup_02.jpg", result[:])   # result       (투명인간 적용한 결과 이미지)

out.release()
out2.release()
cap.release()