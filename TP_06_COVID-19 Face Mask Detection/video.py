from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Load Models
## 얼굴 인식 모델 (OpenCV)
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

## 마스크 인식 모델 (keras)
model = load_model('models/mask_detector.model')

# Load Video
cap = cv2.VideoCapture('imgs/04.mp4')
# 웹캠 화면 불러오기
# cap = cv2.VideoCapture(0)
ret, img = cap.read()

## video이기 때문에 MP4V로 코덱셋팅
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

## 결과 video 저장
out = cv2.VideoWriter('result/output2.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]

    # OpenCV의 facenet 모델을 사용하기 때문에, 그 모델에서 학습시킨대로 parameter를 설정함
    # dnn 모듈이 사용하는 형태의 이미지를 변형 (axis 순서만 바뀜)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))

    # 모델에 input 데이터 넣어주기
    facenet.setInput(blob)

    # 결과를 추론하고, dets에 저장
    dets = facenet.forward()

    result_img = img.copy()

    # 여러개의 얼굴이 인식될 수 있기때문에 for문 사용
    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]   # detection한 결과의 신뢰도
        if confidence < 0.5:            # 신뢰도가 0.5 미만인 결과는 pass함
            continue

        # x, y의 바운딩박스 구하기
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        # 원본이미지에서 바운딩박스에 담긴 얼굴만 잘라내기
        face = img[y1:y2, x1:x2]

        # 이미지 전처리
        face_input = cv2.resize(face, dsize=(224, 224))           # 이미지 크기 변환
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)  # 이미지의 컬러시스템 변경(BGR -> RGB)
        face_input = preprocess_input(face_input)                 # mobilenet_v2에서 하는 preprocessing과 똑같이 해줌 (224, 224, 3)
        face_input = np.expand_dims(face_input, axis=0)           # shape을 (1, 224, 224, 3)으로 변환시키기 위해 차원추가

        # 예측
        mask, nomask = model.predict(face_input).squeeze()

        # 마스크 쓴 확률이 더 높을 경우
        if mask > nomask:
            color = (0, 255, 0)                  # 바운딩박스 컬러 green
            label = 'Mask %d%%' % (mask * 100)

        # 마스크 안쓴 확률이 더 높을 경우
        else:
            color = (0, 0, 255)                  # 바운딩박스 컬러 red
            label = 'No Mask %d%%' % (nomask * 100)

        # 사각형 바운딩박스 그리기
        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)

        # 텍스트 넣기
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

    out.write(result_img)
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break

# 결과를 이미지로 저장
cv2.imwrite("result/output_img2.jpg", result_img[:])

out.release()
cap.release()