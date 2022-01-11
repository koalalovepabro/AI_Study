import cv2
import dlib
from imutils import face_utils, resize
import numpy as np

# 이미지 불러오기
orange_img = cv2.imread('data/orange.jpg')
# 가로 512 , 세로 512로 사이즈 조정
orange_img = cv2.resize(orange_img, dsize=(512, 512))

# 얼굴영역 탐지
detector = dlib.get_frontal_face_detector()
# 얼굴 특징점(landmark) 탐지
# 모델파일 다운받아서 사용
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# 오렌지에 합성할 비디오 불러오기
# cap = cv2.VideoCapture('data/face.mp4')

# 웹캠사용
cap = cv2.VideoCapture(0)

# 카메라 속성 설정 메서드(capture.set)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 카메라의 속성(너비)의 값을 설정
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 카메라의 속성(높이)의 값을 설정

# print(len(cap.read()))   # 2

while cap.isOpened():      # capture.isOpened: 동영상 파일 열기 성공여부 확인
    ret, img = cap.read()  # capture.read: 카메라의 상태 및 프레임을 받아옴
                           # ret은 카메라의 상태가 저장되며, 정상 작동할 경우 True 반환
    if not ret:            # img를 읽고 프레임이 더이상 없으면, 반복문 종료
        break

    # 얼굴 인식
    faces = detector(img)

    result = orange_img.copy()

    # 얼굴이 안나올때 예외처리
    if len(faces) == 0:
        continue

    # 얼굴이 1개이상이면, 첫번째 얼굴만 사용
    if len(faces) > 0:
        face = faces[0]

        # 얼굴의 좌, 우, 상, 하
        x1, x2, y1, y2 = face.left(), face.right(), face.top(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy()  # 인식된 얼굴만 crop해서 저장

        # 얼굴의 특징점 68개 구하기
        shape = predictor(img, face)
        # dlib 형태를 numpy로 바꿔줌
        shape = face_utils.shape_to_np(shape)

        for p in shape:
            cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        # 눈
        le_x1 = shape[36, 0] # 왼쪽눈의 left
        le_y1 = shape[37, 1] # 왼쪽눈의 top
        le_x2 = shape[39, 0] # 왼쪽눈의 right
        le_y2 = shape[41, 1] # 왼쪽눈의 bottom
        le_margin = int((le_x2-le_x1)*0.18)     # 왼쪽눈의 좌우길이 * 0.18만큼의 여유

        re_x1 = shape[42, 0] # 오른쪽눈의 left
        re_y1 = shape[43, 1] # 오른쪽눈의 top
        re_x2 = shape[45, 0] # 오른쪽눈의 right
        re_y2 = shape[47, 1] # 오른쪽눈의 bottom
        re_margin = int((re_x2-re_x1)*0.18)      # 왼쪽눈의 좌우길이 * 0.18만큼의 여유

        # 너무 타이트하게 자르면 안되니까, 마진 포함해서 crop
        left_eye_img = img[ le_y1 - le_margin : le_y2 + le_margin, le_x1 - le_margin : le_x2 + le_margin ].copy()
        right_eye_img = img[ re_y1 - re_margin : re_y2 + re_margin, re_x1 - re_margin : re_x2 + re_margin ].copy()

        # 왼쪽눈, 오른쪽눈 crop한 이미지를 가로 100으로  resize
        left_eye_img = resize(left_eye_img, width=100)
        right_eye_img = resize(right_eye_img, width=100)

        # cv2.seamlessClone : Poisson Blending ( 티가 안나게 합성해주는 메소드 )
        result = cv2.seamlessClone(
            left_eye_img,   # 왼쪽눈을 합성
            result,         # result(오렌지 이미지를 copy한 이미지)에 합성
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            (100, 200),
            cv2.MIXED_CLONE   # 알아서 잘 합성
        )

        result = cv2.seamlessClone(
            right_eye_img,  # 오른쪽눈을 합성
            result,         # result(오렌지 이미지를 copy한 이미지)에 합성
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            (250, 200),
            cv2.MIXED_CLONE  # 알아서 잘 합성
        )

        # 입
        mouth_x1 = shape[48, 0]  # 입의 left
        mouth_y1 = shape[50, 1]  # 입의 top
        mouth_x2 = shape[54, 0]  # 입의 right
        mouth_y2 = shape[57, 1]  # 입의 bottom
        mouth_margin = int((mouth_x2 - mouth_x1)*0.1)

        mouth_img = img[mouth_y1 - mouth_margin : mouth_y2 + mouth_margin, mouth_x1 - mouth_margin : mouth_x2 + mouth_margin ].copy()
        mouth_img = resize(mouth_img, width=250)

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (180, 320),
            cv2.MIXED_CLONE
        )

        # cv2.imshow(winname, mat) : 특정 윈도우 창에 이미지 띄움(winname:윈도우 창의 제목, mat:이미지)
        cv2.imshow('left', left_eye_img)    # 왼쪽 눈
        cv2.imshow('right', right_eye_img)  # 오른쪽 눈
        cv2.imshow('mouth', mouth_img)      # 입
        cv2.imshow('face', face_img)        # 내얼굴 (캠 화면)
        cv2.imshow('result', result)        # 오렌지에 내 얼굴 합성한 것
        # cv2.waitKey(1)

    # 키 입력 대기함수(cv2.waitKey) : 지정된 시간동안 키 입력이 있을 때까지 프로그램을 지연시킴
    if cv2.waitKey(1) == ord('q'):   # q가 입력될 때 프로그램 종료
        break

    # 결과 이미지 저장
    cv2.imwrite("output/result.jpg", result[:])