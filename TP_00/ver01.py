# 얼굴영역 + landmark68 (dlib), 라이언 얼굴 입히기 -> 얼굴 1개만 인식

import cv2, dlib
import numpy as np

# 동영상 사이즈를 20%로 축소하기위한 변수
# cam(내얼굴)으로 테스트 할때에는 scaler값을 0.4이상으로 줘야 인식가능
scaler = 0.4

# 얼굴 인식 모듈 초기화
detector = dlib.get_frontal_face_detector()

# 얼굴 특징점 모듈 초기화
# shape_predictor는 머신러닝으로 학습된 모델이기 때문에 모델 파일이 필요함
# shape_predictor_68_face_landmarks.dat 모델 파일을 다운받아서 사용
predictor = dlib.shape_predictor('samples/shape_predictor_68_face_landmarks.dat')

# 비디오 불러오기
# 파일이름대신 0을 넣으면 웹캠이 켜지고 본인 얼굴로 테스트 가능함
cap = cv2.VideoCapture('samples/faces.mp4')
# 내 얼굴로 테스트
# cap = cv2.VideoCapture(0)

# 동영상의 현재 프레임 초기화
# 동영상의 "현재 프레임 수" = 동영상의 "총 프레임 수" 일 경우 -> 현재 재생되고 있는 프레임은 가장 마지막이 됨
# 마지막 프레임은 동영상이 종료되는 시점이 되므로, 비디오 속성 설정 메서드(capture.get)으로 동영상의 현재 프레임 초기화
# if(cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 얼굴에 씌울 이미지 불러오기
# cv2.IMREAD_UNCHANGED를 해줘야 알파채널까지 읽을 수 있음 (원본사용)
overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)

# overlay 이미지를 동영상에 띄우는 함수
'''
함수의 역할:
overlay 이미지를 얼굴의 중심인 center_x, center_y를 중심으로 놓고
overlay 사이즈만큼 Resize해서 원본 이미지에 넣어줌 (동영상의 얼굴 크키게 맞게 Resize)

이 결과를 result라는 변수에 저장
'''
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try :
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:  # background 이미지의 채널이 3일 경우 (다색일 경우)
            # 색상공간변환함수(cv2.cvtColor)
            # BGR2BGRA : BGR채널 이미지를 BGRA(Blue, Green, Red, Alpha) 이미지로 변경
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size) # 지정한 overlay_size로 변환

        # 채널분리함수(cv2.split): 영상이나 이미지의 색상 공간의 채널을 분리
        b, g, r, a = cv2.split(img_to_overlay_t)

        # cv2.medianBlur
        # 관심화소 주변으로 지정한 커널크기(5x5)내의 픽셀을 크기순으로 정렬한 후 중간값을 뽑아서 픽셀값으로 사용
        # "무작위 노이즈를 제거"하는데 효과적
        # 엣지가 있는 이미지의 경우, 결과 이미지에서 엣지가 사라질 수도 있음
        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        # ROI ( Region of Image)
        # 오버레이 이미지의 영역에 여유공간을 줘서 background 이미지를 덮을 수 있게 함
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

        # 비트연산으로 오버레이 이미지가 background 이미지 위에 올라오도록 설정
        # background 이미지
        # 적용영역(mask) = 오버레이 이미지의 알파채널이 아닌 영역(=background 이미지의 영역)을 보이게!
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        # 오버레이 이미지
        # 적용영역(mask) = 오버레이 이미지의 알파채널이 보이게!
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        # background 이미지와 오버레이 이미지 합성
        bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 3 channels
        # background 이미지를 4채널에서 3채널로 다시 변환
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        return bg_img

    except Exception:
        return background_img

# face_roi = []
# face_sizes = []

# 비디오가 끝날 때까지 계속 프레임단위로 읽기
while True:
    ret, img = cap.read()
    # 만약 프레임이 없다면, 프로그램 종료
    if not ret:
        break

    # 동영상 사이즈를 scaler 비율만큼 조절 (Resize)
    # cv2.resize의 변수는 이미지의 사이즈이므로 반드시 int로 가져와야 함
    # print(img.shape)  # 높이(height), 너비(width), 채널 (2160, 4096, 3)
    # cv2.resize(해당이미지, (너비, 높이))
    img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    # print(img.shape)  # (432, 819, 3)

    # 원본 이미지 저장
    ori = img.copy()

    # 얼굴인식
    faces = detector(img)

    # 얼굴이 없을 경우
    if len(faces) == 0:
        print('no faces!')
        continue

    else:
        face = faces[0]      # 여러 얼굴이 나오기 때문에, 찾은 모든 얼굴 중 첫번째 얼굴만 가져오기

    # print(len(faces))

    # # 여러 얼굴이 나오기 때문에, 찾은 모든 얼굴 중 첫번째 얼굴만 가져오기
    # face = faces[0]

    ### full version ###
    # if len(face_roi) == 0:
    #     faces = detector(img, 1)
    # else:
    #     roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    #     # cv2.imshow('roi', roi_img)
    #     faces = detector(roi_img)
    #
    # # 얼굴이 없을 경우
    # if len(faces) == 0:
    #     print('no faces!')

    # 얼굴 특징점 추출 (img의 face 영역 안의 얼굴 특징점 찾기)
    dlib_shape = predictor(img, face)
    # 연산을 쉽게 하기 위해, dlib 객체를 numpy 객체로 변환
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    ### full version ###
    # for face in faces:
    #     if len(face_roi) == 0:
    #         # img의 face 영역 안의 얼굴 특징점 찾기
    #         dlib_shape = predictor(img, face)
    #         # 연산을 쉽게 하기 위해, dlib 객체를 numpy 객체로 변환
    #         shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    #     else:
    #         dlib_shape = predictor(roi_img, face)
    #         shape_2d = np.array([[p.x + face_roi[2], p.y+face_roi[0]] for p in dlib_shape.parts()])
    #
    #     for s in shape_2d:
    #         cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # compute center and boundaries of face
    top_left = np.min(shape_2d, axis=0)                            # 얼굴의 좌상단
    bottom_right = np.max(shape_2d, axis=0)                        # 얼굴의 우하단
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)  # 얼굴의 중심

    # 얼굴 사이즈 구하기
    # 얼굴 사이즈 = 우하단 좌표 - 좌상단 좌표
    face_size = max(bottom_right - top_left)
    # overlay 이미지가 동영상의 얼굴사이즈에 비해 작기 때문에, 1.2을 곱해서 키워줌
    mean_face_size = int(face_size * 1.2)

    ### full version ###
    # face_size = max(bottom_right-top_left)
    # face_sizes.append(face_size)
    # if len(face_sizes) > 10:
    #     del face_size[0]
    # # 얼굴크기 계산값 10개의 평균을 구하고
    # # overlay 이미지가 동영상의 얼굴사이즈에 비해 작기 때문에, 1.2을 곱해서 키워줌
    # mean_face_size = int(np.mean(face_sizes) * 1.2)

    # # compute face roi
    # face_roi = np.array([int(top_left[1] - face_size / 2), int(bottom_right[1] + face_size / 2),
    #                     int(top_left[0] - face_size / 2), int(bottom_right[0] + face_size / 2)])
    # face_roi = np.clip(face_roi, 0, 10000)

    # overlay_transparent 함수의 결과값 저장
    # overlay 이미지를 동영상 얼굴 위치에서 약간 왼쪽(center_x-3), 위로(center_y-25) 옮겨주기
    result = overlay_transparent(ori, overlay, center_x-3, center_y-25, overlay_size=(face_size, face_size))

    # 얼굴 인식이 잘 됐는지 시각화해서 확인
    # 얼굴 영영만 네모로 그리기 (흰색)
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2 =(face.right(), face.bottom()), color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # cv2.circle 메소드를 사용해서 얼굴 특징점 68개 그리기 (흰색 점)
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴의 좌상단을 파란색 점으로 그리기
    cv2.circle(img, center=tuple(top_left), radius=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
    # 얼굴의 우하단을 파란색 점으로 그리기
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
    # 얼굴의 중심을 빨간색 점으로 그리기
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)

    # 이미지 띄우기
    cv2.imshow('original', ori)   # 원본
    cv2.imshow('img', img)        # 얼굴 영역만 네모표시 + 얼굴의 좌상단&우하단 파란색 점 + 얼굴중심 빨간색 점
    cv2.imshow('result', result)  # overlay 이미지 씌운 것

    # 1 mili-second만큼 대기 (동영상이 제대로 보이게 하기위함)
    if cv2.waitKey(1) == ord('q'):   # q가 입력될 때 프로그램 종료
        break

    # 결과 이미지 저장
    cv2.imwrite("output/original_1.jpg", ori[:])
    cv2.imwrite("output/img_1.jpg", img[:])
    cv2.imwrite("output/result_1.jpg", result[:])

cap.release()
cv2.destroyWindow()