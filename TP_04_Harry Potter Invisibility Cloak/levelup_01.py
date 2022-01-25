import cv2
import numpy as np
import time, argparse
import mediapipe as mp

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
out = cv2.VideoWriter('output/output_levelup_01.mp4', fourcc,
                      20,                                          # 영상 FPS(Frame Per Second): 고정값으로 설정해야 에러 안남
                      (background.shape[1], background.shape[0]))  # 프레임크기는 background와 같은 사이즈로 설정
# 저장2. 웹캠상 이미지
out2 = cv2.VideoWriter('output/original_levelup_01.mp4', fourcc,
                       20,
                       (background.shape[1], background.shape[0]))

while (cap.isOpened()):
    ret, img = cap.read()  # 웹캠을 한 프레임씩 읽어오기
    if not ret:
        break

    ########################## [selfie segmentation으로 mask 생성]
    # MediaPipe 모델에 넣기 전에 전처리
    # BGR -> RGB (OpenCV는 BGR 컬러시스템, MediaPipe는 RGB 사용하기 때문)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    ##############################################################

    # Remove noise
    # Refining the mask corresponding to the detected red color
    # 사람(흰색영역)
    mask_cloak = cv2.morphologyEx(mask1, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
    mask_cloak = cv2.dilate(mask_cloak, kernel=np.ones((3, 3), np.uint8), iterations=1)

    # 사람영역을 제외시키기
    mask_bg = cv2.bitwise_not(mask_cloak)

    cv2.imshow('mask_cloak', mask_cloak)  # mask_cloack  (사람영역만 뽑아서 흰색으로 표시한 것)

    # Generate the final output
    res1 = cv2.bitwise_and(background, background, mask=mask_cloak)  # background에서 사람영역만큼만 뽑은 것
    res2 = cv2.bitwise_and(img, img, mask=mask_bg)                   # 현재 웹캠 이미지에서 사람영역을 제외한 영역

    # 2개 이미지를 합성하기 ( cv2.addWeighted )
    '''
    Parameters:	
    src1  – first input array.
    alpha – weight of the first array elements.
    src2  – second input array of the same size and channel number as src1.
    beta  – weight of the second array elements.
    dst   – output array that has the same size and number of channels as the input arrays.
    gamma – scalar added to each sum.
    dtype – optional depth of the output array. when both input arrays have the same depth,
            dtype can be set to -1, which will be equivalent to src1.depth().
    '''
    result = cv2.addWeighted(src1=res1, alpha=1, src2=res2, beta=1, gamma=0)

    cv2.imshow('res1', res1)      # res1   (background에서 사람영역만큼만 뽑은 것)
    cv2.imshow('res2', res2)      # res2   (웹캠상 이미지에서 사람영역 제외한 것)
    cv2.imshow('ori', img)        # 웹캠 원본 이미지
    cv2.imshow('result', result)  # result (투명인간 적용한 결과 이미지)

    out.write(result)  # 결과 이미지
    out2.write(img)    # 원본 이미지 (웹캠상 이미지)

    if cv2.waitKey(1) == ord('q'):
        break
    #
    # # 결과를 이미지로 저장
    # cv2.imwrite("output/original.jpg", img[:])               # 웹캠 원본 이미지
    # cv2.imwrite("output/mask_cloack.jpg", mask_cloak[:])     # mask_cloack  (사람영역만 뽑아서 흰색으로 표시한 것)
    # cv2.imwrite("output/res1.jpg", res1[:])                  # res1         (background에서 사람영역만큼만 뽑은 것)
    # cv2.imwrite("output/res2.jpg", res2[:])                  # res2         (웹캠상 이미지에서 사람영역 제외한 것)
    cv2.imwrite("output/result_levelup_01.jpg", result[:])   # result       (투명인간 적용한 결과 이미지)

out.release()
out2.release()
cap.release()





