# Snow Camera
detect 68 face landmarks in Real-time

---
### Goal
실시간 또는 영상에서 사람의 얼굴을 인식해서 라이언 얼굴 띄우기

### Dependency

1. Python 3
2. [OpenCV (cv2)](https://076923.github.io/posts/Python-opencv-1/) : 이미지 처리
3. [Dlib](http://blog.dlib.net/) : 얼굴 인식
4. numpy : 행렬 연산

### Model
[68 face landmark model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)

### Data
인물 동영상
([무료 동영상 다운로드](https://www.pexels.com/ko-kr/search/videos/face/))

### Run & Result
- 얼굴영역 바운딩박스 + 68개 landmark (얼굴 1개 인식) => [`img_1.jpg`](https://github.com/koalalovepabro/KaggleStudy/blob/master/TP_00_SnowCamera/output/img_1.jpg)
- 라이언 얼굴 입히기 (얼굴 1개 인식) => [`result_1.jpg`](https://github.com/koalalovepabro/KaggleStudy/blob/master/TP_00_SnowCamera/output/result_1.jpg)
    ```
    ver01.py
    ```
- 분류기 사용 (cv2.CascadeClassifier)
- 실시간으로 웹캠의 내 얼굴 영역, 눈 영역을 바운딩박스로 표시 => [`frame_2.jpg`]
    ```
    ver02.py
    ```
- 분류기 사용 (cv2.CascadeClassifier)
- 얼굴영역 바운딩박스 + 68개 landmark (얼굴 3개 인식) => `img_3.jpg`
- 라이언 얼굴 입히기 (얼굴 1개 인식) => `result_3.jpg`
    ```
    ver03.py
    ```
- 분류기 사용안함
- 얼굴영역 바운딩박스 + 68개 landmark 有 (얼굴 3개 인식) => `img_4.jpg`
- 라이언 얼굴 입히기 (얼굴 1개만 표시되는 듯...?) => `result_4.jpg`

    ```
    ver04.py
    ```
- 분류기 사용안함
- 얼굴영역 바운딩박스 + 68개 landmark 有 + **코 최상단과 턱 최하단을 선으로 그리기** (얼굴 3개 인식) => `img_5.jpg`
- 라이언 얼굴 입히기 + **얼굴 각도에 따라 회전** (얼굴 1개만 표시되는 듯...?) => `result_5.jpg`

    ```
    ver05.py
    ```
  
### Study
1. [비트연산](https://copycoding.tistory.com/156)  
  함수 `overlay_transparent` 中  
    ```python
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))  
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
    ```
2. [Dlib 함수](http://dlib.net/python/index.html)

### Level up
1. 코드실행시 동영상이 끝까지 재생되지 않고, 멈추는 이유?  
   - 동영상에서 얼굴 영역이 사라졌기때문  
   - 얼굴 영역이 완전하게 포함되는 영상으로 바꿔서 테스트해보면 끝까지 재생됨  
   - 얼굴 영역이 사라졌을때 멈추지 않고 'no faces!'라고 알려주기만 하는 방법  
     ```python
     if len(faces) == 0:
         print('no faces!')  
         continue
     ```   

2. 얼굴의 움직임에 따라 overlay 이미지의 기울기를 움직이게 하려면?
    ```python
    top = shape_2d[27, :]  # 코 최상단 x, y 좌표
    down = shape_2d[8, :]  # 턱 최하단 x, y 좌표
            
    # 코 최상단에서 턱 최하단까지 선으로 이어 그리기
    img = cv2.line(img, pt1=top, pt2=down, color=(0, 255, 0), thickness=2)
            
    fc_slope = (down[1] - top[1]) / (down[0] - top[0])
            
    angle_rad = np.arctan(fc_slope)  # x의 arctan값 구하기 (radian값)
    angle = np.degrees(angle_rad)    # radian값 -> degree값 변환

    if angle < 0:              # 시계반대방향 회전은 (+) , 시계방향 회전은 (-)가 되도록
        angle = -(90 + angle)  # (-) 로 만들어 시계방향 회전
    else:
        angle = 90 - angle     # (+) 로 만들어 시계반대방향 회전

    # 라이언에 적용하기
    overlay_copy = imutils.rotate(overlay, angle)
    ```

3. 동영상의 얼굴이 여러개일때의 적용방법?  
   - for문으로 face 정의  
     ```python
     for face in faces:
         ...
     ```  
   - 바운딩박스와 landmark는 동시적용 OK (`ver04.py`)  
   - <i>layover 이미지를 동시에 띄우는건 시도 中</i>

4. <i>결과물을 gif 움짤로 저장하는 방법?</i>

### Reference
1. [`빵형의 개발도상국`님의 유튜브 영상](https://www.youtube.com/watch?v=tpWVyJqehG4&t=2s)
2. [`kairess`님의 github](https://github.com/kairess/face_detector)
