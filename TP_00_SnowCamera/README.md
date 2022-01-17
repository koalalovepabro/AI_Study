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
- 얼굴영역 바운딩박스 + 68개 landmark (얼굴 1개 인식) => `img_1.jpg`
- 라이언 얼굴 입히기 (얼굴 1개 인식) => `result_1.jpg`
    ```
    ver01.py
    ```
- 분류기 사용 (cv2.CascadeClassifier)
- 실시간으로 웹캠의 내 얼굴 영역, 눈 영역을 바운딩박스로 표시 => `frame_2.jpg`
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
  -> 동영상에서 얼굴 영역이 사라졌기때문  
  -> 얼굴 영역이 완전하게 포함되는 영상으로 바꿔서 테스트해보면 끝까지 재생됨  
  -> 얼굴 영역이 사라졌을때 멈추지 않고 'no faces!'라고 알려주기만 하는 방법  
    ```python
    if len(faces) == 0:
        print('no faces!')  
        continue
    ```   

2. <i>얼굴의 움직임에 따라 overlay 이미지의 기울기를 움직이게 하려면?</i>


3. 동영상의 얼굴이 여러개일때의 적용방법?  
  -> for문으로 face 정의
    ```python
    for face in faces:
        ...
    ```
4. <i>결과물을 gif 움짤로 저장하는 방법?</i>

### Reference
1. [`빵형의 개발도상국`님의 유튜브 영상](https://www.youtube.com/watch?v=tpWVyJqehG4&t=2s)
2. [`kairess`님의 github](https://github.com/kairess/face_detector)
