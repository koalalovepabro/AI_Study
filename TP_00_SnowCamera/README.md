# Snow Camera
---
### Goal
사람 얼굴을 인식해서 라이언 얼굴 띄우기

### Library
1. [OpenCV (cv2)](https://076923.github.io/posts/Python-opencv-1/) : 이미지 처리
2. [Dlib](http://blog.dlib.net/) : 얼굴 인식
3. numpy : 행렬 연산

### Data
인물 동영상
([무료 동영상 다운로드](https://www.pexels.com/ko-kr/search/videos/face/))

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
