# Annoying Orange
AI face detection and composition in Real-time

---
### Goal
OpenCV의 Seamless clone(Poisson blending)을 사용해서  
내 얼굴을 합성시킨 어노잉 오렌지 만들기 (실시간)

### Library
1. Python 3
2. [OpenCV (cv2)](https://076923.github.io/posts/Python-opencv-1/) : 이미지 합성
3. [Dlib](http://blog.dlib.net/) : 얼굴 인식
4. imutils : resize
5. numpy : 행렬 연산

### Model
[68 face landmark model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)

### Data
오렌지 이미지

### Study
1. face landmark  
![image](https://blog.kakaocdn.net/dn/baNXvY/btqChGYfEU4/l59Bgkbdir5CQ4wdDqZIw0/img.png)

### Reference
1. [`빵형의 개발도상국`님의 유튜브 영상](https://www.youtube.com/watch?v=9VYUXchrMcM)
2. [`kairess`님의 github](https://github.com/kairess/annoying-orange-face)
