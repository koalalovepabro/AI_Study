# Fxck You Filter
Mosaic the middle finger in video.

---
### Goal
영상에서 가운데 손가락 모자이크하기

### Run
- fxck you 제스처(label=11)의 학습 데이터셋 추가
    ```
    gather_dataset.py
    ```
- fxck you 제스처를 인식해서 모자이크처리
    ```
    fy_filter.py
    ```
- 손하트 제스처를 인식해서 텍스트와 바운딩박스 표시
    ```
    heart_filter.py
    ```

### Dependency
1. Python 3
2. OpenCV
3. MediaPipe

### Model
KNN(K-Nearest Neighbors)

### Data
1. 제스처 학습 데이터 (gesture_train.csv)
2. fxck you 제스처 데이터가 추가된 데이터셋 (gesture_train_fy.csv)
3. 손하트 제스처 데이터가 추가된 데이터셋 (gesture_train_heart.csv)

### Study
[사각형 그리기](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=pk3152&logNo=221442217481)
`cv2.rectanble`

![img_1.png](img_1.png)


### Level up
1. 손하트 제스처 인식  
   -> `gather_heart.py`를 실행시켜서 손하트 제스처 데이터셋 추가  
   -> `gesture_train_heart.csv`에 저장하고, 이 데이터로 학습하는 `heart_filter.py` 코드실행  
   -> 손하트 제스처 인식되는것을 확인 💗

### Reference
1. [`빵형의 개발도상국`님의 유튜브 영상](https://www.youtube.com/watch?v=tQeuPrX821w&list=PL-xmlFOn6TUJ9KjFo0VsM3BI9yrCxTnAz)
2. [`kairess`님의 github](https://github.com/kairess/Rock-Paper-Scissors-Machine)
