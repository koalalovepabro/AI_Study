import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('samples/face.mp4')

# 분류기 load
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # scalefactor = 1.3 (1.05 is the best. 숫자가 커질수록 정확도는 떨어지지만 속도가 빠름)
                                                         # minNeighbors = 5 ( 3~6 is the best.최소 몇개의 얼굴을 인식할건지)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5) # red
        roi_gray = gray[ y:y+w, x:x+w ]
        roi_color = frame[ y:y+h, x:x+w ]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,255,0), 5)  # green

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    # 결과 이미지 저장
    cv2.imwrite("output/frame_2.jpg", frame[:])

cap.release()
cv2.destroyAllWindows()
