import cv2

cap = cv2.VideoCapture('http://172.16.13.151:4747/video')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('DroidCam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
