import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

url = f'http://{config.PI_HOST}:8080/video'
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
