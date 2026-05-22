import sys
import socket
import time
import math
import threading
import logging
import logging.config
from pathlib import Path
import cv2
import numpy as np
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.lane_utils import fit_polynomial, single_fit

# --- Logging ---
_log_cfg = Path(__file__).resolve().parent.parent / 'logging.ini'
if _log_cfg.exists():
    logging.config.fileConfig(_log_cfg)
else:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

font = cv2.FONT_HERSHEY_SIMPLEX
http = urllib3.PoolManager()
thread = threading.Thread(target=lambda: None)

HAARCASCADE_STOP_SIGN = 'haarcascade_stop_sign.xml'
FORWARD_SPEED = 70
TURN_SPEED = 50


def waiter():
    time.sleep(0.015)


def waiter_for():
    time.sleep(0.1)


def _ping_loop():
    """Send /ping to the car every 500 ms so the watchdog doesn't trigger."""
    while True:
        try:
            http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/ping',
                         timeout=0.4)
        except Exception:
            pass
        time.sleep(0.5)


class LaneDriver(object):
    def __init__(self):
        self.server_socket = socket.socket()
        self.server_socket.bind((config.LAPTOP_HOST, config.LAPTOP_STREAM_PORT))
        self.server_socket.listen(0)
        self.connection = self.server_socket.accept()[0].makefile('rb')
        self.isReceiving = True
        self.stop_cascade = cv2.CascadeClassifier(HAARCASCADE_STOP_SIGN)

        # Start watchdog ping thread
        ping_thread = threading.Thread(target=_ping_loop, daemon=True)
        ping_thread.start()

        self.drive()

    def drive(self):
        global thread
        logger.info('Starting lane-following autonomous drive...')
        prediction = 0

        try:
            stream_bytes = b''
            while self.isReceiving:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first == -1 or last == -1:
                    continue

                jpg = stream_bytes[first:last + 2]
                stream_bytes = stream_bytes[last + 2:]
                color_image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), -1)
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                img_org = color_image.copy()

                a, img = cv2.threshold(gray_image, 75, 255, cv2.THRESH_BINARY_INV)
                img = cv2.erode(img, kernel=(4, 4), iterations=4)
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=(5, 5))
                height, width = img.shape
                mask = np.zeros_like(img)
                polygon = np.array([[(0, height * 0.4), (width, height * 0.4),
                                     (width, height), (0, height)]], np.int32)
                cv2.fillPoly(mask, polygon, 255)
                masked_image = cv2.bitwise_and(img, mask)

                edge = 130
                double_hit = single_hit = 0
                left_x, left_y, right_x, right_y = [], [], [], []
                single_x, single_y = [], []

                while edge < 240:
                    temp = masked_image[edge:edge + 5]
                    contour, _ = cv2.findContours(temp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contours = [c for c in contour if 30 <= cv2.contourArea(c) < 140]
                    le = len(contours)
                    cent = []

                    if le == 0:
                        edge += 5
                        continue
                    elif le == 1:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        cent.append([x, y + edge])
                        cv2.circle(img_org, (x, y + edge), 5, (255, 255, 255), -1)
                        if double_hit < 8:
                            single_x.append(cent[0][0])
                            single_y.append(cent[0][1])
                            single_hit += 1
                    elif le == 2:
                        for cont in contours:
                            x, y, w, h = cv2.boundingRect(cont)
                            cent.append([x + int(w / 2), y + edge])
                            cv2.circle(img_org, (x + int(w / 2), y + edge), 5, (255, 255, 255), -1)
                        if cent[0][0] > cent[1][0]:
                            left_x.append(cent[0][0]); left_y.append(cent[0][1])
                            right_x.append(cent[1][0]); right_y.append(cent[1][1])
                        else:
                            left_x.append(cent[1][0]); left_y.append(cent[1][1])
                            right_x.append(cent[0][0]); right_y.append(cent[0][1])
                        double_hit += 1
                    else:
                        pts, abso = [], []
                        for cont in contours:
                            x, y, w, h = cv2.boundingRect(cont)
                            pts.append([x, y + edge])
                            abso.append(abs(x - 210))
                        ins = np.argsort(abso)
                        pts = np.array(pts)[ins]
                        cent = [pts[0], pts[1]]
                        if cent[0][0] > cent[1][0]:
                            left_x.append(cent[0][0]); left_y.append(cent[0][1])
                            right_x.append(cent[1][0]); right_y.append(cent[1][1])
                        else:
                            left_x.append(cent[1][0]); left_y.append(cent[1][1])
                            right_x.append(cent[0][0]); right_y.append(cent[0][1])
                        double_hit += 1
                    edge += 5

                if len(single_x) > 2 or len(left_x) > 2:
                    if double_hit > 4:
                        centre_x = (left_x[0] + right_x[0]) // 2
                        centre_y = (left_y[0] + right_y[0]) // 2
                        current = len(left_x) - 1
                        curr_centre_x = (left_x[current] + right_x[current]) // 2
                        curr_centre_y = (left_y[current] + right_y[current]) // 2
                        cv2.line(img_org, (centre_x, centre_y),
                                 (curr_centre_x, curr_centre_y), (255, 255, 255), 3)
                        b2 = float(abs(centre_x - curr_centre_x))
                        h2 = float(abs(centre_y - curr_centre_y))
                        AngleInRad = math.atan(h2 / b2) if b2 != 0.0 else 1.5708
                        degree_comp = abs(90 - AngleInRad * (180.0 / math.pi))
                        if degree_comp > 55:
                            prediction = 3
                        elif degree_comp > 40:
                            prediction = 1 if curr_centre_x > centre_x else 2
                        else:
                            prediction = 0
                        if centre_x > 390 or centre_x < 30:
                            prediction = 3
                    elif single_hit < 4:
                        prediction = 1 if np.mean(single_x) < 210 else 2
                    elif single_hit > 4:
                        current = len(single_x) - 1
                        curr_x = single_x[current]
                        first_x = single_x[0]
                        prediction = 2 if curr_x < first_x else (1 if curr_x > first_x else 0)
                    else:
                        prediction = 3
                else:
                    prediction = 3

                # Stop sign detection
                if len(self.stop_cascade.detectMultiScale(
                        color_image, scaleFactor=1.1, minNeighbors=5,
                        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)) > 0:
                    logger.info("Stop sign detected")
                    prediction = 4

                pre_text = "STOP"
                if not thread.is_alive():
                    if prediction == 0:
                        http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}'
                                           f'/forward?speed={FORWARD_SPEED}')
                        thread = threading.Thread(target=waiter_for)
                        thread.start()
                        pre_text = "FORWARD"
                        logger.debug('forward')
                    elif prediction == 1:
                        http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}'
                                           f'/left?speed={TURN_SPEED}')
                        thread = threading.Thread(target=waiter)
                        thread.start()
                        pre_text = "LEFT"
                        logger.debug('left')
                    elif prediction == 2:
                        http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}'
                                           f'/right?speed={TURN_SPEED}')
                        thread = threading.Thread(target=waiter)
                        thread.start()
                        pre_text = "RIGHT"
                        logger.debug('right')
                    else:
                        http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/stop')
                        logger.debug('stop/no lane')

                cv2.putText(img_org, pre_text, (340, 50), font, 1, (255, 0, 255), 5)
                cv2.imshow('view', img_org)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.isReceiving = False

        finally:
            self.connection.close()
            self.server_socket.close()
            cv2.destroyAllWindows()
            logger.info("Connection closed")


if __name__ == '__main__':
    LaneDriver()
