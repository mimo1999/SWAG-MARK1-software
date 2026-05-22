import sys
import time
import threading
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pygame
from pygame.locals import *
import socket
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Run with --laplace to apply Sobel edge preprocessing before saving frames.
# The laplace mode also captures frames on key-up (stop event) in addition to key-down.
USE_LAPLACE = '--laplace' in sys.argv


def waiter():
    time.sleep(0.05)


def waiter_for():
    time.sleep(0.15)


def preprocess(frame):
    """
    Produce a flat (1, 50400) float32 array from a raw BGR frame.

    Without --laplace : bottom-half grayscale only (raw mode).
    With    --laplace : applies the same pipeline as pi_driver_on_pi.py so
                        that training and inference see identical feature maps:
                        GaussianBlur → Laplacian → erode × 2 → dilate → CLOSE.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    half = gray[120:240, :]
    if USE_LAPLACE:
        half = cv2.GaussianBlur(half, (5, 5), 0)
        half = cv2.Laplacian(half, cv2.CV_64F)
        half = cv2.erode(half, kernel=(3, 3), iterations=3)
        half = cv2.erode(half, kernel=(4, 4))
        half = cv2.dilate(half, kernel=(2, 2), iterations=2)
        half = cv2.morphologyEx(half, cv2.MORPH_CLOSE, kernel=(3, 3))
    return half.reshape(1, 50400).astype(np.float32)


class CollectTrainingData(object):
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((config.LAPTOP_HOST, config.COLLECT_PORT))
        self.server_socket.listen(0)

        self.connection = self.server_socket.accept()[0].makefile('rb')
        self.isReceiving = True

        self.k = np.zeros((3, 3), 'float')
        for i in range(3):
            self.k[i, i] = 1

        pygame.init()
        pygame.display.set_mode((420, 240))
        pygame.key.set_repeat(1, 100)
        self.collect_imgdata()

    def collect_imgdata(self):
        saved_frame = 0
        total_frame = 0

        print('start collecting images.... (laplace=%s)' % USE_LAPLACE)
        image_array = np.zeros((1, 50400))
        label_array = np.zeros((1, 3), 'float')

        thread = threading.Thread(target=waiter)
        http = urllib3.PoolManager()

        def save_frame(flat_image, label):
            nonlocal image_array, label_array, saved_frame
            image_array = np.vstack((image_array, flat_image))
            label_array = np.vstack((label_array, label))
            saved_frame += 1

        try:
            stream_bytes = b''

            while self.isReceiving:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')

                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), -1)
                    flat_image = preprocess(frame)
                    cv2.imshow('view', frame)
                    total_frame += 1

                    for event in pygame.event.get():
                        if event.type == KEYDOWN and not thread.is_alive():
                            keypress = event.key

                            if keypress == pygame.K_SPACE:
                                print('stop')
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/stop')

                            elif keypress == pygame.K_DOWN:
                                print('rev')
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/reverse')

                            elif keypress == pygame.K_ESCAPE:
                                print('exit')
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/stop')
                                self.isReceiving = False
                                break

                            elif keypress == pygame.K_LEFT:
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/left')
                                thread = threading.Thread(target=waiter)
                                thread.start()
                                print('left')
                                save_frame(flat_image, self.k[1])

                            elif keypress == pygame.K_RIGHT:
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/right')
                                thread = threading.Thread(target=waiter)
                                thread.start()
                                print('right')
                                save_frame(flat_image, self.k[2])

                            elif keypress == pygame.K_UP:
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/forward')
                                thread = threading.Thread(target=waiter_for)
                                thread.start()
                                print('forward')
                                save_frame(flat_image, self.k[0])

                        elif USE_LAPLACE and event.type == KEYUP:
                            keypress = event.key

                            if keypress == pygame.K_LEFT:
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/leftst')
                                save_frame(flat_image, self.k[1])

                            elif keypress == pygame.K_RIGHT:
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/rightst')
                                save_frame(flat_image, self.k[2])

                            elif keypress == pygame.K_UP:
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/forwardst')
                                save_frame(flat_image, self.k[0])

                            elif keypress == pygame.K_DOWN:
                                http.request('GET', f'http://{config.PI_HOST}:{config.CONTROLLER_PORT}/reversest')

            train = image_array[1:, :]
            train_labels = label_array[1:, :]
            print('Training data shape', train.shape)
            print('Training label shape', train_labels.shape)
            print('Total frame:', total_frame)
            print('Saved frame', saved_frame)
            print('Dropped frame', total_frame - saved_frame)

        finally:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode  = 'laplace' if USE_LAPLACE else 'raw'
            out_path = Path(__file__).resolve().parent.parent / 'data' / f'training_{mode}_{stamp}.npz'
            np.savez(str(out_path), train=train, train_labels=train_labels)
            print('Saved:', out_path)
            self.connection.close()
            self.server_socket.close()
            print('connection closed')


if __name__ == '__main__':
    CollectTrainingData()
