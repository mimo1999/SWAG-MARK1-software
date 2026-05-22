import io
import socket
import struct
import sys
import time
import logging
import logging.config
from pathlib import Path
import picamera
import cv2
import numpy as np
import RPi.GPIO as GPIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# --- Logging ---
_log_cfg = Path(__file__).resolve().parent.parent / 'logging.ini'
if _log_cfg.exists():
    logging.config.fileConfig(_log_cfg)
else:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- TFLite runtime (lightweight on Pi; fall back to full TF if not installed) ---
try:
    import tflite_runtime.interpreter as tflite
    _TFLiteInterpreter = tflite.Interpreter
    logger.info("Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    _TFLiteInterpreter = tf.lite.Interpreter
    logger.info("tflite_runtime not found, using tensorflow.lite")

GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
GPIO.output(13, False)
GPIO.output(12, False)
GPIO.output(7, False)
GPIO.output(11, False)

logger.info("Connecting to %s:%d", config.LAPTOP_HOST, config.LAPTOP_STREAM_PORT)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.bind((config.PI_HOST, config.PI_STREAM_PORT))
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((config.LAPTOP_HOST, config.LAPTOP_STREAM_PORT))
logger.info("Connected")
connection = client_socket.makefile('wb')


class LaneCNN:
    """
    TFLite wrapper for lane_cnn inference.

    Expects a flat float32 array of shape (1, 50400) — the same raw pixel
    layout produced by the preprocessing pipeline below.  Rescaling and
    spatial reshape are embedded inside the TFLite model graph.
    """

    _MODEL_PATH = str(
        Path(__file__).resolve().parent.parent / 'models' / 'lane_cnn.tflite'
    )

    def __init__(self):
        self.interpreter = _TFLiteInterpreter(model_path=self._MODEL_PATH)
        self.interpreter.allocate_tensors()
        self._in_idx  = self.interpreter.get_input_details()[0]['index']
        self._out_idx = self.interpreter.get_output_details()[0]['index']
        logger.info("Loaded TFLite model: %s", self._MODEL_PATH)

    def predict(self, flat_frame: np.ndarray) -> int:
        """
        Run inference on one frame.

        Parameters
        ----------
        flat_frame : np.ndarray, shape (1, 50400), dtype float32

        Returns
        -------
        int  — 0 = forward, 1 = left, 2 = right
        """
        self.interpreter.set_tensor(self._in_idx, flat_frame)
        self.interpreter.invoke()
        return int(self.interpreter.get_tensor(self._out_idx).argmax(-1))


cnn = LaneCNN()

try:
    with picamera.PiCamera() as camera:
        camera.resolution = (420, 240)
        camera.framerate = 10
        camera.rotation = 180
        time.sleep(2)
        stream = io.BytesIO()
        GPIO.output(7, False)
        GPIO.output(11, False)

        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            stream_value = stream.read()
            connection.write(stream_value)
            stream.seek(0)
            stream.truncate()

            image = cv2.imdecode(np.frombuffer(stream_value, dtype=np.uint8), -1)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = gray_image[120:240, :]
            gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            gray_image = cv2.Laplacian(gray_image, cv2.CV_64F)
            gray_image = cv2.erode(gray_image, kernel=(3, 3), iterations=3)
            gray_image = cv2.erode(gray_image, kernel=(4, 4))
            gray_image = cv2.dilate(gray_image, kernel=(2, 2), iterations=2)
            gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel=(3, 3))

            temp_image_array = gray_image.reshape(1, 50400).astype(np.float32)
            prediction = cnn.predict(temp_image_array)

            if prediction == 0:
                GPIO.output(13, True)
                GPIO.output(12, False)
                GPIO.output(7, False)
                GPIO.output(11, False)
                logger.debug('forward')
            elif prediction == 1:
                GPIO.output(13, True)
                GPIO.output(12, False)
                GPIO.output(7, False)
                GPIO.output(11, True)
                logger.debug('left')
                time.sleep(0.06)
            elif prediction == 2:
                GPIO.output(13, True)
                GPIO.output(12, False)
                GPIO.output(7, True)
                GPIO.output(11, False)
                logger.debug('right')
                time.sleep(0.06)

            time.sleep(0.06)
            GPIO.output(7, False)
            GPIO.output(11, False)
            GPIO.output(13, False)
            GPIO.output(12, False)
            time.sleep(0.16)

    connection.write(struct.pack('<L', 0))

except socket.error as e:
    logger.error("Socket error: %s", e)
finally:
    connection.close()
    client_socket.close()
    GPIO.cleanup()
    logger.info("Connection closed, GPIO cleaned up")
