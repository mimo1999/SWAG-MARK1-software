import io
import socket
import struct
import sys
import time
from pathlib import Path
import picamera

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

print("about to connect")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.bind((config.PI_HOST, config.PI_STREAM_PORT))
print("got socket")
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((config.LAPTOP_HOST, config.LAPTOP_STREAM_PORT))
print("finish connection")
connection = client_socket.makefile('wb')


try:
    with picamera.PiCamera() as camera:
        camera.resolution = (420,240)
        camera.framerate = 10
        camera.rotation=180
        #time.sleep(2)
        start = time.time()
        stream = io.BytesIO()
        instruct = 0

        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            stream.seek(0)
            stream.truncate()
    connection.write(struct.pack('<L', 0))
except socket.error as e:
    print(e)
finally:
    connection.close()
    client_socket.close()
    print('connection closed')

