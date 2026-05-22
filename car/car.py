import time
import RPi.GPIO as GPIO

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

p = GPIO.PWM(13, 50)


def forward(speed=80):
    """Drive forward. speed: PWM duty cycle 0-100."""
    GPIO.output(13, True)
    p.start(40)
    p.ChangeDutyCycle(speed)
    GPIO.output(12, False)
    GPIO.output(7, False)
    GPIO.output(11, False)


def left(speed=55):
    """Steer left for ~5 ms. speed: PWM duty cycle 0-100."""
    start = time.time() * 1000
    while time.time() * 1000 <= start + 5:
        GPIO.output(13, True)
        p.start(40)
        p.ChangeDutyCycle(speed)
        GPIO.output(12, False)
        GPIO.output(7, False)
        GPIO.output(11, True)


def right(speed=55):
    """Steer right for ~5 ms. speed: PWM duty cycle 0-100."""
    start = time.time() * 1000
    while time.time() * 1000 <= start + 5:
        GPIO.output(13, True)
        p.start(40)
        p.ChangeDutyCycle(speed)
        GPIO.output(12, False)
        GPIO.output(7, True)
        GPIO.output(11, False)


def pause():
    GPIO.output(13, False)
    p.ChangeDutyCycle(0)
    GPIO.output(12, False)
    GPIO.output(7, False)
    GPIO.output(11, False)


def rev():
    p.ChangeDutyCycle(0)
    GPIO.output(12, True)
    GPIO.output(7, False)
    GPIO.output(11, False)


def stop():
    GPIO.output(13, False)
    p.ChangeDutyCycle(0)
    p.stop()
    GPIO.output(12, False)
    GPIO.output(7, False)
    GPIO.output(11, False)
