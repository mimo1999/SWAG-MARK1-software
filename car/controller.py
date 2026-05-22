import os
import time
import threading
import logging
import logging.config
from pathlib import Path
from flask import Flask, request
from car import forward, left, right, pause, rev, stop

# --- Logging ---
_log_cfg = Path(__file__).resolve().parent.parent / 'logging.ini'
if _log_cfg.exists():
    logging.config.fileConfig(_log_cfg)
else:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Watchdog ---
# Activates after the first /ping is received.
# If no ping arrives within WATCHDOG_TIMEOUT seconds, the car is stopped.
WATCHDOG_TIMEOUT = 1.0
_last_ping = None
_watchdog_lock = threading.Lock()


def _watchdog():
    while True:
        time.sleep(0.1)
        with _watchdog_lock:
            if _last_ping is not None and time.time() - _last_ping > WATCHDOG_TIMEOUT:
                logger.warning("Watchdog: no ping received — stopping car")
                stop()


_watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
_watchdog_thread.start()


@app.route('/ping', methods=['GET'])
def ping():
    global _last_ping
    with _watchdog_lock:
        _last_ping = time.time()
    return 'pong'


# --- Motor routes ---
def _speed():
    """Read optional ?speed= query param, clamped to 0-100."""
    try:
        return max(0, min(100, int(request.args.get('speed', 80))))
    except (TypeError, ValueError):
        return 80


@app.route('/forward', methods=['GET'])
def fwd():
    speed = _speed()
    forward(speed)
    logger.info('forward speed=%d', speed)
    return f"Forward speed={speed}"


@app.route('/reverse', methods=['GET'])
def reve():
    rev()
    logger.info('reverse')
    return "Reverse"


@app.route('/left', methods=['GET'])
def left_turn():
    speed = _speed()
    left(speed)
    logger.info('left speed=%d', speed)
    return f"Turn Left speed={speed}"


@app.route('/right', methods=['GET'])
def right_turn():
    speed = _speed()
    right(speed)
    logger.info('right speed=%d', speed)
    return f"Turn Right speed={speed}"


@app.route('/pause', methods=['GET'])
def pause_mot():
    pause()
    logger.info('pause')
    return "Pause"


@app.route('/stop', methods=['GET'])
def end():
    stop()
    logger.info('stop')
    return "Stop"


@app.route('/leftst', methods=['GET'])
def left_stop():
    pause()
    return "Left Stop"


@app.route('/rightst', methods=['GET'])
def right_stop():
    pause()
    return "Right Stop"


@app.route('/forwardst', methods=['GET'])
def forward_stop():
    pause()
    return "Forward Stop"


@app.route('/reversest', methods=['GET'])
def reverse_stop():
    pause()
    return "Reverse Stop"


if __name__ == "__main__":
    app.run(debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
            port=5000, host='0.0.0.0')
