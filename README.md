# SWAG MARK1 — Autonomous Self-Driving Car

A Raspberry Pi-based autonomous car that uses a camera and a neural network to navigate road-like environments. A human driver first demonstrates the driving behavior; the system records labeled frames, trains an ANN, and the car then drives itself using live inference.

---

## How It Works

```
[PiCamera] → frame → image pipeline → ANN inference → GPIO motor commands
                          │
               grayscale → crop bottom half
               → Gaussian blur → Laplacian edge detection
               → erode / dilate / morphological close
               → flatten to 1×50400 vector
```

The network is a 3-class OpenCV `ANN_MLP` (50400 → 32 → 3) trained via backpropagation. The three output classes are **forward**, **left**, and **right**, which map directly to GPIO pin states on the motor and steering drivers.

A separate remote-control Flask API (`car/controller.py`) lets you drive the car manually over HTTP during data collection.

---

## Hardware

- Raspberry Pi (3B+ or 4 recommended)
- PiCamera module
- DC motor + motor driver (PWM on GPIO pin 13)
- Servo steering (GPIO pins 7, 11)
- Motor direction/brake (GPIO pin 12)
- Wi-Fi on both the Pi and the laptop used for training data collection

---

## Project Structure

```
car/          Scripts deployed to and run on the Raspberry Pi
training/     ML pipeline — data collection, training, evaluation (runs on laptop)
models/       Trained OpenCV ANN_MLP model files (.xml)
data/         Labeled training datasets (.npz)
docs/         Project documentation and design report
```

---

## Setup

**On the Raspberry Pi:**
```bash
pip install -r car/requirements.txt
```

**On the laptop (training pipeline):**
```bash
pip install -r training/requirements.txt
```

---

## Usage

### 1. Collect training data

Run the collection server on your **laptop** and the streaming client on the **Pi** simultaneously.

Update the IP addresses in `training/collect_training.py` and `car/stream_client.py` to match your network before running.

```bash
# Laptop — starts the collection server and pygame window
python training/collect_training.py

# Raspberry Pi — streams camera frames to the laptop
python car/stream_client.py
```

Use the arrow keys in the pygame window to drive the car. Each keypress labels the current frame and sends an HTTP command to the car via `car/controller.py`. Press **Escape** to stop and save the dataset to `data/`.

`collect_training_laplace_filter.py` is an alternative collector that applies a Sobel filter before labeling, which can improve training quality on high-contrast tracks.

### 2. Train the model

```bash
# Run from the repo root
python training/ann_training.py
```

Loads all `.npz` files from `data/`, trains the ANN, prints validation accuracy, and saves the model to `models/ann.xml`.

### 3. Deploy and drive autonomously

Copy `models/ann.xml` to the Raspberry Pi, then:

```bash
python car/pi_driver_on_pi.py
```

The car will start driving based on live camera input. Update the server IP in the script before running.

### 4. Remote control (optional)

```bash
python car/controller.py
```

Exposes a REST API on port 5000:

| Endpoint | Action |
|---|---|
| `GET /forward` | Move forward |
| `GET /reverse` | Reverse |
| `GET /left` | Turn left |
| `GET /right` | Turn right |
| `GET /pause` | Pause motors |
| `GET /stop` | Stop and cut power |

---

## Configuration

Network IPs and ports are hardcoded in:
- `car/pi_driver_on_pi.py` — bind address and server IP
- `car/stream_client.py` — bind address and server IP
- `training/collect_training.py` — listen address and car controller IP

Update these to match your network before running.

---

## Team

Built by **Team SWAG** as part of the SwayamChalitGadi programme.
Website: [teamswag.in](https://www.teamswag.in)
