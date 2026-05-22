# SWAG MARK1 — Autonomous Self-Driving Car

A Raspberry Pi-based autonomous car that uses a camera and a CNN to navigate road-like environments. A human driver first demonstrates the driving behavior; the system records labeled frames, trains a convolutional neural network, and the car then drives itself using live inference.

---

## How It Works

```
[PiCamera] → frame → image pipeline → CNN inference → GPIO motor commands
                          │
               grayscale → crop bottom half
               → Gaussian blur → Laplacian edge detection
               → erode / dilate / morphological close
               → flatten to 1×50400 vector
               → lane_cnn.tflite (embedded Reshape → Resize 64×64 → Rescale ÷255)
```

The network is a 4-block custom CNN (~455 K parameters) trained with 5× data augmentation (horizontal flip with label swap, random shift, brightness jitter, contrast jitter). The three output classes are **forward**, **left**, and **right**, which map directly to GPIO pin states on the motor and steering drivers. The model is exported as a TFLite file for efficient inference on the Pi.

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
models/       Trained model files (.tflite, .keras, .xml)
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

Update IP addresses in `config.py` to match your network before running anything.

---

## Usage

### 1. Collect training data

Run the collection server on your **laptop** and the streaming client on the **Pi** simultaneously.

```bash
# Laptop — starts the collection server and pygame window
python training/collect_training.py --laplace

# Raspberry Pi — streams camera frames to the laptop
python car/stream_client.py
```

Use the arrow keys in the pygame window to drive the car. Each keypress labels the current frame and sends an HTTP command to the car via `car/controller.py`. Press **Escape** to stop and save the dataset to `data/`.

> **Always use `--laplace`** when collecting data for the CNN. It applies the identical preprocessing pipeline as the Pi (Gaussian blur → Laplacian → erode/dilate/close), ensuring training and inference see the same feature maps. Each session is saved as a timestamped `.npz` file so no data is overwritten.

### 2. Train the model

```bash
# Run from the training/ directory
python training/cnn_training.py
```

Loads all `.npz` files from `data/`, applies 5× augmentation, trains the CNN for up to 50 epochs with early stopping, and saves:
- `models/lane_cnn.keras` — full Keras model
- `models/lane_cnn.tflite` — TFLite model for Pi deployment

To compare all models in `models/` against the full dataset:
```bash
python training/benchmark_models.py
```

### 3. Deploy and drive autonomously

Copy `models/lane_cnn.tflite` to the Raspberry Pi, then:

```bash
python car/pi_driver_on_pi.py
```

The car will start driving based on live camera input.

### 4. Remote control (optional)

```bash
python car/controller.py
```

Exposes a REST API on port 5000. A watchdog stops the car automatically if `/ping` is not received within 1 second.

| Endpoint | Action |
|---|---|
| `GET /forward` | Move forward |
| `GET /reverse` | Reverse |
| `GET /left` | Turn left |
| `GET /right` | Turn right |
| `GET /pause` | Pause motors |
| `GET /stop` | Stop and cut power |
| `GET /ping` | Watchdog keepalive |

---

## Model Performance

| Model | Accuracy | Notes |
|---|---|---|
| `lane_cnn_best.keras` | **78.4%** | Best CNN checkpoint |
| `lane_cnn.tflite` | 76.5% | Deployed on Pi |
| `ann_best.xml` | 57.1% | Baseline ANN |

---

## Configuration

All network IPs and ports are centralised in `config.py` at the repo root. Update `PI_HOST` and `LAPTOP_HOST` to match your network before running.

---

## Team

Built by **Team SWAG** as part of the SwayamChalitGadi programme.
Website: [teamswag.in](https://www.teamswag.in)
