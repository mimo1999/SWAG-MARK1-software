"""
Mid-scale custom CNN for lane classification.

Augmentation pipeline (applied only to training split):
  1. Horizontal flip  — left/right labels are swapped
  2. Random shift     — cv2.warpAffine with edge-replication fill
  3. Brightness jitter
  4. Contrast jitter

The model embeds Reshape → Resize → Rescale as its first three layers,
so its public interface is identical to the ANN: a flat float32 (50400,)
vector. The benchmark script therefore needs zero special-casing.

Outputs:
  ../models/lane_cnn.keras     — full Keras model (includes preprocessing)
  ../models/lane_cnn.tflite    — TFLite model for Pi deployment

Usage:
    python cnn_training.py
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load_data, train_val_split

IMG_H, IMG_W = 120, 420          # original image dimensions stored in npz
CNN_SIZE     = 64                 # resize target for CNN input
BATCH_SIZE   = 32
LABEL_NAMES  = ['forward', 'left', 'right']


# ------------------------------------------------------------------ #
# Augmentation helpers (operate on (H, W) float32 numpy arrays)
# ------------------------------------------------------------------ #

def hflip_with_label_swap(img, label_idx):
    """
    Horizontally flip the image.
    Left (1) ↔ Right (2) labels are swapped; Forward (0) is unchanged.
    """
    flipped = img[:, ::-1].copy()
    if label_idx == 1:
        label_idx = 2
    elif label_idx == 2:
        label_idx = 1
    return flipped, label_idx


def random_shift(img, max_h_frac=0.05, max_v_frac=0.03):
    """
    Translate the image by a random amount.
    Borders are filled by edge-replication (no wrap-around artefacts).
    """
    h, w = img.shape
    dx = int(np.random.uniform(-max_h_frac, max_h_frac) * w)
    dy = int(np.random.uniform(-max_v_frac, max_v_frac) * h)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def brightness_jitter(img, low=0.70, high=1.30):
    """Multiply all pixels by a random scalar."""
    return np.clip(img * np.random.uniform(low, high), 0, 255).astype(np.float32)


def contrast_jitter(img, low=0.70, high=1.30):
    """Scale pixel values around the image mean."""
    mean   = img.mean()
    factor = np.random.uniform(low, high)
    return np.clip(mean + factor * (img - mean), 0, 255).astype(np.float32)


def augment(X_flat, y):
    """
    Expand the training set ~5× by applying four augmentation passes:
      1. Horizontal flip with label swap
      2. Random shift
      3. Brightness jitter
      4. Contrast jitter

    Augmentation is applied AFTER train/val split so validation data
    remains clean (unaugmented originals only).
    """
    n = len(X_flat)
    imgs       = X_flat.reshape(n, IMG_H, IMG_W).astype(np.float32)
    label_idxs = y.argmax(-1)

    aug_X, aug_y = [X_flat.copy()], [y.copy()]

    # --- 1. Horizontal flip with label swap ----------------------------
    flip_imgs   = np.empty_like(imgs)
    flip_labels = np.zeros_like(y)
    for i, (img, li) in enumerate(zip(imgs, label_idxs)):
        fimg, fli        = hflip_with_label_swap(img, li)
        flip_imgs[i]     = fimg
        flip_labels[i, fli] = 1.0
    aug_X.append(flip_imgs.reshape(n, -1))
    aug_y.append(flip_labels)

    # --- 2. Random shift (original images, unchanged labels) -----------
    shifted = np.array([random_shift(img) for img in imgs], dtype=np.float32)
    aug_X.append(shifted.reshape(n, -1))
    aug_y.append(y.copy())

    # --- 3. Brightness jitter -----------------------------------------
    bright = np.array([brightness_jitter(img) for img in imgs], dtype=np.float32)
    aug_X.append(bright.reshape(n, -1))
    aug_y.append(y.copy())

    # --- 4. Contrast jitter -------------------------------------------
    contrast = np.array([contrast_jitter(img) for img in imgs], dtype=np.float32)
    aug_X.append(contrast.reshape(n, -1))
    aug_y.append(y.copy())

    X_aug = np.vstack(aug_X)
    y_aug = np.vstack(aug_y)

    # Shuffle so all augmentation variants are interleaved
    idx = np.random.default_rng(42).permutation(len(X_aug))
    return X_aug[idx].astype(np.float32), y_aug[idx].astype(np.float32)


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

def build_cnn():
    """
    4-block CNN (~455 K parameters).

    Input: flat float32 vector of length 50400 (raw pixel values 0-255).
    The first three layers perform Reshape → Resize(64×64) → Rescale([0,1])
    internally, so the model is a drop-in replacement for the ANN in
    both training and inference.
    """
    inputs = keras.Input(shape=(50400,), name='flat_input')

    # --- Internal preprocessing (embedded in model graph) -------------
    x = layers.Reshape((IMG_H, IMG_W, 1), name='reshape')(inputs)
    x = layers.Resizing(CNN_SIZE, CNN_SIZE, name='resize')(x)
    x = layers.Rescaling(1.0 / 255.0, name='rescale')(x)

    # --- Conv block 1 -------------------------------------------------
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)          # 32 × 32 × 32

    # --- Conv block 2 -------------------------------------------------
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)          # 16 × 16 × 64

    # --- Conv block 3 -------------------------------------------------
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)          # 8 × 8 × 128

    # --- Conv block 4 (no pooling — GAP collapses spatial dims) -------
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x) # 256

    # --- Classifier head ---------------------------------------------
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='lane_cnn')


# ------------------------------------------------------------------ #
# Evaluation helper
# ------------------------------------------------------------------ #

def evaluate_detail(model, X, y, split_name):
    y_pred = model.predict(X, verbose=0).argmax(-1)
    y_true = y.argmax(-1)
    acc    = np.mean(y_pred == y_true)
    print("\n  %s accuracy: %.2f%%" % (split_name, acc * 100))
    print("  %-10s  %8s  %8s  %8s" % ("class", "prec", "recall", "F1"))
    for i, name in enumerate(LABEL_NAMES):
        tp = np.sum((y_pred == i) & (y_true == i))
        fp = np.sum((y_pred == i) & (y_true != i))
        fn = np.sum((y_pred != i) & (y_true == i))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print("  %-10s  %7.1f%%  %7.1f%%  %7.1f%%" % (name, p*100, r*100, f1*100))
    return acc


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    # ---- Load and split BEFORE augmentation to prevent leakage ------
    print("=== Loading data ===")
    X_flat, y = load_data('../data/*.npz')
    X_train_raw, X_val_raw, y_train_raw, y_val = train_val_split(
        X_flat, y, val_ratio=0.2)
    print("Original — train: %d  val: %d" % (len(X_train_raw), len(X_val_raw)))

    # ---- Augment only the training split ----------------------------
    print("\n=== Augmenting training data ===")
    X_train, y_train = augment(X_train_raw, y_train_raw)
    print("Augmented train size: %d  (%.1fx)" % (
        len(X_train), len(X_train) / len(X_train_raw)))

    # ---- Class distribution after augmentation ----------------------
    counts = Counter(y_train.argmax(-1))
    print("Class distribution: " + ", ".join(
        "%s=%d" % (LABEL_NAMES[i], counts[i]) for i in range(3)))

    # ---- Class weights to counter remaining imbalance ---------------
    total = sum(counts.values())
    class_weight = {i: total / (3.0 * max(counts[i], 1)) for i in range(3)}
    print("Class weights: " + ", ".join(
        "%s=%.2f" % (LABEL_NAMES[i], class_weight[i]) for i in range(3)))

    # ---- Build model ------------------------------------------------
    model = build_cnn()
    model.summary()
    print("\nTotal params: %d" % model.count_params())

    # ---- Compile and train (50 epochs) ------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(
            '../models/lane_cnn_best.keras',
            save_best_only=True, monitor='val_accuracy', verbose=1)
    ]

    print("\n=== Training (max 50 epochs) ===")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_raw, y_val),
        class_weight=class_weight,
        callbacks=callbacks
    )

    # ---- Save -------------------------------------------------------
    model.save('../models/lane_cnn.keras')
    print("\nSaved: ../models/lane_cnn.keras")

    try:
        converter    = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open('../models/lane_cnn.tflite', 'wb') as f:
            f.write(tflite_model)
        print("Saved: ../models/lane_cnn.tflite  (%d KB)" % (len(tflite_model) // 1024))
    except Exception as e:
        print("TFLite conversion failed: %s" % e)

    # ---- Final evaluation -------------------------------------------
    best_val_acc = max(history.history['val_accuracy'])
    print("\nBest val accuracy during training: %.2f%%" % (best_val_acc * 100))
    print("\n=== Final evaluation ===")
    evaluate_detail(model, X_train_raw, y_train_raw, "Train (unaugmented)")
    evaluate_detail(model, X_val_raw,   y_val,       "Val   (unaugmented)")


if __name__ == '__main__':
    main()
