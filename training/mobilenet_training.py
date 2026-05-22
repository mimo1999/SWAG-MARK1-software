"""
MobileNetV2 transfer learning for lane classification.

Two-phase training:
  Phase 1 — freeze base, train classification head only (fast convergence)
  Phase 2 — unfreeze last 30 base layers, fine-tune at low LR

Outputs:
  ../models/mobilenet_lane.h5       — full Keras model
  ../models/mobilenet_lane.tflite   — TFLite model for Pi deployment

Usage:
    python mobilenet_training.py
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF info/warning noise

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load_data, train_val_split

IMG_SIZE   = 96
BATCH_SIZE = 32
LABEL_NAMES = ['forward', 'left', 'right']


def preprocess(X_flat):
    """
    Convert flat grayscale arrays (n, 50400) to MobileNetV2 input (n, 96, 96, 3).
    Pixel values normalised to [-1, 1].
    """
    n = len(X_flat)
    out = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for i, flat in enumerate(X_flat):
        img = flat.reshape(120, 420).astype(np.float32)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = (img / 127.5) - 1.0
        out[i] = np.stack([img, img, img], axis=-1)
    return out


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Inline augmentation — only active during model.fit()
    x = layers.RandomBrightness(0.2)(inputs)
    x = layers.RandomContrast(0.2)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(3, activation='softmax')(x)

    return keras.Model(inputs, outputs), base


def evaluate_detail(model, X, y, split_name):
    y_pred = model.predict(X, verbose=0).argmax(-1)
    y_true = y.argmax(-1)
    acc = np.mean(y_pred == y_true)
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


def main():
    print("=== Loading data ===")
    X_flat, y = load_data('../data/*.npz')
    X_train_f, X_val_f, y_train, y_val = train_val_split(X_flat, y, val_ratio=0.2)
    print("Train: %d  Val: %d" % (len(X_train_f), len(X_val_f)))

    print("\n=== Preprocessing (reshape + resize + normalise) ===")
    X_train = preprocess(X_train_f)
    X_val   = preprocess(X_val_f)
    print("Input shape: %s" % str(X_train.shape))

    model, base = build_model()
    print("\nTrainable params (phase 1): %d" %
          sum(tf.size(w).numpy() for w in model.trainable_weights))

    # ------------------------------------------------------------------ #
    # Phase 1 — head only
    # ------------------------------------------------------------------ #
    print("\n=== Phase 1: training classification head ===")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                          verbose=1),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
        ]
    )

    # ------------------------------------------------------------------ #
    # Phase 2 — fine-tune last 30 base layers
    # ------------------------------------------------------------------ #
    print("\n=== Phase 2: fine-tuning last 30 base layers ===")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    print("Trainable params (phase 2): %d" %
          sum(tf.size(w).numpy() for w in model.trainable_weights))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                          verbose=1),
            keras.callbacks.ModelCheckpoint(
                '../models/mobilenet_lane_best.keras',
                save_best_only=True, monitor='val_accuracy', verbose=1)
        ]
    )

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    model.save('../models/mobilenet_lane.keras')
    print("\nSaved: ../models/mobilenet_lane.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('../models/mobilenet_lane.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Saved: ../models/mobilenet_lane.tflite  (%d KB)" % (len(tflite_model) // 1024))

    # ------------------------------------------------------------------ #
    # Final evaluation
    # ------------------------------------------------------------------ #
    print("\n=== Final evaluation ===")
    evaluate_detail(model, X_train, y_train, "Train")
    evaluate_detail(model, X_val,   y_val,   "Val  ")


if __name__ == '__main__':
    main()
