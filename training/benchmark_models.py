"""
Benchmark all trained models in ../models/ against the full dataset.
Handles both OpenCV ANN (.xml) and Keras MobileNetV2 (.h5) models.

Usage:
    python benchmark_models.py
    python benchmark_models.py --data ../data/test025.npz
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import cv2
import numpy as np
from model import load_data

LABEL_NAMES = ['forward', 'left', 'right']
IMG_SIZE = 96


# ------------------------------------------------------------------ #
# Preprocessing
# ------------------------------------------------------------------ #

def preprocess_mobilenet(X_flat):
    """(n, 50400) → (n, 96, 96, 3) normalised to [-1, 1]."""
    n = len(X_flat)
    out = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for i, flat in enumerate(X_flat):
        img = flat.reshape(120, 420).astype(np.float32)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = (img / 127.5) - 1.0
        out[i] = np.stack([img, img, img], axis=-1)
    return out


# ------------------------------------------------------------------ #
# Metrics helpers
# ------------------------------------------------------------------ #

def confusion_matrix(y_true, y_pred, n=3):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm):
    metrics = []
    for i, name in enumerate(LABEL_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics.append((name, precision, recall, f1))
    return metrics


def print_confusion_matrix(cm, model_name):
    print("\n  Confusion matrix — %s" % model_name)
    header = "         " + "  ".join("%-8s" % n for n in LABEL_NAMES)
    print(header)
    for i, name in enumerate(LABEL_NAMES):
        row = "  %-7s" % name + "  ".join("%8d" % cm[i, j] for j in range(len(LABEL_NAMES)))
        print(row)


def make_result(y_true, y_pred):
    accuracy = float(np.mean(y_pred == y_true))
    cm = confusion_matrix(y_true, y_pred)
    return dict(accuracy=accuracy, cm=cm, metrics=per_class_metrics(cm))


# ------------------------------------------------------------------ #
# Per-model-type benchmarks
# ------------------------------------------------------------------ #

def benchmark_xml(path, X_flat, y_true):
    try:
        model = cv2.ml.ANN_MLP_load(path)
    except Exception as e:
        return None, "load error: %s" % e
    _, resp = model.predict(np.float32(X_flat))
    y_pred = resp.argmax(-1)
    return make_result(y_true, y_pred), None


def benchmark_keras(path, X_flat, y_true):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
    except Exception as e:
        return None, None, "load error: %s" % e
    # Auto-detect input shape: flat (50400,) CNN vs image (96,96,3) MobileNet
    in_shape = model.input_shape  # e.g. (None, 50400) or (None, 96, 96, 3)
    if len(in_shape) == 2 and in_shape[1] == 50400:
        X   = X_flat  # flat input — preprocessing is embedded in the model graph
        tag = "(CNN)"
    else:
        X   = preprocess_mobilenet(X_flat)
        tag = "(MobileNetV2)"
    y_pred = model.predict(X, verbose=0).argmax(-1)
    return make_result(y_true, y_pred), tag, None


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    glob_pattern = '../data/*.npz'
    if '--data' in sys.argv:
        glob_pattern = sys.argv[sys.argv.index('--data') + 1]

    print("Loading data from:", glob_pattern)
    X, y = load_data(glob_pattern)
    y_true = y.argmax(-1)
    print("Total samples: %d\n" % len(X))

    # Collect all model files
    xml_paths = sorted(glob.glob('../models/*.xml'))
    h5_paths  = sorted(glob.glob('../models/*.h5') + glob.glob('../models/*.keras'))

    if not xml_paths and not h5_paths:
        print("No model files found in ../models/")
        sys.exit(1)

    results = []

    print("Benchmarking ANN models (.xml)...")
    for path in xml_paths:
        name = os.path.basename(path)
        result, err = benchmark_xml(path, X, y_true)
        tag = "(ANN)"
        if err:
            print("  %-35s  SKIPPED (%s)" % (name, err))
        else:
            print("  %-35s  %.2f%%" % (name, result['accuracy'] * 100))
            results.append((name + " " + tag, result))

    if h5_paths:
        print("\nBenchmarking Keras models (.h5 / .keras)...")
        for path in h5_paths:
            name = os.path.basename(path)
            result, tag, err = benchmark_keras(path, X, y_true)
            if err:
                print("  %-35s  SKIPPED (%s)" % (name, err))
            else:
                print("  %-35s  %.2f%%" % (name, result['accuracy'] * 100))
                results.append((name + " " + tag, result))

    if not results:
        print("No models could be loaded.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    col = "%-46s  %8s  %14s  %14s  %14s"
    divider = "=" * 102
    print("\n" + divider)
    print(col % ("Model", "Accuracy", "Forward P/R/F1", "Left P/R/F1", "Right P/R/F1"))
    print(divider)

    # Sort by accuracy descending
    for name, r in sorted(results, key=lambda x: x[1]['accuracy'], reverse=True):
        cells = ["%3.0f/%3.0f/%3.0f" % (p*100, rec*100, f1*100)
                 for _, p, rec, f1 in r['metrics']]
        print(col % (name, "%.2f%%" % (r['accuracy'] * 100), cells[0], cells[1], cells[2]))

    print(divider)

    # Confusion matrices
    for name, r in sorted(results, key=lambda x: x[1]['accuracy'], reverse=True):
        print_confusion_matrix(r['cm'], name)

    best_name, best_r = max(results, key=lambda x: x[1]['accuracy'])
    print("\n  Best model: %s  (%.2f%%)" % (best_name, best_r['accuracy'] * 100))


if __name__ == '__main__':
    main()
