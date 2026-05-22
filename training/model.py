import sys
import time
import glob
import os
import tempfile
import cv2
import numpy as np


def load_data(glob_pattern='../data/*.npz'):
    X = np.empty((0, 50400))
    y = np.empty((0, 3), 'float')
    files = glob.glob(glob_pattern)
    if not files:
        print("No data found at:", glob_pattern)
        sys.exit(1)
    loaded, skipped = 0, 0
    for path in files:
        with np.load(path) as data:
            train = data['train']
            labels = data['train_labels']
            if train.ndim != 2 or train.shape[1] != 50400 or len(train) == 0:
                print("  Skipping %s (shape %s)" % (path, train.shape))
                skipped += 1
                continue
            X = np.vstack((X, train))
            y = np.vstack((y, labels))
            loaded += 1
    print("Loaded %d samples from %d files (%d skipped)" % (len(X), loaded, skipped))
    return X, y


def train_val_split(X, y, val_ratio=0.2, seed=42):
    """Randomly split (X, y) into train and validation sets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


class NeuralNetwork(object):
    def __init__(self):
        self.model = None

    def create(self, layer_sizes):
        self.model = cv2.ml.ANN_MLP_create()
        self.model.setLayerSizes(np.int32(layer_sizes))
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2.5, 1)
        self.model.setBackpropWeightScale(0.0001)
        self.model.setBackpropMomentumScale(0.00001)

    def train(self, X, y, X_val=None, y_val=None, max_epochs=50, patience=5):
        """
        Train the model. If X_val/y_val are provided, uses early stopping:
        training halts when validation accuracy has not improved for `patience`
        consecutive epochs. The best weights are restored at the end.

        Without validation data, trains for a fixed 200 iterations.
        """
        start = time.time()
        X, y = np.float32(X), np.float32(y)

        if X_val is None:
            # Simple fixed-iteration training
            self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 200, 0.01))
            print("Training (no validation)...")
            self.model.train(X, cv2.ml.ROW_SAMPLE, y)
            print("Done in %.2fs" % (time.time() - start))
            return

        # Early stopping: run 10 iterations per epoch, check val accuracy
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 10, 1e-6))
        X_val_f, y_val_f = np.float32(X_val), np.float32(y_val)

        best_val_acc = -1.0
        no_improve = 0
        best_path = tempfile.mktemp(suffix='.xml')

        print("Training with early stopping (max_epochs=%d, patience=%d)..." % (max_epochs, patience))

        # First call initialises weights
        self.model.train(X, cv2.ml.ROW_SAMPLE, y)

        for epoch in range(1, max_epochs + 1):
            # UPDATE_WEIGHTS = 1 in OpenCV ml flags
            self.model.train(X, cv2.ml.ROW_SAMPLE, y, flags=1)
            val_acc = self.evaluate(X_val_f, y_val_f)
            train_acc = self.evaluate(X, y)
            print("Epoch %3d | train=%.2f%%  val=%.2f%%" % (epoch, train_acc * 100, val_acc * 100))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                self.model.save(best_path)
            else:
                no_improve += 1

            if no_improve >= patience:
                print("Early stopping at epoch %d — best val=%.2f%%" % (epoch, best_val_acc * 100))
                break

        # Restore best weights
        self.model = cv2.ml.ANN_MLP_load(best_path)
        os.unlink(best_path)
        print("Training complete in %.2fs" % (time.time() - start))

    def evaluate(self, X, y):
        _, resp = self.model.predict(np.float32(X))
        prediction = resp.argmax(-1)
        true_labels = np.array(y).argmax(-1)
        return float(np.mean(prediction == true_labels))

    def save(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print("Model saved to:", path)

    def load(self, path):
        if not os.path.exists(path):
            print("Model not found:", path)
            sys.exit(1)
        self.model = cv2.ml.ANN_MLP_load(path)

    def predict(self, X):
        _, resp = self.model.predict(np.float32(X))
        return resp.argmax(-1)
