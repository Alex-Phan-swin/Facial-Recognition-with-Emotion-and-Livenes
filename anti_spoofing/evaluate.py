import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anti_spoofing.config import (
    FRAMES_ROOT,
    MODEL_SAVE_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    LIVENESS_THRESHOLD,
)
from anti_spoofing.dataset import build_dataset


def evaluate_split(split: str = "test") -> dict:
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_SAVE_PATH}\n"
            "Train first:  python -m anti_spoofing.train"
        )

    # Load the trained model from disk
    model = keras.models.load_model(str(MODEL_SAVE_PATH))

    # Build the data pipeline for this split (no augmentation, no shuffle)
    ds = build_dataset(FRAMES_ROOT, split, IMG_SIZE, BATCH_SIZE)

    # collect predictions and true labels for every batch
    all_probs, all_labels = [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        all_probs.extend(preds.flatten().tolist())
        all_labels.extend(labels.numpy().flatten().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # Convert probabilities to hard predictions using the threshold (default 0.5)
    y_pred = (y_prob >= LIVENESS_THRESHOLD).astype(int)

    accuracy = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))

    return {"accuracy": accuracy, "auc": auc}


def main():
    for split in ("val", "test"):
        print(f"\n── {split.upper()} SET ──────────────────")
        m = evaluate_split(split)
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  AUC      : {m['auc']:.4f}")


if __name__ == "__main__":
    main()
