import sys

from pathlib import Path

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anti_spoofing.config import (
    FRAMES_ROOT,
    MODEL_SAVE_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    LR_PATIENCE,
    EARLY_STOP,
    WARMUP_EPOCHS,
    SEED,
)
from anti_spoofing.dataset import build_dataset, compute_class_weight
from anti_spoofing.model import build_liveness_model, unfreeze_backbone

tf.random.set_seed(SEED)


def make_callbacks() -> list:
    return [
        # save the model whenever val_auc improves
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # halve the learning rate if val_auc stalls
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=LR_PATIENCE,
            verbose=1,
        ),
        # stop early if val_auc stops improving
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def main():
    # load datasets
    train_ds = build_dataset(FRAMES_ROOT, "train", IMG_SIZE, BATCH_SIZE, SEED)
    val_ds = build_dataset(FRAMES_ROOT, "val", IMG_SIZE, BATCH_SIZE, SEED)

    # balance classes (more fake videos than real)
    class_weight = compute_class_weight(FRAMES_ROOT, "train")
    print(f"Class weights: {class_weight}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")

    model = build_liveness_model(img_size=IMG_SIZE)
    model.summary()

    # phase 1: train the new head only, backbone frozen
    print(f"\n── Phase 1: frozen backbone, {WARMUP_EPOCHS} epochs ──")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        class_weight=class_weight,
        callbacks=make_callbacks(),
    )

    # phase 2: unfreeze everything and fine-tune with a lower learning rate
    print(
        f"\n── Phase 2: full fine-tuning, up to {NUM_EPOCHS - WARMUP_EPOCHS} more epochs ──"
    )
    model = unfreeze_backbone(model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR * 0.1),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS - WARMUP_EPOCHS,
        class_weight=class_weight,
        callbacks=make_callbacks(),
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
