import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from laptop_detection.config import (
    MODEL_SAVE_PATH, IMG_SIZE, BATCH_SIZE,
    NUM_EPOCHS, LR, LR_PATIENCE, EARLY_STOP, WARMUP_EPOCHS, SEED,
)
from laptop_detection.dataset import build_tf_dataset
from laptop_detection.model import build_laptop_model, unfreeze_backbone

tf.random.set_seed(SEED)


def make_callbacks() -> list:
    return [
        # save the model file whenever val_auc improves
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # reduce learning rate if val_auc stops improving
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max",
            factor=0.5, patience=LR_PATIENCE, verbose=1,
        ),
        # stop training early if no improvement for a while
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=EARLY_STOP, restore_best_weights=True, verbose=1,
        ),
    ]


def main():
    # load train and val datasets
    print("Loading datasets...")
    train_ds = build_tf_dataset("train")
    val_ds   = build_tf_dataset("val")

    model = build_laptop_model(img_size=IMG_SIZE)
    model.summary()

    # phase 1: train only the new head, backbone stays frozen
    print(f"\nPhase 1: frozen backbone, {WARMUP_EPOCHS} epochs")
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc")],
    )
    model.fit(train_ds, validation_data=val_ds,
              epochs=WARMUP_EPOCHS, callbacks=make_callbacks())

    # phase 2: unfreeze everything and fine-tune with a lower learning rate
    print(f"\nPhase 2: full fine-tuning, up to {NUM_EPOCHS - WARMUP_EPOCHS} epochs")
    model = unfreeze_backbone(model)
    model.compile(
        optimizer=keras.optimizers.Adam(LR * 0.1),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc")],
    )
    model.fit(train_ds, validation_data=val_ds,
              epochs=NUM_EPOCHS - WARMUP_EPOCHS, callbacks=make_callbacks())

    print(f"\nDone. Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
