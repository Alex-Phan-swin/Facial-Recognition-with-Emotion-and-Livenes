from pathlib import Path
import numpy as np
import tensorflow as tf

# fake=0, real=1
_CLASS_NAMES = ["fake", "real"]


def build_dataset(
    frames_root: Path,
    split: str,
    img_size: int = 224,
    batch_size: int = 32,
    seed: int = 42,
) -> tf.data.Dataset:
    directory = frames_root / split

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {directory}\nRun extract_frames.py first."
        )

    # load images from the split folder and assign labels from subfolder names
    ds = tf.keras.utils.image_dataset_from_directory(
        str(directory),
        class_names=_CLASS_NAMES,
        label_mode="binary",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=(split == "train"),
        seed=seed,
    )

    # augment only training data
    if split == "train":
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def _augment(images, labels):
    # random colour and flip changes to help the model generalise
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=0.2)
    images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
    images = tf.image.random_saturation(images, lower=0.8, upper=1.2)
    images = tf.clip_by_value(images, 0.0, 255.0)
    return images, labels


def compute_class_weight(frames_root: Path, split: str = "train") -> dict:
    # count frames per class so training treats both equally
    real_dir = frames_root / split / "real"
    fake_dir = frames_root / split / "fake"

    n_real  = len(list(real_dir.glob("*.jpg"))) if real_dir.exists() else 0
    n_fake  = len(list(fake_dir.glob("*.jpg"))) if fake_dir.exists() else 0
    n_total = n_real + n_fake

    if n_total == 0:
        return {0: 1.0, 1: 1.0}

    return {
        0: n_total / (2 * n_fake) if n_fake else 1.0,
        1: n_total / (2 * n_real) if n_real else 1.0,
    }
