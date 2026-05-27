import random
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from laptop_detection.config import (
    LAPTOP_DATA_ROOT, NEGATIVE_DATA_ROOT,
    NUM_NEGATIVES, TRAIN_FRAC, VAL_FRAC, SEED,
    IMG_SIZE, BATCH_SIZE,
)


def _load_image(path: Path) -> np.ndarray | None:
    # PIL handles webp as well as jpg/png
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        return np.array(img, dtype=np.float32)
    except Exception:
        return None


def _collect_paths(root: Path, extensions: tuple) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(root.rglob(f"*{ext}"))
    return paths


def build_tf_dataset(split: str) -> tf.data.Dataset:
    rng = random.Random(SEED)

    # laptop images — all 17 angle classes = positive examples
    pos_paths = _collect_paths(LAPTOP_DATA_ROOT, (".jpg", ".jpeg", ".png", ".webp"))
    rng.shuffle(pos_paths)

    # face images from FR dataset — no laptops = negative examples
    all_neg  = _collect_paths(NEGATIVE_DATA_ROOT, (".jpg", ".jpeg", ".png"))
    rng.shuffle(all_neg)
    neg_paths = all_neg[:NUM_NEGATIVES]

    # split into train / val / test
    def split_list(lst):
        n       = len(lst)
        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)
        return (
            lst[:n_train],
            lst[n_train : n_train + n_val],
            lst[n_train + n_val :],
        )

    pos_train, pos_val, pos_test = split_list(pos_paths)
    neg_train, neg_val, neg_test = split_list(neg_paths)

    split_map = {
        "train": (pos_train, neg_train),
        "val":   (pos_val,   neg_val),
        "test":  (pos_test,  neg_test),
    }

    positives, negatives = split_map[split]

    # load images into numpy arrays
    images, labels = [], []

    for p in positives:
        arr = _load_image(p)
        if arr is not None:
            images.append(arr)
            labels.append(1.0)

    for p in negatives:
        arr = _load_image(p)
        if arr is not None:
            images.append(arr)
            labels.append(0.0)

    if not images:
        raise ValueError(f"No images loaded for split='{split}'")

    images_np = np.stack(images)
    labels_np = np.array(labels, dtype=np.float32)

    # shuffle images and labels together
    idx = list(range(len(images_np)))
    rng.shuffle(idx)
    images_np = images_np[idx]
    labels_np = labels_np[idx]

    ds = tf.data.Dataset.from_tensor_slices((images_np, labels_np))

    if split == "train":
        ds = ds.shuffle(buffer_size=len(images_np), seed=SEED)
        ds = ds.map(_augment_single, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(f"  [{split}] {int(labels_np.sum())} laptops + "
          f"{int((labels_np == 0).sum())} non-laptops = {len(labels_np)} total")
    return ds


def _augment_single(image, label):
    # random flips and brightness to help the model generalise
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label
