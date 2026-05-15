import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import json

from config import BASE_DIR, PROJECT_ROOT


# =========================================
# DIRECTORIES
# =========================================
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'classification_data', 'train_data')
VAL_DIR   = os.path.join(PROJECT_ROOT, 'dataset', 'classification_data', 'val_data')

MODEL_SAVE_PATH  = os.path.join(BASE_DIR, 'face_model.h5')

# =========================================
# CONFIG
# =========================================
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EMBEDDING_SIZE = 128
EPOCHS = 10

# =========================================
# LOAD DATA
# =========================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_data.num_classes

# =========================================
# BUILD MODEL
# =========================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# Freeze all base model layers for feature extraction phase
base_model.trainable = False

# Use full base model output instead of cutting at block_13
x = layers.GlobalAveragePooling2D()(base_model.output)

# Embedding layer
embedding = layers.Dense(
    EMBEDDING_SIZE,
    activation=None,
    name="face_embedding"
)(x)
x = layers.BatchNormalization()(embedding)
x = layers.ReLU()(x)
x = layers.Dropout(0.1)(x)  # reduced from 0.3

output = layers.Dense(
    num_classes,
    activation="softmax",
    name="identity_classifier"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# =========================================
# PHASE 1 — FEATURE EXTRACTION
# =========================================
print("\n=== Phase 1: Feature Extraction ===")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]
)

# =========================================
# PHASE 2 — FINE-TUNING
# =========================================
print("\n=== Phase 2: Fine-Tuning ===")

for layer in base_model.layers:
    if any(layer.name.startswith(b) for b in ["block_11", "block_12", "block_13"]):
        layer.trainable = True

# Lower learning rate to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]
)

# =========================================
# SAVE
# =========================================
model.save(MODEL_SAVE_PATH)