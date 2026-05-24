import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import json
import keras

from config import BASE_DIR, PROJECT_ROOT


# =========================================
# DIRECTORIES
# =========================================
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'classification_data', 'train_data')
VAL_DIR   = os.path.join(PROJECT_ROOT, 'dataset', 'classification_data', 'val_data')

MODEL_SAVE_PATH  = os.path.join(BASE_DIR, 'face_model.keras')
CLASS_INDEX_PATH = os.path.join(BASE_DIR, 'face_classes.json')

# =========================================
# CONFIG
# =========================================
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EMBEDDING_SIZE = 128
EPOCHS = 10

# =========================================
# REGISTER CUSTOM LAYERS
# So the model can be saved and reloaded without rebuild
# =========================================
@keras.saving.register_keras_serializable(package="FaceModel")
class L2NormLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()

# =========================================
# LOAD DATA
# =========================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
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

base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)

embedding = layers.Dense(
    EMBEDDING_SIZE,
    activation=None,
    name="face_embedding"
)(x)
x = layers.BatchNormalization()(embedding)
x = layers.ReLU()(x)
x = layers.Dropout(0.1)(x)

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
    epochs=EPOCHS,
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
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
# Save full model in .keras format (handles custom objects cleanly)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# Save class indices so main.py can map output index → name
with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train_data.class_indices, f)
print(f"Class indices saved to: {CLASS_INDEX_PATH}")
