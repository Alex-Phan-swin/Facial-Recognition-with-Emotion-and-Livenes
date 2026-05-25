# =========================================
# 1. IMPORTS
# =========================================
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
from PIL import Image

# =========================================
# 2. CONFIG
# =========================================
DATA_ROOT = r"C:\Users\minhp\Music\Year 3\COS30082\project\Facial-Recognition-with-Emotion-and-Livenes\dataset"

CLS_ROOT = os.path.join(DATA_ROOT, "classification_data")
TRAIN_DIR = os.path.join(CLS_ROOT, "train_data")
VAL_DIR   = os.path.join(CLS_ROOT, "val_data")

IMG_SIZE = (80, 80)
BATCH_SIZE = 64
EMBED_DIM = 128
EPOCHS = 1

# =========================================
# 3. LOAD DATASET
# =========================================
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Normalize
def normalize(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

# =========================================
# 4. FACE Supervised Learning MODEL (CNN)
# =========================================
base_model = keras.applications.MobileNetV2(
    input_shape=(80, 80, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # transfer learning allowed

inputs = keras.Input(shape=(80, 80, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

# EMBEDDING LAYER
embeddings = layers.Dense(EMBED_DIM, name="embedding_layer")(x)

# CLASSIFIER
outputs = layers.Dense(num_classes, activation="softmax")(embeddings)

model = keras.Model(inputs, outputs)

# Separate embedding model (IMPORTANT FOR VERIFICATION)
embedding_model = keras.Model(inputs, embeddings)

# =========================================
# 5. TRAIN MODEL (SUPERVISED LEARNING)
# =========================================
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


# =========================================
# 5.1 SAVE MODELS (ADDED)
# =========================================

# Save full classification model
#model.save("face_classifier_Supervised.keras")

# Save embedding model (IMPORTANT for verification)
#embedding_model.save("face_embedding_model.keras")

print("Models saved successfully!")

# =========================================
# 6. LOAD VERIFICATION PAIRS
# =========================================
def load_pairs(file_path):
    pairs = []
    with open(file_path, "r") as f:
        for line in f:
            p1, p2, label = line.strip().split()
            pairs.append((p1, p2, int(label)))
    return pairs

# =========================================
# 7. IMAGE PREPROCESSING FOR VERIFICATION
# =========================================
def load_image(path):
    full_path = os.path.join(DATA_ROOT, path)
    img = Image.open(full_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# =========================================
# 8. EMBEDDING EXTRACTION
# =========================================
def get_embedding(img):
    return embedding_model.predict(img, verbose=0)[0]

# =========================================
# 9. SIMILARITY METRICS
# =========================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# =========================================
# 10. EVALUATION (ROC + AUC)
# =========================================
def evaluate(pairs, metric="cosine"):
    scores = []
    labels = []

    for p1, p2, label in pairs:

        img1 = load_image(p1)
        img2 = load_image(p2)

        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)

        if metric == "cosine":
            score = cosine_similarity(emb1, emb2)
        else:
            score = -euclidean_distance(emb1, emb2)

        scores.append(score)
        labels.append(label)

    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)

# =========================================
# 11. RUN VERIFICATION
# =========================================
pairs_path = os.path.join(DATA_ROOT, "verification_pairs_val.txt")
pairs = load_pairs(pairs_path)

cosine_auc = evaluate(pairs, "cosine")
euclid_auc = evaluate(pairs, "euclidean")

print("Cosine AUC:", cosine_auc)
print("Euclidean AUC:", euclid_auc)