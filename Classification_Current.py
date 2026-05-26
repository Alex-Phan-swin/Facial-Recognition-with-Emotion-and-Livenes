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
DATA_ROOT = r"C:\Users\fayiz\OneDrive\Desktop\UNI\SEM 6\Applied Machine Learning\Project\Facial-Recognition-with-Emotion-and-Livenes\dataset"

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
# 4. FACE EMBEDDING MODEL (CNN)
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
# 5.1 SAVE MODELS
# =========================================
model.save("face_classifier.keras")
embedding_model.save("face_embedding_model.keras")
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
# 9.1 TRIPLET LOSS FUNCTION
# =========================================
def triplet_loss(anchor, positive, negative, margin=0.5):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)

# =========================================
# 9.2 TRIPLET DATASET BUILDER
# =========================================
def build_triplet_dataset(directory, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    raw_ds = keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=None,
        shuffle=True
    )
    class_names_local = raw_ds.class_names

    # Store only file paths grouped by class — no images loaded into RAM
    class_image_paths = {name: [] for name in class_names_local}
    for class_name in os.listdir(directory):
        class_folder = os.path.join(directory, class_name)
        if not os.path.isdir(class_folder):
            continue
        if class_name not in class_image_paths:
            continue
        for fname in os.listdir(class_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                class_image_paths[class_name].append(
                    os.path.join(class_folder, fname)
                )

    def load_img_from_path(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def triplet_generator():
        classes = [c for c in class_names_local if len(class_image_paths[c]) >= 2]
        other_classes = {cls: [c for c in classes if c != cls] for cls in classes}

        for cls in classes:
            imgs = class_image_paths[cls]
            if not other_classes[cls]:
                continue
            for i in range(0, len(imgs) - 1, 5):  # step=5 keeps memory low
                anchor_path   = imgs[i]
                positive_path = imgs[i + 1]
                neg_cls       = np.random.choice(other_classes[cls])
                neg_imgs      = class_image_paths[neg_cls]
                negative_path = neg_imgs[np.random.randint(len(neg_imgs))]

                anchor   = load_img_from_path(anchor_path)
                positive = load_img_from_path(positive_path)
                negative = load_img_from_path(negative_path)

                yield anchor, positive, negative

    ds = tf.data.Dataset.from_generator(
        triplet_generator,
        output_signature=(
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
        )
    )

    return ds.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# =========================================
# 10. EVALUATION FUNCTION
# =========================================
def evaluate_pairs(pairs, metric="cosine"):
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
# 11. TRIPLET LOSS MODEL + TRAINING
# =========================================
print("\nBuilding triplet dataset...")
triplet_train_ds = build_triplet_dataset(TRAIN_DIR)

# Separate model with same architecture — does not share weights with softmax model
triplet_inputs = keras.Input(shape=(80, 80, 3))
tx = base_model(triplet_inputs, training=False)
tx = layers.GlobalAveragePooling2D()(tx)
triplet_embeddings = layers.Dense(EMBED_DIM, name="triplet_embedding_layer")(tx)
triplet_embedding_model = keras.Model(triplet_inputs, triplet_embeddings)

optimizer = keras.optimizers.Adam(1e-3)

@tf.function
def train_step(anchor, positive, negative):
    with tf.GradientTape() as tape:
        emb_a = triplet_embedding_model(anchor,   training=True)
        emb_p = triplet_embedding_model(positive, training=True)
        emb_n = triplet_embedding_model(negative, training=True)
        loss = triplet_loss(emb_a, emb_p, emb_n)
    grads = tape.gradient(loss, triplet_embedding_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, triplet_embedding_model.trainable_variables))
    return loss

print("Training with triplet loss...")
for epoch in range(EPOCHS):
    total_loss = 0
    steps = 0
    for anchor, positive, negative in triplet_train_ds:
        loss = train_step(anchor, positive, negative)
        total_loss += loss.numpy()
        steps += 1
    print(f"Epoch {epoch+1}/{EPOCHS} — Triplet Loss: {total_loss/steps:.4f}")

triplet_embedding_model.save("face_triplet_model.keras")
print("Triplet model saved!")

# =========================================
# 12. COMPARE SOFTMAX vs TRIPLET MODELS
# =========================================
pairs_path = os.path.join(DATA_ROOT, "verification_pairs_val.txt")
pairs = load_pairs(pairs_path)

# --- Softmax embedding model ---
print("\n--- Softmax Embedding Model ---")
# embedding_model is still pointing to the softmax-trained embeddings from Section 4
cosine_auc_soft = evaluate_pairs(pairs, "cosine")
euclid_auc_soft = evaluate_pairs(pairs, "euclidean")
print(f"Cosine AUC:     {cosine_auc_soft:.4f}")
print(f"Euclidean AUC:  {euclid_auc_soft:.4f}")

# --- Triplet embedding model ---
print("\n--- Triplet Loss Model ---")
# Temporarily swap embedding_model so evaluate_pairs uses the triplet model
embedding_model = triplet_embedding_model
cosine_auc_trip = evaluate_pairs(pairs, "cosine")
euclid_auc_trip = evaluate_pairs(pairs, "euclidean")
print(f"Cosine AUC:     {cosine_auc_trip:.4f}")
print(f"Euclidean AUC:  {euclid_auc_trip:.4f}")

# --- Summary table ---
print("\n--- Summary ---")
print(f"{'Metric':<20} {'Softmax':>10} {'Triplet':>10}")
print(f"{'Cosine AUC':<20} {cosine_auc_soft:>10.4f} {cosine_auc_trip:>10.4f}")
print(f"{'Euclidean AUC':<20} {euclid_auc_soft:>10.4f} {euclid_auc_trip:>10.4f}")