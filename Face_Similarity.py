# =========================================
# 1. IMPORTS
# =========================================
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
from PIL import Image
from tensorflow.keras import mixed_precision


# =========================================
# 2. CONFIG
# =========================================
DATA_ROOT = r"C:\Users\minhp\Music\Year 3\COS30082\project\Facial-Recognition-with-Emotion-and-Livenes"

DATASET_ROOT = os.path.join(DATA_ROOT, "dataset")

CLS_ROOT = os.path.join(DATASET_ROOT, "classification_data")

TRAIN_DIR = os.path.join(CLS_ROOT, "train_data")
VAL_DIR   = os.path.join(CLS_ROOT, "val_data")
TEST_DIR  = os.path.join(CLS_ROOT, "test_data")

IMG_SIZE = 80
BATCH_SIZE = 64
EMBED_DIM = 128
EPOCHS = 30


#Speed optimization for 
mixed_precision.set_global_policy("mixed_float16")

# =========================================
# 3. LOAD DATASET
# =========================================
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# =========================================
# 4. NORMALIZATION
# =========================================
def normalize(img, label):

    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5

    return img, label

train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)
test_ds = test_ds.map(normalize)

# =========================================
# 6. Mobile FaceNet ARCHITECTURE
# =========================================
def conv_bn(x, filters, stride):
    x = layers.Conv2D(filters, 3, stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    return x


def conv_dw(x, filters, stride):
    x = layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)

    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    return x


def bottleneck(x, inp, oup, stride, expand):
    hidden = int(inp * expand)
    shortcut = x

    if expand != 1:
        x = layers.Conv2D(hidden, 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)

    x = layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(oup, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and inp == oup:
        x = layers.Add()([shortcut, x])

    x = layers.PReLU()(x)
    return x


def build_model(num_classes, emb_dim=128):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = conv_bn(inputs, 64, 2)
    x = conv_dw(x, 64, 1)

    x = bottleneck(x, 64, 64, 2, 2)
    x = bottleneck(x, 64, 64, 1, 2)
    x = bottleneck(x, 64, 128, 2, 4)
    x = bottleneck(x, 128, 128, 1, 2)
    x = bottleneck(x, 128, 128, 1, 4)
    x = bottleneck(x, 128, 128, 1, 2)

    x = conv_bn(x, 512, 1)

    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(emb_dim)(x)
    x = layers.BatchNormalization()(x)

    # L2 normalize embeddings
    x = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1))(x)

    outputs = layers.Dense(num_classes, dtype="float32")(x)

    return keras.Model(inputs, outputs)


model = build_model(num_classes, EMBED_DIM)
model.summary()

# =========================================
# 7. COMPILE MODEL and loss optimizer
# =========================================
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy"]
)

# =========================================
# 8. EARLY STOPPING, Reduce learning rate on plateau, and Model Checkpoint callbacks
# =========================================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    
    keras.callbacks.ModelCheckpoint(
        'face_classifier.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

# =========================================
# 9. TRAIN MODEL
# =========================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================
# 9. Evaluuation
# =========================================
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

# =========================================
# 10. SAVE MODELS
# =========================================
model.save("face_classifier.keras")

print("Models saved successfully!")

# =========================================
# 11. LOAD VERIFICATION PAIRS
# =========================================
def load_pairs(file_path):

    pairs = []

    with open(file_path, "r") as f:

        for line in f:

            p1, p2, label = line.strip().split()

            pairs.append((p1, p2, int(label)))

    return pairs

# =========================================
# 12. IMAGE PREPROCESSING
# =========================================
def load_image(path):

    full_path = os.path.join(DATASET_ROOT, path)

    img = Image.open(full_path).convert("RGB")

    img = img.resize(IMG_SIZE)

    img = np.array(img).astype("float32") / 255.0

    return np.expand_dims(img, axis=0)

# =========================================
# 13. EMBEDDING EXTRACTION
# =========================================
def get_embedding(img):

    embedding = embedding_model.predict(
        img,
        verbose=0
    )[0]

    return embedding

# =========================================
# 14. SIMILARITY METRICS
# =========================================
def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def euclidean_distance(a, b):

    return np.linalg.norm(a - b)

# =========================================
# 15. EVALUATION
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
# 16. RUN VERIFICATION
# =========================================
pairs_path = os.path.join(
    DATASET_ROOT,
    "verification_pairs_val.txt"
)

pairs = load_pairs(pairs_path)

cosine_auc = evaluate(pairs, "cosine")

euclid_auc = evaluate(pairs, "euclidean")

print("\n===================================")
print("VERIFICATION RESULTS")
print("===================================")

print("Cosine AUC:", cosine_auc)

print("Euclidean AUC:", euclid_auc)