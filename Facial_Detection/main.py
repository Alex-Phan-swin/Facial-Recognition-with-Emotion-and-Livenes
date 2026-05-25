import cv2
import os
import time
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import keras

from .config import DB_PATH, EXIT_DELAY, DISPLAY_DELAY, MAX_IMAGES, BASE_DIR, PROJECT_ROOT
from .logger import init_log, log_event

# -------------------------------
# SETUP
# -------------------------------
os.makedirs(DB_PATH, exist_ok=True)
init_log()

# -------------------------------
# REGISTER SAME CUSTOM OBJECTS
# Must match exactly what was registered in model_training.py
# so load_model can deserialize the saved model correctly
# -------------------------------
@keras.saving.register_keras_serializable(package="FaceModel")
class L2NormLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()

# -------------------------------
# PATHS & CONFIG
# -------------------------------
MODEL_PATH       = os.path.join(BASE_DIR, "face_model.keras")
#Model path for supervised learning model
#MODEL_PATH       = os.path.join(BASE_DIR, "face_classifier_Supervised.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "face_classes.json")
TRAIN_DIR        = os.path.join(PROJECT_ROOT, 'dataset', 'classification_data', 'train_data')

CONFIDENCE_THRESHOLD = 0.60

# -------------------------------
# LOAD CLASS NAMES
# -------------------------------
if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    class_names = [None] * len(class_indices)
    for name, index in class_indices.items():
        class_names[index] = name
else:
    # Fallback: read folder names directly (must be sorted to match training order)
    class_names = sorted(os.listdir(TRAIN_DIR))

num_classes = len(class_names)
print("Class names:", class_names)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

embedding_model = keras.Model(
    inputs=model.input,
    #outputs=model.get_layer("face_embedding").output
    outputs=model.get_layer("embedding_layer").output
)

# -------------------------------
# REGISTER STATE
# -------------------------------
REGISTER_MODE = False
SAVE_COUNT = 0
new_person_name = ""

# -------------------------------
# STATE TRACKING
# -------------------------------
face_database = {}
last_seen = {}
inside = set()
visible = set()
last_visible = {}
frame_count = 0
last_predictions = []

# -------------------------------
# HELPER FUNCTION
# -------------------------------
def get_embedding(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    #Image Size supervised Learning model
    #face_img = cv2.resize(face_img, (80, 80))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    emb = embedding_model.predict(face_img, verbose=0)[0]

    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb

    emb = emb / norm
    return emb

def build_face_database():
    global face_database
    face_database = {}

    for person_name in os.listdir(DB_PATH):
        person_folder = os.path.join(DB_PATH, person_name)

        if not os.path.isdir(person_folder):
            continue

        embeddings = []

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            emb = get_embedding(img)
            embeddings.append(emb)

        if len(embeddings) > 0:
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            face_database[person_name] = avg_emb

    print("Loaded people from faces_db:", list(face_database.keys()))

def predict_face(face_img):
    if len(face_database) == 0:
        return "Unknown", 0.0

    query_emb = get_embedding(face_img)

    best_name = "Unknown"
    best_score = -1.0

    for name, saved_emb in face_database.items():
        similarity = np.dot(query_emb, saved_emb)

        if similarity > best_score:
            best_score = similarity
            best_name = name

    if best_score < CONFIDENCE_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score

def run():
    global frame_count, REGISTER_MODE, SAVE_COUNT, new_person_name, last_predictions

    build_face_database()

    # -------------------------------
    # CAMERA
    # -------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("Press 'q' to quit | Press 'r' to register")

    # -------------------------------
    # MAIN LOOP
    # -------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detected_people = set()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        face_present = len(faces) > 0

        key = cv2.waitKey(1) & 0xFF

        # -------------------------------
        # EXIT KEY
        # -------------------------------
        if key == ord('q'):
            break

        # -------------------------------
        # REGISTER MODE
        # -------------------------------
        if key == ord('r') and not REGISTER_MODE:
            new_person_name = input("Enter employee name: ")
            person_path = os.path.join(DB_PATH, new_person_name)
            os.makedirs(person_path, exist_ok=True)

            REGISTER_MODE = True
            SAVE_COUNT = 0
            print(f"Registering {new_person_name}...")

        # -------------------------------
        # SAVE FACES
        # -------------------------------
        if REGISTER_MODE and face_present:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))

                save_path = os.path.join(
                    DB_PATH,
                    new_person_name,
                    f"{new_person_name}_{SAVE_COUNT}.jpg"
                )

                cv2.imwrite(save_path, face)
                SAVE_COUNT += 1
                print(f"Saved {save_path}")

                time.sleep(0.5)

                if SAVE_COUNT >= MAX_IMAGES:
                    print(f"Finished registering {new_person_name}")
                    REGISTER_MODE = False
                    break

        # -------------------------------
        # FACE RECOGNITION
        # -------------------------------
        if face_present and not REGISTER_MODE:
            current_predictions = []

            if frame_count % 10 == 0:
                try:
                    for (x, y, w, h) in faces:
                        face = frame[y:y + h, x:x + w]
                        predicted_name, predicted_confidence = predict_face(face)

                        current_predictions.append((x, y, w, h, predicted_name, predicted_confidence))

                        print("Prediction:", predicted_name, "Confidence:", round(float(predicted_confidence), 3))

                        if predicted_name != "Unknown":
                            detected_people.add(predicted_name)
                            visible.add(predicted_name)
                            last_visible[predicted_name] = time.time()

                    last_predictions = current_predictions

                except Exception as e:
                    print("Model prediction error:", e)

            # Draw last known prediction every frame
            for (x, y, w, h, predicted_name, predicted_confidence) in last_predictions:
                label = f"{predicted_name} ({predicted_confidence:.2f})"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0) if predicted_name != "Unknown" else (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if predicted_name != "Unknown" else (0, 0, 255),
                    2
                )


        # -------------------------------
        # DISPLAY
        # -------------------------------
        for i, name in enumerate(visible):
            cv2.putText(frame, name, (50, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # -------------------------------
        # ENTRY LOGIC
        # -------------------------------
        for person in detected_people:
            last_seen[person] = time.time()

            if person not in inside:
                inside.add(person)
                log_event(person, "ENTER")
                print(person, "ENTER")

        # -------------------------------
        # EXIT LOGIC
        # -------------------------------
        for person in list(inside):
            if person not in last_seen:
                continue

            if time.time() - last_seen[person] > EXIT_DELAY:
                inside.remove(person)
                log_event(person, "EXIT")
                print(person, "EXIT")

        # -------------------------------
        # CLEAN MEMORY
        # -------------------------------
        for person in list(last_seen.keys()):
            if time.time() - last_seen[person] > EXIT_DELAY * 2:
                del last_seen[person]

        # -------------------------------
        # DISPLAY CLEANUP
        # -------------------------------
        for person in list(visible):
            if person not in last_visible:
                continue

            if time.time() - last_visible[person] > DISPLAY_DELAY:
                visible.remove(person)
                del last_visible[person]

        # -------------------------------
        # SHOW FRAME
        # -------------------------------
        cv2.imshow("Face Recognition", frame)

    # -------------------------------
    # CLEANUP
    # -------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
