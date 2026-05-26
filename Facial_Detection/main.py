import cv2
import os
import time
import numpy as np
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


from config import DB_PATH, EXIT_DELAY, DISPLAY_DELAY, MAX_IMAGES
from logger import init_log, log_event

# -------------------------------
# SETUP
# -------------------------------
# -------------------------------
# DATABASE PATH
# -------------------------------
DB_PATH = r"C:\Uni\Applied Machine Learning\Assignment\Project\Facial-Recognition-with-Emotion-and-Livenes\Facial_Detection"

os.makedirs(DB_PATH, exist_ok=True)
init_log()

# -------------------------------
# REGISTER STATE
# -------------------------------
REGISTER_MODE = False
SAVE_COUNT = 0
new_person_name = ""

# -------------------------------
# STATE TRACKING
# -------------------------------
last_seen = {}
inside = set()
visible = set()
last_visible = {}

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# -------------------------------
# FACE DETECTOR
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# LOAD CLASSIFIER
# -------------------------------
classifier = load_model("face_classifier.keras")


#Emotion Model (Using Pytorch rather than Tensor)
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Build the model again (imports weights from the emotion_train file)
def build_emotion_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1024, len(EMOTIONS))
    )
    return model


emotion_model = build_emotion_model()
emotion_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
emotion_model.to(DEVICE)
emotion_model.eval()

emotion_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #mean and std of the ImageNet ds that the original model was trained on
])

#Predicts the emotion from the face detected. 
def predict_emotion(face_bgr):
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(face_rgb)
        tensor   = emotion_transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(emotion_model(tensor), dim=1)[0]
        return EMOTIONS[int(probs.argmax())], float(probs.max())
    except Exception as e:
        print("Emotion error:", e)
        return "unknown", 0.0

# ONLY LOAD FOLDERS
class_names = sorted([
    name for name in os.listdir(DB_PATH)
    if os.path.isdir(os.path.join(DB_PATH, name))
])

print("Loaded classes:", class_names)

print("Press 'q' to quit | Press 'r' to register")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    detected_people = set()

    # -------------------------------
    # FACE DETECTION
    # -------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    face_present = len(faces) > 0

    # -------------------------------
    # KEY INPUT
    # -------------------------------
    key = cv2.waitKey(1) & 0xFF

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

            # MUST MATCH TRAINING SIZE
            face = cv2.resize(face, (80, 80))

            save_path = os.path.join(
                DB_PATH,
                new_person_name,
                f"{new_person_name}_{SAVE_COUNT}.jpg"
            )

            cv2.imwrite(save_path, face)

            SAVE_COUNT += 1

            print(f"Saved {save_path}")

            time.sleep(0.3)

            if SAVE_COUNT >= MAX_IMAGES:

                print(f"Finished registering {new_person_name}")

                REGISTER_MODE = False

                # reload class names
                class_names = sorted([
                    name for name in os.listdir(DB_PATH)
                    if os.path.isdir(os.path.join(DB_PATH, name))
                ])

                print("Updated classes:", class_names)

                break

    # -------------------------------
    # FACE RECOGNITION
    # -------------------------------
    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        try:

            # preprocess
            face_resized = cv2.resize(face, (80, 80))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            # prediction
            preds = classifier.predict(face_resized, verbose=0)[0]

            confidence = float(np.max(preds))
            label_index = int(np.argmax(preds))

            # safety check
            if label_index >= len(class_names):
                label = "Unknown"

            elif confidence < 0.70:
                label = "Unknown"

            else:
                label = class_names[label_index]

                detected_people.add(label)
                visible.add(label)
                last_visible[label] = time.time()

            # -------------------------------
            # DRAW BOX
            # -------------------------------
            color = (0, 0, 255) if label == "Unknown" else (0, 255, 0)

            emotion, emo_conf = predict_emotion(face)  # face = raw crop before resizing

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f}) [{emotion} {emo_conf:.2f}]",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # slightly smaller to fit the longer string
                color,
                2
            )

        except Exception as e:
            print("Prediction error:", e)

    # -------------------------------
    # DISPLAY LIST
    # -------------------------------
    for i, name in enumerate(visible):

        cv2.putText(
            frame,
            name,
            (50, 50 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

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