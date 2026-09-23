# Facial Recognition with Emotion & Liveness

A computer vision application that combines **facial recognition, emotion detection, and liveness detection** to identify registered users and reduce the risk of authentication using photos or other spoofing attempts.

The system can be used for applications such as **attendance tracking, access control, and identity verification**.

## Features

* 👤 **Face Recognition** — Identifies registered users from their facial features.
* 😊 **Emotion Detection** — Detects the user's facial emotion.
* 🛡️ **Liveness Detection** — Helps distinguish a real person from a static image.
* 📸 **Webcam Support** — Performs recognition using a live camera feed.
* 📝 **Attendance Logging** — Records recognised users and timestamps.
* ⭐ **VIP User Support** — Allows selected users to receive special handling.
* 💾 **Face Database** — Stores multiple reference images for registered users.

## System Overview

The application follows this general pipeline:

```text
                Webcam
                   │
                   ▼
            Face Detection
                   │
                   ▼
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
   Face Recognition   Emotion Detection
          │                 │
          └────────┬────────┘
                   ▼
           Liveness Detection
                   │
                   ▼
          Identity Verification
                   │
                   ▼
           Attendance Logging
```

## Project Structure

```text
Facial-Recognition-with-Emotion-and-Livenes/
│
├── classification_data/
│   ├── train_data/
│   ├── val_data/
│   └── test_data/
│
├── verification_data/
│
├── faces_db/
│   ├── Alex/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
│
├── verification_pairs_val.txt
│
├── face_classifier.keras
├── log.csv
│
├── src/
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

> The exact project structure may vary depending on the version of the application.

## Dataset

The facial recognition component uses a dataset organised by identity.

```text
classification_data/
├── train_data/
│   ├── identity_1/
│   ├── identity_2/
│   └── ...
│
├── val_data/
│   ├── identity_1/
│   ├── identity_2/
│   └── ...
│
└── test_data/
    ├── identity_1/
    ├── identity_2/
    └── ...
```

Each identity contains multiple facial images.

The verification dataset contains pairs of images used to determine whether two images belong to the same person.

```text
verification_pairs_val.txt
```

The verification labels use:

```text
1 = Same person
0 = Different people
```

## Model

The facial recognition system uses a deep learning model to convert a face image into a compact **face embedding**.

The current implementation uses **MobileNetV2** as the feature extraction backbone.

### Configuration

| Setting        |             Value |
| -------------- | ----------------: |
| Input size     |           80 × 80 |
| Backbone       |       MobileNetV2 |
| Pre-training   |          ImageNet |
| Embedding size |               128 |
| Batch size     |                64 |
| Training       | Transfer learning |

The model learns a representation where images of the same person should have similar embeddings, while images of different people should be more separated.

## Face Database

Registered users are stored inside the `faces_db` directory.

Example:

```text
faces_db/
└── Alex/
    ├── alex_1.jpg
    ├── alex_2.jpg
    ├── alex_3.jpg
    ├── alex_4.jpg
    └── alex_5.jpg
```

Multiple images can be stored for each person to improve recognition under different:

* facial expressions
* lighting conditions
* head positions
* camera angles

The application can limit the number of reference images used per person.

## Attendance System

When a registered person is successfully recognised, the application records the event in:

```text
log.csv
```

Example:

```csv
Name,Event,Time
Alex,Recognised,2026-09-24 10:30:15
Alex,Recognised,2026-09-24 10:35:21
```

The log can be used for attendance monitoring or other authentication-related events.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Alex-Phan-swin/Facial-Recognition-with-Emotion-and-Livenes.git
cd Facial-Recognition-with-Emotion-and-Livenes
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn
```

Additional dependencies may be required depending on the emotion and liveness implementation.

## Running the Application

After installing the dependencies, run the application's main script.

For example:

```bash
python main.py
```

The application will access the webcam and process the incoming frames.

The general workflow is:

1. Start the application.
2. Allow access to the webcam.
3. The camera captures a live video stream.
4. Faces are detected.
5. The system extracts facial features.
6. The identity is compared against registered users.
7. Emotion is detected.
8. Liveness checks are performed.
9. A recognised user is recorded in the attendance log.

## Adding a New User

Create a directory for the new user:

```text
faces_db/
└── NewUser/
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    └── ...
```

For better recognition, provide several images with different:

* expressions
* lighting
* orientations
* distances from the camera

Then run the face processing/training process required by
