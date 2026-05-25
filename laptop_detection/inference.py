from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from laptop_detection.config import MODEL_SAVE_PATH, IMG_SIZE, DETECTION_THRESHOLD


# stores the result of one laptop check
@dataclass
class LaptopResult:
    detected:   bool
    confidence: float

    @property
    def label(self) -> str:
        # returns a display string
        return "LAPTOP DETECTED" if self.detected else ""

    def __str__(self) -> str:
        if self.detected:
            return f"LAPTOP DETECTED ({self.confidence:.2f})"
        return f"No laptop ({self.confidence:.2f})"


# loads the trained model once and checks each frame for a laptop
class LaptopDetector:
    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float = DETECTION_THRESHOLD,
    ):
        self.threshold = threshold
        # load model from disk on startup
        self._model = self._load_model(Path(model_path or MODEL_SAVE_PATH))

    def check(self, frame_bgr: np.ndarray) -> LaptopResult:
        # preprocess the frame and run it through the model
        tensor = self._preprocess(frame_bgr)
        prob   = float(self._model.predict(tensor, verbose=0)[0, 0])
        return LaptopResult(detected=prob >= self.threshold, confidence=prob)

    @staticmethod
    def _load_model(path: Path) -> keras.Model:
        # raise a clear error if the model file hasn't been trained yet
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at: {path}\n"
                "Train first:  python -m laptop_detection.train"
            )
        return keras.models.load_model(str(path))

    @staticmethod
    def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        # convert BGR to RGB, resize to model input size, add batch dimension
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        return np.expand_dims(resized.astype(np.float32), axis=0)
