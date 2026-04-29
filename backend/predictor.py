import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

class SignPredictor:
    def __init__(self):
        # Load ASL Model
        self.asl_model = load_model("backend/models/ASL/asl_model.h5")
        self.asl_labels = np.load("backend/models/ASL/labels.npy")

        # Load ISL Model
        self.isl_model = load_model("backend/models/ISL/isl_alphabet_model.h5")
        self.isl_mean = np.load("backend/models/ISL/mean.npy")
        self.isl_std = np.load("backend/models/ISL/std.npy")
        self.isl_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

    def predict(self, frame, mode="ASL"):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, 0.0

        if mode == "ASL":
            return self._predict_asl(results.multi_hand_landmarks[0])
        else:
            return self._predict_isl(results.multi_hand_landmarks)

    def _predict_asl(self, hand_landmarks):
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
        
        data = np.array(data, dtype=np.float32)
        data = np.expand_dims(data, axis=0)

        preds = self.asl_model.predict(data, verbose=0)
        idx = np.argmax(preds)
        return self.asl_labels[idx], float(preds[0][idx])

    def _predict_isl(self, multi_hand_landmarks):
        features = []
        # Take up to 2 hands
        for hand_landmarks in multi_hand_landmarks[:2]:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

        # Pad if only one hand detected
        while len(features) < 126:
            features.append(0.0)

        data = np.array(features, dtype=np.float32)
        # Normalization
        data = (data - self.isl_mean) / (self.isl_std + 1e-6)
        data = np.expand_dims(data, axis=0)

        preds = self.isl_model.predict(data, verbose=0)
        idx = np.argmax(preds)
        return self.isl_labels[idx], float(preds[0][idx])
