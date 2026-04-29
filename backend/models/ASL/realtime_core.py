import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ===== LOAD MODEL =====
model = load_model("models/ASL/asl_model.h5")
labels = np.load("models/ASL/labels.npy")

# ===== MEDIAPIPE (SAME AS ORIGINAL PROJECT) =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

def predict(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, 0.0, frame

    hand_landmarks = results.multi_hand_landmarks[0]

    # DRAW LANDMARKS (FIXED)
    mp_draw.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )

    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])

    data = np.array(data, dtype=np.float32)
    data = np.expand_dims(data, axis=0)

    preds = model.predict(data, verbose=0)
    idx = np.argmax(preds)

    return labels[idx], float(preds[0][idx]), frame
