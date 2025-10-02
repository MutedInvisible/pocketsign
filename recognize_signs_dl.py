import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- SETTINGS ---
MODEL_PATH = "sign_model_dl.h5"
ENCODER_PATH = "label_encoder.pkl"
EXPECTED_LEN = 63

# --- Load model and encoder ---
try:
    model = load_model(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
except Exception as e:
    print(f"[ERROR] Could not load model or encoder: {e}")
    exit()

# --- Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Camera ---
cap = cv2.VideoCapture(0)
print("[INFO] Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label_text = "Neutral"
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            if landmarks.shape[1] == EXPECTED_LEN:
                preds = model.predict(landmarks)
                class_id = np.argmax(preds)
                confidence = preds[0][class_id]
                label_text = le.inverse_transform([class_id])[0]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"{label_text} ({confidence*100:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
