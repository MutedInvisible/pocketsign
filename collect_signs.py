import cv2
import mediapipe as mp
import os
import time
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- User input ---
sign_name = input("Enter the name of the sign you want to record: ").strip()
num_samples = int(input("How many samples do you want to record? "))

# --- Create folder ---
DATA_DIR = "sign_data"
sign_dir = os.path.join(DATA_DIR, sign_name)
os.makedirs(sign_dir, exist_ok=True)

# --- Countdown ---
COUNTDOWN = 10
print(f"Get ready... recording starts in {COUNTDOWN} seconds!")
time.sleep(COUNTDOWN)

# --- Start capture ---
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save landmarks as .npy
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                np.save(os.path.join(sign_dir, f"{count}.npy"), np.array(landmarks))
                count += 1
                print(f"[RECORDED] Sample {count}/{num_samples}")

        # Show frame
        cv2.putText(frame, f"Label: {sign_name} | Samples: {count}/{num_samples}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Signs", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Finished recording {count} samples for '{sign_name}'.")
