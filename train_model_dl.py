import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- SETTINGS ---
DATA_DIR = "sign_data"  # folder where collect_signs.py saved .npy files
MODEL_PATH = "sign_model_dl.h5"
ENCODER_PATH = "label_encoder.pkl"
EXPECTED_LEN = 63  # 21 landmarks * 3

# --- Load data ---
X = []
y = []

if not os.path.exists(DATA_DIR):
    print(f"[ERROR] Data directory '{DATA_DIR}' does not exist.")
    exit()

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for file in os.listdir(label_dir):
        if file.endswith(".npy"):
            path = os.path.join(label_dir, file)
            try:
                data = np.load(path)
                if len(data) == EXPECTED_LEN:
                    X.append(data)
                    y.append(label)
            except Exception as e:
                print(f"[SKIP] {file}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("[ERROR] No valid data found to train!")
    exit()

print(f"[INFO] Loaded {len(X)} samples across {len(set(y))} classes.")

# --- Encode labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded, num_classes=len(le.classes_))

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Build model ---
model = Sequential([
    Dense(256, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --- Train ---
print("[INFO] Training model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# --- Evaluate ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Test Accuracy: {acc:.2f}")

# --- Save model and encoder ---
model.save(MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"[SAVED] Model -> {MODEL_PATH}, LabelEncoder -> {ENCODER_PATH}")
