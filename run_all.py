import time
import os
import subprocess

# --- SETTINGS ---
COUNTDOWN = 10  # seconds to get ready before recording
SIGNS = ["hello", "thanks"]  # list of signs you want to record
NUM_SAMPLES = 150
DATA_DIR = "sign_data"

# Make sure dataset folder exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- 1️⃣ Collect signs ---
for sign in SIGNS:
    print(f"\nPrepare to record sign: '{sign}'")
    print(f"Starting in {COUNTDOWN} seconds...")
    time.sleep(COUNTDOWN)

    # Run your collect_signs.py for this label
    subprocess.run(["python", "collect_signs.py"], input=f"{sign}\n{NUM_SAMPLES}\n", text=True)

print("\n[INFO] Finished collecting all signs!")

# --- 2️⃣ Train model ---
print("\n[INFO] Starting training...")
subprocess.run(["python", "train_model_dl.py"])
print("\n[INFO] Training complete!")

# --- 3️⃣ Start recognition ---
print("\n[INFO] Launching recognition...")
time.sleep(3)  # small delay before recognition
subprocess.run(["python", "recognize_signs_dl.py"])
