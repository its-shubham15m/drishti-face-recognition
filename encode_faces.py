import os
import cv2
import face_recognition
import joblib

# === Configuration ===
DATASET_DIR = "dataset"
ENCODINGS_PATH = "encodings.pkl"

known_encodings = []
known_names = []

# === Load images and encode ===
print("[INFO] Processing images...")

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Unable to read {img_path}")
            continue

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(rgb)

        # Compute encodings for each face found
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)
    print(f"[INFO] Completed encoding for {person_name}")

print(f"[INFO] Found {len(known_encodings)} face encodings.")

# Save encodings to file
joblib.dump({"encodings": known_encodings, "names": known_names}, ENCODINGS_PATH)
print(f"[INFO] Encodings saved to {ENCODINGS_PATH}")
