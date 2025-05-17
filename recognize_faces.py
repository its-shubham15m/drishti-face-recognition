import cv2
import os
import face_recognition
import joblib
import numpy as np

# === Configuration ===
ENCODINGS_PATH = "encodings.pkl"
IP_CAMERA_URL = None  # Set to your IP camera URL or leave None to use webcam

# === Load known encodings ===
if not os.path.exists(ENCODINGS_PATH):
    print("Encodings not found. Run encode_faces.py first.")
    exit()

print("[INFO] Loading known face encodings...")
data = joblib.load(ENCODINGS_PATH)
known_encodings = data["encodings"]
known_names = data["names"]

# === Start video stream ===
print("[INFO] Starting video stream...")
video = cv2.VideoCapture(IP_CAMERA_URL if IP_CAMERA_URL else 0)

total_faces = 0
correct_matches = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names = []
    confidences = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        distances = face_recognition.face_distance(known_encodings, encoding)

        name = "Unknown"
        confidence = 0.0

        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            confidence = 1 - distances[best_match_index]  # confidence is inverse of distance

            if matches[best_match_index]:
                name = known_names[best_match_index]
                correct_matches += 1

        names.append(name)
        confidences.append(confidence)
        total_faces += 1

    # Draw boxes and names with confidence
    for (top, right, bottom, left), name, confidence in zip(face_locations, names, confidences):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        label = f"{name} ({confidence*100:.2f}%)"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition with Accuracy", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

video.release()
cv2.destroyAllWindows()

# === Final accuracy ===
if total_faces > 0:
    accuracy = (correct_matches / total_faces) * 100
    print(f"\n[RESULT] Total Faces Detected: {total_faces}")
    print(f"[RESULT] Correct Matches: {correct_matches}")
    print(f"[RESULT] Recognition Accuracy: {accuracy:.2f}%")
else:
    print("[RESULT] No faces detected.")
