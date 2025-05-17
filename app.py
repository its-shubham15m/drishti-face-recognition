import streamlit as st
import cv2
import face_recognition
import joblib
import numpy as np
from deepface import DeepFace
from PIL import Image
import io

# Load face encodings
ENCODINGS_PATH = "encodings.pkl"
data = joblib.load(ENCODINGS_PATH)
known_encodings = data["encodings"]
known_names = data["names"]

# Confidence threshold for face recognition
CONFIDENCE_THRESHOLD = 0.45  # 45%

# Function Definitions
def recognize_and_display(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []
    unknown_count = 1  # Counter for numbering unknown persons
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = f"Unknown[Person {unknown_count}]"
        confidence = 0.0
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            confidence = 1 - distances[best_match_index]
            if matches[best_match_index] and confidence >= CONFIDENCE_THRESHOLD:
                name = known_names[best_match_index]
            else:
                unknown_count += 1

        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        label = f"{name} ({confidence*100:.2f}%)" if name not in name.startswith("Unknown") else name

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append({"name": name, "confidence": confidence*100})

    return frame, results

def detect_emotion_and_display(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        results = []
        for face in result:
            x, y, w, h = face["region"]['x'], face["region"]['y'], face["region"]['w'], face["region"]['h']
            emotion = max(face["emotion"], key=face["emotion"].get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            results.append({"emotion": emotion})
        return frame, results
    except Exception as e:
        return frame, []

def combined_recognition_emotion(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []
    unknown_count = 1  # Counter for numbering unknown persons
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = f"Unknown[Person {unknown_count}]"
        confidence = 0.0
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            confidence = 1 - distances[best_match_index]
            if matches[best_match_index] and confidence >= CONFIDENCE_THRESHOLD:
                name = known_names[best_match_index]
            else:
                unknown_count += 1

        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Perform emotion detection on the face region
        face_region = frame[top:bottom, left:right]
        emotion = "Unknown"
        try:
            if face_region.size > 0:
                emotion_result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
                if emotion_result:
                    emotion = max(emotion_result[0]["emotion"], key=emotion_result[0]["emotion"].get)
        except Exception:
            pass

        # Draw rectangle and labels
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence*100:.2f}%)" if not name.startswith("Unknown") else name, 
                    (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{emotion}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append({"name": name, "confidence": confidence*100, "emotion": emotion})

    return frame, results

def process_image(image, mode):
    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    results = []
    if mode == "Identify Person":
        processed, results = recognize_and_display(frame)
    elif mode == "Read Emotions":
        processed, results = detect_emotion_and_display(frame)
    else:  # Identify & Read Emotions
        processed, results = combined_recognition_emotion(frame)
    
    # Convert back to RGB for display
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return processed_rgb, results

# Main App Logic
st.set_page_config(page_title="Real-Time Face & Emotion Detection", layout="centered", page_icon="dataset/face-id.png")
st.header("DrishtiShakti")
st.markdown("Face Recognition & Emotion Detection using AI")
st.sidebar.image("dataset/face-recognition.png", caption="Real-Time Face & Emotion Detection")

# Place "Identify & Read Emotions" first in the radio options
mode = st.sidebar.radio("Select Task", ["Identify & Read Emotions", "Identify Person", "Read Emotions"])

# Show the header once based on mode
if mode == "Identify & Read Emotions":
    st.subheader("🧐 Face Recognition & Emotion Detection Mode")
elif mode == "Identify Person":
    st.subheader("📸 Face Recognition Mode")
elif mode == "Read Emotions":
    st.subheader("😄 Emotion Detection Mode")

# Image Upload Section
st.caption("Analyze Image")
uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
image_window = st.image([])
image_results_placeholder = st.empty()

if uploaded_image is not None:
    try:
        # Read and process the uploaded image
        image = Image.open(uploaded_image)
        processed_image, results = process_image(image, mode)
        
        # Display the processed image
        image_window.image(processed_image, caption="Processed Image")
        
        # Display results below the image
        if results:
            result_text = ""
            for res in results:
                if mode == "Identify Person":
                    result_text += f"{res['name']}: {res['confidence']:.2f}%\n"
                elif mode == "Read Emotions":
                    result_text += f"Person {results.index(res)+1}: {res['emotion']}\n"
                else:
                    result_text += f"{res['name']}: {res['emotion']} ({res['confidence']:.2f}%)\n"
            image_results_placeholder.text(result_text)
        else:
            image_results_placeholder.text("No faces detected in the image.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.markdown("---")  # Separator between image and webcam sections

# Webcam Section
st.subheader("Analyze Webcam Feed")
FRAME_WINDOW = st.image([])
results_placeholder = st.empty()

video_capture = cv2.VideoCapture(0)

run = st.checkbox("Start Webcam")

while run:
    ret, frame = video_capture.read()
    if not ret:
        st.warning("Failed to grab frame.")
        break

    results = []
    if mode == "Identify Person":
        processed, results = recognize_and_display(frame)
    elif mode == "Read Emotions":
        processed, results = detect_emotion_and_display(frame)
    else:  # Identify & Read Emotions
        processed, results = combined_recognition_emotion(frame)

    # Display the processed frame
    FRAME_WINDOW.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    # Display results below the window
    if results:
        result_text = ""
        for res in results:
            if mode == "Identify Person":
                result_text += f"{res['name']}: {res['confidence']:.2f}%\n"
            elif mode == "Read Emotions":
                result_text += f"Person {results.index(res)+1}: {res['emotion']}\n"
            else:
                result_text += f"{res['name']}: {res['emotion']} ({res['confidence']:.2f}%)\n"
        results_placeholder.text(result_text)
    else:
        results_placeholder.text("No faces detected.")

video_capture.release()


st.markdown("---")
# Theme-adaptive Footer
st.markdown(
    """
    <div class="footer">
        <p>Face Recognition & Emotion Detection using AI | Version 1.0 | Powered by Streamlit & OpenCV</p>
        <p>Made with ❤️ by <i>Shubham Gupta</i></p>
        <p>Built for face recognition and emotion detection. For feedback, contact: <a href="mailto:shubhamgupta15m@gmail.com">shubhamgupta15m@gmail.com</a></p>
    </div>
    <style>
        .footer {
            text-align: center;
            text-decoration-thickness: 50%;
        }
        /* Light theme */
        @media (prefers-color-scheme: light) {
            .footer {
                color: #211951;
            }
            .footer a {
                color: #1a73e8;
            }
        /* Dark theme */
        @media (prefers-color-scheme: dark) {
            .footer {
                color: #F0F3FF;
            }
            .footer a {
                color: #8ab4f8;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)