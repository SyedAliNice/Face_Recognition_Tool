import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Face Recognition System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- STYLES ----------
def set_custom_styles():
    st.markdown("""
    <style>
    .stApp {
        background-color: #2f3640;
        color: #f5f6fa;
        font-family: 'Arial', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #1e272e;
        padding: 2rem;
        border-right: 2px solid #57606f;
    }

    h1, h2, h3, .stMarkdown, .stTextInput label, .stFileUploader label {
        color: #f5f6fa;
    }

    .stTextInput > div > input,
    .stFileUploader > div > button,
    .stButton > button {
        background-color: #3742fa;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #2f3542;
        color: #dcdde1;
    }

    img {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
        margin-top: 1rem;
    }

    .block-container {
        padding: 2rem 3rem;
    }

    .stSuccess, .stWarning, .stError {
        border-radius: 6px;
        padding: 1rem;
        font-weight: bold;
        color: white;
    }

    .stSuccess {
        background-color: #44bd32;
    }

    .stWarning {
        background-color: #e1b12c;
    }

    .stError {
        background-color: #e84118;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- FACE DETECTION SETUP ----------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
orb = cv2.ORB_create(nfeatures=1000)
DATA_FILE = "face_embeddings.pkl"

def load_features():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {"names": [], "descriptors_list": []}
    return {"names": [], "descriptors_list": []}

def save_features(names, descriptors_list):
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"names": names, "descriptors_list": descriptors_list}, f)

def get_face_features(image):
    # Use original image for extraction, and convert a copy for detection only
    rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if not results.detections:
        return None, None, "No face detected."

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w = image.shape[:2]
    x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
    face_region = cv2.cvtColor(image[max(0, y):y+height, max(0, x):x+width], cv2.COLOR_BGR2GRAY)

    if face_region.size == 0:
        return None, None, "Invalid face region."

    face_region = cv2.resize(face_region, (100, 100))
    keypoints, descriptors = orb.detectAndCompute(face_region, None)

    if descriptors is None:
        return None, None, "No features extracted."

    return keypoints, descriptors, None

def find_match(descriptors, stored_descriptors_list, names):
    if not stored_descriptors_list:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    min_avg_distance = float('inf')
    matched_name = None

    for idx, stored_descriptors in enumerate(stored_descriptors_list):
        avg_distance = 0
        num_matches = 0
        for stored_desc in stored_descriptors:
            matches = bf.match(descriptors, stored_desc)
            if matches:
                avg_distance += sum([m.distance for m in matches]) / len(matches)
                num_matches += 1
        if num_matches > 0:
            avg_distance /= num_matches
            if avg_distance < min_avg_distance and avg_distance < 60:
                min_avg_distance = avg_distance
                matched_name = names[idx]
    return matched_name, min_avg_distance if matched_name else None

# ---------- MAIN ----------
def main():
    set_custom_styles()
    st.title("Face Recognition System")

    if "data" not in st.session_state:
        st.session_state.data = load_features()

    st.sidebar.header("Add New Face")
    name = st.sidebar.text_input("Name")
    uploaded_files = st.sidebar.file_uploader("Upload up to 6 face images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if st.sidebar.button("Add Face"):
        if name and uploaded_files:
            if len(uploaded_files) > 6:
                st.sidebar.error("Max 6 images allowed.")
                return
            descriptors_for_user = []
            for uploaded_file in uploaded_files:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
                keypoints, descriptors, error = get_face_features(image)
                if error:
                    st.sidebar.error(f"{uploaded_file.name}: {error}")
                    return
                descriptors_for_user.append(descriptors)
            if name in st.session_state.data["names"]:
                st.sidebar.error("Name already exists.")
                return
            st.session_state.data["names"].append(name)
            st.session_state.data["descriptors_list"].append(descriptors_for_user)
            save_features(st.session_state.data["names"], st.session_state.data["descriptors_list"])
            st.sidebar.success(f"Added: {name}")
        else:
            st.sidebar.error("Name & image(s) required.")

    st.header("Recognize Face")
    test_image = st.file_uploader("Upload image to recognize", type=["jpg", "png", "jpeg"])
    if test_image:
        # Read the image and keep the original for display
        image_data = test_image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Convert to RGB for display in Streamlit
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Use the BGR image for processing
        keypoints, descriptors, error = get_face_features(image_bgr)
        if error:
            st.error(error)
        else:
            # Display the RGB image to preserve original colors
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
            matched_name, distance = find_match(descriptors, st.session_state.data["descriptors_list"], st.session_state.data["names"])
            if matched_name:
                st.success(f"Match Found: *{matched_name}*\nAverage Distance: {distance:.2f}")
            else:
                st.warning("No match found.")

if __name__ == "__main__":
    main()

