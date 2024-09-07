import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
from PIL import Image
import pandas as pd
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "Dataset")

def extract_celebrity_name(result):
    if isinstance(result, pd.DataFrame) and not result.empty:
        identity = result.iloc[0]['identity']
        parts = identity.split(os.path.sep)
        if len(parts) > 1:
            return parts[-2]
    return "Unknown"

def recognize_celebrity(image):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        recognized_celebrities = []
        
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            
            result = DeepFace.find(img_path=face_img, db_path=dataset_path, enforce_detection=False, model_name="VGG-Face")
            
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], pd.DataFrame) and not result[0].empty:
                celebrity_name = extract_celebrity_name(result[0])
                recognized_celebrities.append((celebrity_name, (x, y, w, h)))
            else:
                recognized_celebrities.append(("Unknown", (x, y, w, h)))
        
        return recognized_celebrities
    except Exception as e:
        print(f"Error in recognition: {e}")
        return [("An error occurred", (0, 0, 0, 0))]

def main():
    st.set_page_config(layout="wide")
    
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .center {
        display: flex;
        justify-content: center;
    }
    .custom-button {
        padding: 10px 20px;
        font-size: 18px;
        border-radius: 5px;
        border: none;
        color: white;
        background-color: #4CAF50;
        cursor: pointer;
        width:150px;
    }
    .custom-button:hover {
        background-color: #45a049;
    }
    .image-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .prediction-label {
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    .home-button {
        position: fixed;
        top: 20px;
        left: 20px;
        font-size: 24px;
    }
    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.get('page') in ['upload', 'webcam']:
        if st.button('🏠', key='home_button'):
            st.session_state.page = 'home'
            st.rerun()

    st.markdown('<p class="big-font center">Celebrity Recognition App</p>', unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        col1, col2, col3, col4, col5 = st.columns([42, 8, 2, 8, 40])
        with col2:
            if st.button("Upload Image"):
                st.session_state.page = "upload"
                st.rerun()
        with col4:
            if st.button("Use Webcam"):
                st.session_state.page = "webcam"
                st.rerun()

    elif st.session_state.page == "upload":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.markdown('<div class="center-button">', unsafe_allow_html=True)
            if st.button("Recognize Celebrities", key="recognize_button", use_container_width=True):
                with st.spinner("Recognizing..."):
                    recognized_celebrities = recognize_celebrity(image)
                    
                    img_with_boxes = np.array(image)
                    for celebrity, (x, y, w, h) in recognized_celebrities:
                        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(img_with_boxes, celebrity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    with col2:
                        st.image(img_with_boxes, caption="Recognized Celebrities", use_column_width=True)
                    
                    st.markdown(f'<p class="prediction-label">Recognized Celebrities: {", ".join([celeb for celeb, _ in recognized_celebrities])}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.page == "webcam":
        st.header("Webcam Celebrity Recognition")
        
        run = st.button('Start Webcam')
        FRAME_WINDOW = st.image([])
        
        camera = None
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise Exception("Could not open webcam")
            
            last_recognition_time = 0
            recognition_interval = 5  # Perform recognition every 5 seconds
            current_celebrities = []
            
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                current_time = time.time()
                if current_time - last_recognition_time > recognition_interval:
                    image = Image.fromarray(frame_rgb)
                    current_celebrities = recognize_celebrity(image)
                    last_recognition_time = current_time
                
                frame_with_labels = frame_rgb.copy()
                for celebrity, (x, y, w, h) in current_celebrities:
                    cv2.rectangle(frame_with_labels, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_with_labels, celebrity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                FRAME_WINDOW.image(frame_with_labels)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            if camera is not None:
                camera.release()

if __name__ == "__main__":
    main()