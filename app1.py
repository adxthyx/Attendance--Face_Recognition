import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
from PIL import Image
import pandas as pd
import time

def extract_celebrity_name(result):
    if isinstance(result, pd.DataFrame) and not result.empty:
        identity = result.iloc[0]['identity']
        parts = identity.split('\\')
        if len(parts) > 1:
            return parts[-2]
    return "Name not found"

def recognize_celebrity(image):
    try:
        result = DeepFace.find(img_path=image, db_path="dataset", enforce_detection=False)
        print("DeepFace result:", result)  # Debug print
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], pd.DataFrame) and not result[0].empty:
            celebrity_name = extract_celebrity_name(result[0])
            identity = result[0].iloc[0]['identity']
            return celebrity_name, identity
        else:
            return "No match found", None
    except Exception as e:
        return f"An error occurred: {str(e)}", None

def main():
    st.set_page_config(layout="wide")
    
    # Custom CSS
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
    .left-image {
        width: 640px;
        height: 480px;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    .right-image {
        width: 640px;
        height: 480px;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
        margin-left: 90px;
    }
    .left-image img, .right-image img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
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

    # Home button
    if st.session_state.get('page') in ['upload', 'webcam']:
        if st.button('🏠', key='home_button'):
            st.session_state.page = 'home'
            st.experimental_rerun()

    st.markdown('<p class="big-font center">Celebrity Recognition App</p>', unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        # Create two buttons with more precise positioning
        col1, col2, col3, col4, col5 = st.columns([42, 8, 2, 8, 40])
        with col2:
            if st.button("Upload Image"):
                st.session_state.page = "upload"
                st.experimental_rerun()
        with col4:
            if st.button("Use Webcam"):
                st.session_state.page = "webcam"
                st.experimental_rerun()

    elif st.session_state.page == "upload":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1.25])
            with col1:
                st.markdown('<div class="left-image"><img src="data:image/png;base64,{}" alt="Uploaded Image"></div>'.format(image_to_base64(image)), unsafe_allow_html=True)
            
            st.markdown('<div class="center-button">', unsafe_allow_html=True)
            if st.button("Recognize Celebrity", key="recognize_button", use_container_width=True):
                with st.spinner("Recognizing..."):
                    temp_path = "temp_upload.jpg"
                    rgb_image = image.convert('RGB')
                    rgb_image.save(temp_path)
                    
                    celebrity_name, match_path = recognize_celebrity(temp_path)
                    
                    if match_path:
                        match_image = Image.open(match_path)
                        with col2:
                            st.markdown('<div class="right-image"><img src="data:image/png;base64,{}" alt="Matching Image from Dataset"></div>'.format(image_to_base64(match_image)), unsafe_allow_html=True)
                    
                    st.markdown(f'<p class="prediction-label">Recognized Celebrity: {celebrity_name}</p>', unsafe_allow_html=True)
                    
                    os.remove(temp_path)
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.page == "webcam":
        st.header("Webcam Celebrity Recognition")
        
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        
        camera = cv2.VideoCapture(0)
        
        last_recognition_time = 0
        recognition_interval = 3  # Perform recognition every 3 seconds
        current_celebrity = "Unknown"
        
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            current_time = time.time()
            if current_time - last_recognition_time > recognition_interval:
                # Perform celebrity recognition
                temp_path = "temp_webcam.jpg"
                cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                celebrity_name, _ = recognize_celebrity(temp_path)
                current_celebrity = celebrity_name
                
                os.remove(temp_path)
                last_recognition_time = current_time
            
            # Add caption to the frame
            cv2.putText(frame, f"Recognized: {current_celebrity}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            FRAME_WINDOW.image(frame)
        
        camera.release()

def image_to_base64(image):
    import base64
    from io import BytesIO
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == "__main__":
    main()