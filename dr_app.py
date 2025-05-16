import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Login Section
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":  # Replace with proper auth
            st.session_state['authenticated'] = True
        else:
            st.error("Invalid credentials")

# Main Interface
def main_interface():
    st.title("Diabetic Retinopathy Detection")
    
    # Section 1: Overview
    st.image("retinopathy_cover.jpg", use_column_width=True)  # Add image to your directory
    st.markdown("""
        ### About the Project
        Diabetic Retinopathy is a complication of diabetes that affects eyes. 
        This tool helps classify retinal images to detect the severity of the condition.
    """)
    if st.button("Start Prediction"):
        st.session_state['next_step'] = 'select_model'

    # Section 2: Accuracy Metrics
    if st.session_state.get('next_step') == 'select_model':
        st.subheader("Model Accuracy & Metrics Overview")
        st.image("accuracy table.png")  # Plot/model summary image

        model_name = st.selectbox("Select Model", ["AlexNet", "DenseNet", "Inception", "ResNet"])
        if st.button("Continue"):
            st.session_state['selected_model'] = model_name
            st.session_state['next_step'] = 'upload'

    # Section 3: Image Upload & Prediction
    if st.session_state.get('next_step') == 'upload':
        st.subheader(f"Upload Retinal Image - Model: {st.session_state['selected_model']}")
        file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if file is not None:
            img = Image.open(file)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            if st.button("Predict"):
                model = load_selected_model(st.session_state['selected_model'])
                prediction = predict_image(model, img)
                st.success(f"Predicted Severity: {prediction}")

# Helper: Load Model
def load_selected_model(name):
    path_map = {
        "AlexNet": "model_alexnet_result.h5",
        "DenseNet": "model_denseNet_result.h5",
        "Inception": "model_inception_result.h5",
        "ResNet": "model_resNet_result.h5"
    }
    return load_model(path_map[name])

# Helper: Predict Image
def predict_image(model, image):
    image = image.resize((224, 224))  # Adjust to your model’s input size
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    return class_names[np.argmax(pred)]

# Streamlit Session Manager
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login()
else:
    main_interface()