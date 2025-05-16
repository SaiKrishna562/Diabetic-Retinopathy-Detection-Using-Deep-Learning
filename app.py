# prompt: i saved the model as model_alexnet.save('model_alexnet_result.h5')  write a code for streamlit to upload the image and predict

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the saved model
# Make sure the model file 'model_alexnet_result.h5' is accessible in your Streamlit environment.
# If you saved it in Google Drive, you might need to set up access to Drive from wherever you are running Streamlit.
# A common approach is to download the model file to your local machine or to a cloud storage that Streamlit can access.
try:
    model = load_model('model_alexnet_result.h5', compile=False) # Set compile=False because we only need to load the architecture and weights
    # Recompile the model with the custom metric
    def euclideanDist(img1, img2):
        return backend.sqrt(backend.sum((img1-img2)**2))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax', # Use a standard optimizer like adamax or adam for prediction
                  metrics=[euclideanDist, 'accuracy', tf.keras.metrics.CosineSimilarity()])

    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

st.title("Diabetic Retinopathy Prediction")

st.write("Upload an eye image to predict the stage of Diabetic Retinopathy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is not None:
        try:
            # Open the image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            # Resize the image to the target shape
            image = image.resize((224, 224))
            # Convert image to numpy array and normalize
            image_array = np.array(image) / 255.0
            # Expand dimensions to match model input shape (add batch dimension)
            image_array = np.expand_dims(image_array, axis=0)

            # Make prediction
            predictions = model.predict(image_array)

            # Get the predicted class index
            predicted_class_index = np.argmax(predictions, axis=1)[0]

            # Define the class labels based on your training data
            class_labels = {0: "No_DR",
                            1: "Mild",
                            2: "Moderate",
                            3: "Severe",
                            4: "Proliferate_DR"}

            # Get the predicted class label
            predicted_class_label = class_labels.get(predicted_class_index, "Unknown")

            st.subheader("Prediction:")
            st.write(f"The predicted stage of Diabetic Retinopathy is: **{predicted_class_label}**")

            # Optionally show prediction probabilities
            st.subheader("Prediction Probabilities:")
            probabilities = predictions[0]
            prob_df = pd.DataFrame({
                'Class': list(class_labels.values()),
                'Probability': probabilities
            })
            st.dataframe(prob_df)

        except Exception as e:
            st.error(f"Error processing image or making prediction: {e}")
    else:
        st.warning("Model could not be loaded. Please ensure 'model_alexnet_result.h5' is accessible.")
