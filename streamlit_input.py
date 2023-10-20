import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load your model
CLASS_LIST = ['normal', 'bacteria', 'virus']
model = load_model(r"D:\Hackathon\PNEU_MULTICLASS\model.h5")

st.title("Pneumonia Detection App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    
    # Resize and preprocess the image
    resized_image = cv2.resize(image, (150, 150))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    rgb_image = rgb_image / 255.0
    rgb_image = np.expand_dims(rgb_image, axis=0)
    
    # Get prediction
    y_prob = model.predict(rgb_image)
    y_pred = y_prob.argmax(axis=-1)

    total = y_prob[0][1] + y_prob[0][2]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"You have a {total*100:.2f}% chance of having pneumonia.")
    st.write(f"(Bacterial: {y_prob[0][1]*100:.2f}% chance) (Viral: {y_prob[0][2]*100:.2f}% chance)")
