from azure.storage.blob import BlobServiceClient
from keras.models import load_model
import io
import streamlit as st
azure_connection_string = "DefaultEndpointsProtocol=https;AccountName=mirageye;AccountKey=jbLeD8SHPB715HwVk5Ne4Lv4FigrYwc/yuh3IwOMEW3dVkMvNAbPhucqo7AGi6i5rSUpHzPqZvak+AStjClmwQ==;EndpointSuffix=core.windows.net"
azure_container_name = "pneumoniapredict"
azure_model_blob_name = "model.h5"
azure_blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
azure_blob_client = azure_blob_service_client.get_blob_client(azure_container_name, azure_model_blob_name)
azure_model_blob = azure_blob_client.download_blob()
azure_model_bytes = azure_model_blob.readall()
model = load_model(io.BytesIO(azure_model_bytes))
st.title("Pneumonia Detection App")
import cv2
import numpy as np
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
