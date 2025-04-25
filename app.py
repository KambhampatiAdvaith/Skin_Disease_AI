import streamlit as st
from predict import predict_disease
from PIL import Image
import numpy as np

st.title("Skin Disease Detection AI")

uploaded_file = st.file_uploader("Upload an image of a skin condition", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        image_array = np.array(image)
        result = predict_disease(image_array)
        st.success(f"Prediction: {result}")
