import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("deepfake_model.h5")

# Same size used during training
IMG_SIZE = 128

# Streamlit page settings
st.set_page_config(page_title="Deepfake Detector")

st.title("Deepfake Detector")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, width=300)

    # Detect button
    if st.button("Detect"):

        # Resize image
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert to array
        img_array = image.img_to_array(img_resized)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]

        # Output
        if prediction > 0.5:
            st.success("REAL IMAGE")
        else:
            st.error("FAKE IMAGE")