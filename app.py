import gradio as gr
import numpy as np
from keras.models import load_model
from keras.preprocessing import image




model = load_model("deepfake_model.keras")

# Image size used during training
IMG_SIZE = 128



def predict(img):

    try:

        # Resize image
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert image to array
        img_array = image.img_to_array(img)

        # Normalize image
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]

        print("Prediction Value:", prediction)

        

        if prediction > 0.5:

            confidence = prediction * 100

            return f"""
REAL IMAGE
"""

        else:

            confidence = (1 - prediction) * 100

            return f"""
FAKE IMAGE

"""

    except Exception as e:

        return f"Error: {str(e)}"


app = gr.Interface(
    fn=predict,

    inputs=gr.Image(type="pil"),

    outputs=gr.Textbox(label="Prediction"),

    title="Deepfake Detector",

    description="Upload an image to check whether it is REAL or FAKE."
)



app.launch()